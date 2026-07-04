import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional
from app.utils.errors import ReasonCode
from app.verification.constants import (
    LAPLACIAN_VAR_THRESHOLD, MIN_BRIGHTNESS, MAX_BRIGHTNESS, FACE_CENTER_TOLERANCE_PCT
)
from app.verification.models_loader import get_models

def check_blur(image_np: np.ndarray) -> Tuple[bool, float, Optional[str], Optional[str]]:
    """Checks if the image is too blurry using Laplacian variance."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if variance < LAPLACIAN_VAR_THRESHOLD:
        return False, float(variance), ReasonCode.IMAGE_BLURRY, "Photo appears blurry or low resolution. Please retake in good, steady light."
    return True, float(variance), None, None

def check_lighting(image_np: np.ndarray) -> Tuple[bool, float, Optional[str], Optional[str]]:
    """Checks if the image is under-exposed or over-exposed."""
    # Convert to HSV and use V channel for brightness
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    mean_brightness = np.mean(v_channel)
    
    if mean_brightness < MIN_BRIGHTNESS:
        return False, float(mean_brightness), ReasonCode.POOR_LIGHTING, "Photo is too dark. Please move to a better lit area."
    elif mean_brightness > MAX_BRIGHTNESS:
        return False, float(mean_brightness), ReasonCode.POOR_LIGHTING, "Photo is too bright or washed out. Please avoid standing directly in front of strong light sources."
        
    return True, float(mean_brightness), None, None

def check_uneven_lighting(image: mp.Image, image_np: np.ndarray) -> Tuple[bool, Optional[float], Optional[str], Optional[str]]:
    """Checks for harsh shadows/uneven lighting across the face by comparing left vs right cheek."""
    models = get_models()
    result = models.face_landmarker.detect(image)
    if not result.face_landmarks:
        return True, None, None, None
        
    lms = result.face_landmarks[0]
    h, w, _ = image_np.shape
    
    # Left cheek roughly around landmark 50
    # Right cheek roughly around landmark 280
    left_cheek_x = int(lms[50].x * w)
    left_cheek_y = int(lms[50].y * h)
    
    right_cheek_x = int(lms[280].x * w)
    right_cheek_y = int(lms[280].y * h)
    
    patch_size = int(max(w, h) * 0.02) # 2% of image size for patch
    if patch_size < 5: patch_size = 5
    
    def get_patch_brightness(cx, cy):
        x1, y1 = max(0, cx - patch_size), max(0, cy - patch_size)
        x2, y2 = min(w, cx + patch_size), min(h, cy + patch_size)
        patch = image_np[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        return np.mean(gray)
        
    left_b = get_patch_brightness(left_cheek_x, left_cheek_y)
    right_b = get_patch_brightness(right_cheek_x, right_cheek_y)
    
    if left_b is None or right_b is None:
        return True, None, None, None
        
    max_b = max(left_b, right_b) + 1
    min_b = min(left_b, right_b) + 1
    
    ratio = min_b / max_b
    
    # If one cheek is drastically darker (less than 40% brightness of the other)
    if ratio < 0.40:
        return False, float(ratio), ReasonCode.UNEVEN_LIGHTING, "Harsh shadows detected on the face. Please stand in even, natural lighting."
        
    return True, float(ratio), None, None

def check_face_centering(image: mp.Image, image_np: np.ndarray) -> Tuple[bool, Optional[float], Optional[str], Optional[str]]:
    """Checks if the detected face is reasonably centered in the frame."""
    models = get_models()
    result = models.face_detector.detect(image)
    
    if not result.detections:
        return True, None, None, None # Handled by person_checks
        
    bbox = result.detections[0].bounding_box
    
    # Bounding box coordinates from mediapipe are absolute in pixels
    # Wait, MediaPipe FaceDetector returns absolute or relative? 
    # Usually relative in normalized coordinates [0, 1] for FaceDetector, wait, let's check.
    # Ah, `mp.tasks.components.containers.BoundingBox` has `origin_x`, `origin_y`, `width`, `height` in pixels.
    
    img_height, img_width = image_np.shape[:2]
    center_x = bbox.origin_x + (bbox.width / 2)
    center_y = bbox.origin_y + (bbox.height / 2)
    
    norm_cx = center_x / img_width
    norm_cy = center_y / img_height
    
    # Perfect center is (0.5, 0.5)
    dist_x = abs(0.5 - norm_cx)
    dist_y = abs(0.5 - norm_cy)
    
    # We allow FACE_CENTER_TOLERANCE_PCT deviation
    if dist_x > FACE_CENTER_TOLERANCE_PCT or dist_y > FACE_CENTER_TOLERANCE_PCT:
        return False, float(max(dist_x, dist_y)), ReasonCode.FACE_NOT_CENTERED, "Face is not centered. Please ensure your face is in the middle of the frame."
        
    return True, float(max(dist_x, dist_y)), None, None

def check_explicit_content(image_np: np.ndarray) -> Tuple[bool, Optional[float], Optional[str], Optional[str]]:
    """Checks the image for explicit or NSFW content using NudeNet."""
    models = get_models()
    
    # NudeNet expects an OpenCV BGR image or a path. image_np is RGB.
    bgr_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Run inference
    detections = models.nude_detector.detect(bgr_img)
    
    blocked_classes = {
        "FEMALE_GENITALIA_EXPOSED", 
        "MALE_GENITALIA_EXPOSED", 
        "FEMALE_BREAST_EXPOSED", 
        "BUTTOCKS_EXPOSED", 
        "ANUS_EXPOSED"
        # MALE_BREAST_EXPOSED is explicitly allowed
    }
    
    max_score = 0.0
    for det in detections:
        class_name = det.get("class")
        score = det.get("score")
        if class_name in blocked_classes and score > 0.45:
            max_score = max(max_score, score)
            
    if max_score > 0.45:
        return False, float(max_score), ReasonCode.EXPLICIT_CONTENT_DETECTED, "Explicit or inappropriate content detected. Please upload an appropriate photo."
        
    return True, 0.0, None, None
