import mediapipe as mp
import numpy as np
import cv2
import math
from typing import Tuple, Optional
from app.verification.models_loader import get_models
from app.utils.errors import ReasonCode
from app.verification.constants import EAR_THRESHOLD

# Left eye indices (Top, Bottom, Left, Right)
LEFT_EYE = [159, 145, 33, 133]
# Right eye indices
RIGHT_EYE = [386, 374, 362, 263]

def _distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def _get_ear(landmarks, eye_indices):
    top = landmarks[eye_indices[0]]
    bottom = landmarks[eye_indices[1]]
    left = landmarks[eye_indices[2]]
    right = landmarks[eye_indices[3]]
    
    vertical_dist = _distance(top, bottom)
    horizontal_dist = _distance(left, right)
    
    if horizontal_dist == 0:
        return 0.0
    return vertical_dist / horizontal_dist

def check_eyes_open(image: mp.Image) -> Tuple[bool, Optional[float], Optional[str], Optional[str]]:
    """Uses Eye Aspect Ratio (EAR) to determine if eyes are open."""
    models = get_models()
    result = models.face_landmarker.detect(image)
    
    if not result.face_landmarks:
        return True, None, None, None # Handled elsewhere
        
    landmarks = result.face_landmarks[0]
    left_ear = _get_ear(landmarks, LEFT_EYE)
    right_ear = _get_ear(landmarks, RIGHT_EYE)
    
    min_ear = min(left_ear, right_ear)
    if min_ear < EAR_THRESHOLD:
        return False, float(min_ear), ReasonCode.EYES_CLOSED, "Eyes appear closed. Please ensure your eyes are open and looking at the camera."
        
    return True, float(min_ear), None, None

def check_eyewear(image: mp.Image, image_np: np.ndarray) -> Tuple[bool, Optional[float], Optional[str], Optional[str]]:
    """
    Checks for sunglasses/eyewear.
    MediaPipe FaceLandmarker with blendshapes can indicate presence of glasses or 
    we can use the Iris landmarks. If iris is absent/low confidence, likely occluded.
    Since blendshapes don't explicitly give 'glasses', we'll rely on iris or crop contrast.
    For simplicity and speed, we will use a contrast heuristic on the eye bounding box.
    """
    models = get_models()
    result = models.face_landmarker.detect(image)
    
    if not result.face_landmarks:
        return True, None, None, None
        
    lms = result.face_landmarks[0]
    h, w, _ = image_np.shape
    
    # Define an ROI around the eyes
    left_x = int(lms[33].x * w)
    right_x = int(lms[263].x * w)
    top_y = int(min(lms[159].y, lms[386].y) * h)
    bottom_y = int(max(lms[145].y, lms[374].y) * h)
    
    # Expand ROI slightly
    margin = int(abs(right_x - left_x) * 0.2)
    left_x = max(0, left_x - margin)
    right_x = min(w, right_x + margin)
    top_y = max(0, top_y - margin)
    bottom_y = min(h, bottom_y + margin)
    
    if right_x <= left_x or bottom_y <= top_y:
        return True, None, None, None
        
    eye_crop = image_np[top_y:bottom_y, left_x:right_x]
    if eye_crop.size == 0:
        return True, None, None, None
        
    gray_crop = cv2.cvtColor(eye_crop, cv2.COLOR_RGB2GRAY)
    
    # Sunglasses are typically very dark. 
    # Check if a large portion of the eye area is extremely dark.
    dark_pixels = np.sum(gray_crop < 40)
    total_pixels = gray_crop.shape[0] * gray_crop.shape[1]
    dark_ratio = dark_pixels / total_pixels
    
    if dark_ratio > 0.4: # If > 40% of eye region is very dark
        return False, float(dark_ratio), ReasonCode.EYEWEAR_DETECTED, "Please remove sunglasses/eyeglasses and retake the photo."
        
    return True, float(dark_ratio), None, None

def check_face_coverage(image: mp.Image, image_np: np.ndarray) -> Tuple[bool, Optional[float], Optional[str], Optional[str]]:
    """
    Checks if the forehead/hairline region is covered by something other than hair/skin.
    Uses Selfie Multiclass Segmentation.
    Classes: 0-background, 1-hair, 2-body-skin, 3-face-skin, 4-clothes, 5-others
    """
    models = get_models()
    seg_result = models.image_segmenter.segment(image)
    face_result = models.face_landmarker.detect(image)
    
    if not seg_result.category_mask or not face_result.face_landmarks:
        return True, None, None, None
        
    mask = seg_result.category_mask.numpy_view()
    h, w = mask.shape[:2]
    
    # Define hairline zone: above eyebrows (e.g. landmarks 65, 295) 
    # up to the top of the head (landmark 10)
    lms = face_result.face_landmarks[0]
    
    top_y = max(0, int(lms[10].y * h) - int(0.05 * h)) # Slightly above top landmark
    bottom_y = int(min(lms[65].y, lms[295].y) * h)
    left_x = int(min(lms[103].x, lms[332].x) * w) # Sides of upper forehead
    right_x = int(max(lms[103].x, lms[332].x) * w)
    
    # Safety checks
    if top_y >= bottom_y or left_x >= right_x or bottom_y <= 0 or right_x <= 0:
        return True, None, None, None
        
    # Crop the mask
    forehead_mask = mask[top_y:bottom_y, left_x:right_x]
    if forehead_mask.size == 0:
        return True, None, None, None
        
    total_pixels = forehead_mask.shape[0] * forehead_mask.shape[1]
    clothes_pixels = np.sum(forehead_mask == 4) + np.sum(forehead_mask == 5)
    
    clothes_ratio = clothes_pixels / total_pixels
    
    if clothes_ratio > 0.5: # If > 50% of forehead zone is clothing/accessories
        return False, float(clothes_ratio), ReasonCode.FACE_PARTIALLY_COVERED, "Face partially covered. Please ensure your full face and hairline are visible."
        
    return True, float(clothes_ratio), None, None
