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

class _DummyLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

def _get_robust_landmarks(image_np: np.ndarray, mp_image: mp.Image):
    """
    Attempts to get Face Landmarks. If MediaPipe fails due to small/distant faces,
    uses InsightFace to crop the face and retries MediaPipe on the crop, then
    translates the landmarks back to the original image coordinates.
    """
    models = get_models()
    result = models.face_landmarker.detect(mp_image)
    
    if result.face_landmarks:
        return result.face_landmarks[0]
        
    # Fallback for distant/full_body faces where MediaPipe fails
    bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    faces = models.face_analysis.get(bgr)
    if not faces:
        return None
        
    face = faces[0]
    h, w, _ = image_np.shape
    x1, y1, x2, y2 = map(int, face.bbox)
    
    # Expand crop slightly
    margin = int(max(x2-x1, y2-y1) * 0.2)
    crop_x1 = max(0, x1 - margin)
    crop_y1 = max(0, y1 - margin)
    crop_x2 = min(w, x2 + margin)
    crop_y2 = min(h, y2 + margin)
    
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None
        
    crop = np.ascontiguousarray(image_np[crop_y1:crop_y2, crop_x1:crop_x2])
    if crop.size == 0:
        return None
        
    crop_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
    crop_result = models.face_landmarker.detect(crop_mp)
    
    if not crop_result.face_landmarks:
        return None
        
    # Translate normalized crop coordinates back to full image normalized coordinates
    crop_landmarks = crop_result.face_landmarks[0]
    translated_landmarks = []
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    
    for lm in crop_landmarks:
        tx = (crop_x1 + (lm.x * crop_w)) / w
        ty = (crop_y1 + (lm.y * crop_h)) / h
        translated_landmarks.append(_DummyLandmark(tx, ty, lm.z))
        
    return translated_landmarks

def _get_insightface_bbox(image_np: np.ndarray):
    models = get_models()
    bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    faces = models.face_analysis.get(bgr)
    if not faces:
        return None
    faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
    return faces[0].bbox

def check_neutral_expression(image: mp.Image, image_np: np.ndarray) -> Tuple[bool, Optional[float], Optional[str], Optional[str]]:
    """
    Ensures the user has a neutral expression (no big smiles, open mouths, or winking).
    Uses MediaPipe FaceLandmarker blendshapes.
    """
    models = get_models()
    result = models.face_landmarker.detect(image)
    
    if not result.face_blendshapes:
        return True, None, None, None
        
    blendshapes = result.face_blendshapes[0]
    # Convert blendshapes to a dictionary for easy access
    bs_dict = {category.category_name: category.score for category in blendshapes}
    
    jaw_open = bs_dict.get('jawOpen', 0.0)
    smile_left = bs_dict.get('mouthSmileLeft', 0.0)
    smile_right = bs_dict.get('mouthSmileRight', 0.0)
    
    # 1. Check for yawning/shouting (Mouth wide open)
    if jaw_open > 0.3:
        return False, float(jaw_open), ReasonCode.EXPRESSION_NOT_NEUTRAL, "Mouth is open. Please keep a neutral expression with your mouth closed."
        
    # 2. Check for big smiles
    if smile_left > 0.5 or smile_right > 0.5:
        smile_score = max(smile_left, smile_right)
        return False, float(smile_score), ReasonCode.EXPRESSION_NOT_NEUTRAL, "Exaggerated smile detected. Please keep a neutral expression."
        
    # 3. Check for winking / asymmetrical blink
    blink_left = bs_dict.get('eyeBlinkLeft', 0.0)
    blink_right = bs_dict.get('eyeBlinkRight', 0.0)
    
    if abs(blink_left - blink_right) > 0.4:
        return False, float(abs(blink_left - blink_right)), ReasonCode.EXPRESSION_NOT_NEUTRAL, "Winking or asymmetrical eyes detected. Please keep both eyes open evenly."
        
    return True, 0.0, None, None

def check_eyes_open(image: mp.Image, image_np: np.ndarray) -> Tuple[bool, Optional[float], Optional[str], Optional[str]]:
    """Uses Eye Aspect Ratio (EAR) to determine if eyes are open."""
    landmarks = _get_robust_landmarks(image_np, image)
    
    if not landmarks:
        return True, None, None, None # Handled elsewhere
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
    lms = _get_robust_landmarks(image_np, image)
    h, w, _ = image_np.shape
    
    # --- Segmenter Check (Always Run) ---
    models = get_models()
    bbox = _get_insightface_bbox(image_np)
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        seg = models.image_segmenter.segment(image)
        if seg and seg.category_mask:
            mask = seg.category_mask.numpy_view()[y1:y2, x1:x2]
            if mask.size > 0:
                acc_ratio = np.sum(mask == 5) / mask.size
                if acc_ratio > 0.10: # >10% of face is accessories (glasses/phone)
                    return False, float(acc_ratio), ReasonCode.EYEWEAR_DETECTED, "Please remove sunglasses, glasses, or objects blocking your face."
    
    if not lms:
        return True, None, None, None
        
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
    
    # Extract approximate face crop for dynamic contrast
    face_top = int(lms[10].y * h)
    face_bot = int(lms[152].y * h)
    face_left = int(lms[234].x * w)
    face_right = int(lms[454].x * w)
    
    face_crop = image_np[max(0, face_top):min(h, face_bot), max(0, face_left):min(w, face_right)]
    if face_crop.size == 0:
        return True, None, None, None
        
    face_gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
    eye_mean = np.mean(gray_crop)
    face_mean = np.mean(face_gray)
    
    # 1. Absolute Check: Are the eyes pitch black?
    dark_pixels = np.sum(gray_crop < 40)
    total_pixels = gray_crop.shape[0] * gray_crop.shape[1]
    dark_ratio = dark_pixels / total_pixels
    
    # 2. Dynamic Contrast: Are the eyes drastically darker than the face?
    contrast_ratio = eye_mean / (face_mean + 1e-6)
    
    # If the eye area is overwhelmingly pitch black OR the eyes are drastically darker than the skin
    if dark_ratio > 0.40 or contrast_ratio < 0.60:
        return False, float(max(dark_ratio, 1.0 - contrast_ratio)), ReasonCode.EYEWEAR_DETECTED, "Please remove sunglasses/eyeglasses and retake the photo."
        
    return True, float(dark_ratio), None, None

def check_face_coverage(image: mp.Image, image_np: np.ndarray) -> Tuple[bool, Optional[float], Optional[str], Optional[str]]:
    """
    Checks if the forehead/hairline region is covered by something other than hair/skin.
    Uses Selfie Multiclass Segmentation.
    Classes: 0-background, 1-hair, 2-body-skin, 3-face-skin, 4-clothes, 5-others
    """
    models = get_models()
    seg_result = models.image_segmenter.segment(image)
    lms = _get_robust_landmarks(image_np, image)
    h, w, _ = image_np.shape
    
    if not seg_result.category_mask:
        return True, None, None, None
        
    mask = seg_result.category_mask.numpy_view()
        
    # --- Universal Occlusion Check (Always Run) ---
    bbox = _get_insightface_bbox(image_np)
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        face_mask = mask[y1:y2, x1:x2]
        
        if face_mask.size > 0:
            # Class 4 (clothes), 5 (accessories), 2 (body skin/hands blocking face)
            occlusion_pixels = np.sum(face_mask == 4) + np.sum(face_mask == 5) + np.sum(face_mask == 2)
            occlusion_ratio = occlusion_pixels / face_mask.size
            # Require a higher threshold so it doesn't trigger just for wearing glasses (handled by check_eyewear)
            # or having hands near chin. But holding a phone will trigger this.
            if occlusion_ratio > 0.20:
                return False, float(occlusion_ratio), ReasonCode.FACE_PARTIALLY_COVERED, "Face partially covered by objects, hands, or phones. Please ensure your full face is visible."

    if not lms:
        return True, None, None, None
        
    mask = seg_result.category_mask.numpy_view()
    h, w = mask.shape[:2]
    
    # Define hairline zone: above eyebrows (e.g. landmarks 65, 295) 
    # up to the top of the head (landmark 10)
    
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
