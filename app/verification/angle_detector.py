import mediapipe as mp
import numpy as np
import math
from typing import Tuple, Optional
from app.verification.models_loader import get_models
from app.utils.errors import ReasonCode
from app.verification.constants import YAW_FRONT_MAX, YAW_TURN_MIN, YAW_TURN_MAX, PITCH_MAX

def get_euler_angles(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Extracts pitch, yaw, roll from a 3x3 rotation matrix.
    Returns angles in degrees.
    """
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.degrees(x), np.degrees(y), np.degrees(z)

def check_head_pose(image: mp.Image, expected_position: str) -> Tuple[bool, Optional[float], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Classifies the head pose and matches against the expected position.
    Returns: (passed, yaw_score, reason_code, message, detected_angle, expected_angle)
    """
    models = get_models()
    result = models.face_landmarker.detect(image)
    
    # If no face is detected, but position is 'back', we need to check if there's a person/hair.
    # We delegate back checks to a separate function or handle it via pipeline.
    if not result.face_landmarks or not result.facial_transformation_matrixes:
        if expected_position == "back":
            # Just verify pose has head/shoulders or segmenter shows hair
            return _check_back_view(image)
        elif expected_position in ["left", "right"]:
            return _check_side_body_view(image, expected_position)
        return False, 0.0, ReasonCode.NO_FACE_DETECTED, "No face detected.", None, expected_position

    matrix = result.facial_transformation_matrixes[0]
    R = matrix[:3, :3]
    pitch, yaw, roll = get_euler_angles(R)
    
    # Positive yaw = turned right (from subject's perspective, so camera left).
    # We will define:
    # yaw > YAW_TURN_MIN -> 'left' (subject is looking left, revealing their right side to the camera, wait...
    # Let's standardize based on testing. Usually, yaw > 0 means looking to the person's left.
    # The prompt says: "Be explicit and test carefully which sign of yaw corresponds to the person's own left vs right"
    # We will define: 
    # yaw > YAW_TURN_MIN -> 'left'
    # yaw < -YAW_TURN_MIN -> 'right'
    
    # 1. Pitch Check (Eye-level camera)
    if abs(pitch) > PITCH_MAX:
        message = "Camera is not at eye level. Please hold the camera directly in front of your face, not from above or below."
        return False, float(pitch), ReasonCode.ANGLE_MISMATCH, message, "unknown", expected_position
        
    detected_angle = "unknown"
    if abs(yaw) <= YAW_FRONT_MAX:
        detected_angle = "front"
    elif YAW_TURN_MIN <= yaw <= YAW_TURN_MAX:
        detected_angle = "right"
    elif -YAW_TURN_MAX <= yaw <= -YAW_TURN_MIN:
        detected_angle = "left"
    elif YAW_FRONT_MAX < abs(yaw) < YAW_TURN_MIN:
        detected_angle = "partial_turn"
        
    if detected_angle == expected_position:
        return True, float(yaw), None, None, detected_angle, expected_position
        
    if detected_angle == "partial_turn":
        return False, float(yaw), ReasonCode.PARTIAL_TURN, f"Please turn your head further to the side for the {expected_position.title()} View.", detected_angle, expected_position
        
    message = f"This looks like a {detected_angle.title()} View photo, but it was uploaded to the {expected_position.title()} View box. Please upload a photo turned to your {expected_position}."
    return False, float(yaw), ReasonCode.ANGLE_MISMATCH, message, detected_angle, expected_position

def _check_back_view(image: mp.Image) -> Tuple[bool, float, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Fallback logic for back view when no face landmarks are detected."""
    models = get_models()
    pose_result = models.pose_landmarker.detect(image)
    if pose_result.pose_landmarks:
        # Check if shoulders and head (nose approx) exist
        lms = pose_result.pose_landmarks[0]
        # If shoulders are visible, it's a person. We already know face isn't frontally visible.
        # So it's likely a back view.
        if lms[11].visibility > 0.5 and lms[12].visibility > 0.5:
            return True, 1.0, None, None, "back", "back"
            
    # Try segmentation model for hair
    seg_result = models.image_segmenter.segment(image)
    if seg_result.category_mask:
        mask = seg_result.category_mask.numpy_view()
        # label 1 usually hair (need to check Selfie Multiclass labels: 0: background, 1: hair, 2: body-skin, 3: face-skin, 4: clothes)
        hair_pixels = np.sum(mask == 1)
        if hair_pixels > (mask.shape[0] * mask.shape[1] * 0.05): # At least 5% hair
            return True, 1.0, None, None, "back", "back"
            
    return False, 0.0, ReasonCode.BACK_VIEW_NOT_DETECTED, "Could not detect a person from the back. Ensure your head and shoulders are visible.", "unknown", "back"

def _check_side_body_view(image: mp.Image, expected_position: str) -> Tuple[bool, float, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Fallback logic for left/right views when no face landmarks are detected, using full body pose."""
    models = get_models()
    pose_result = models.pose_landmarker.detect(image)
    if not pose_result.pose_landmarks:
        return False, 0.0, ReasonCode.NO_FACE_DETECTED, f"No person detected. Please upload a {expected_position.title()} View photo.", "unknown", expected_position
        
    lms = pose_result.pose_landmarks[0]
    nose = lms[0]
    left_ear, right_ear = lms[7], lms[8]
    left_shoulder, right_shoulder = lms[11], lms[12]
    
    # Check 1: Ear Visibility Difference
    # In a true side profile, one ear is heavily occluded by the head.
    ear_vis_diff = abs(left_ear.visibility - right_ear.visibility)
    if ear_vis_diff > 0.4:  # One ear is significantly more visible
        return True, 1.0, None, None, expected_position, expected_position
            
    # Check 2: Relative Shoulder Width
    # To avoid false positives on distant subjects, we compare shoulder width to torso height
    if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        
        # Get torso height (average shoulder to average hip)
        left_hip, right_hip = lms[23], lms[24]
        if left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
            avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            avg_hip_y = (left_hip.y + right_hip.y) / 2
            torso_height = abs(avg_hip_y - avg_shoulder_y)
            
            # If torso height is valid, check ratio. In side profile, width should be much smaller than height.
            if torso_height > 0.05:
                ratio = shoulder_width / torso_height
                if ratio < 0.4: # Shoulders are narrow relative to torso (turned sideways)
                    return True, 1.0, None, None, expected_position, expected_position
                
    return False, 0.0, ReasonCode.ANGLE_MISMATCH, f"Could not verify a side profile pose. Please upload a valid {expected_position.title()} View.", "unknown", expected_position


