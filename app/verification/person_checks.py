import mediapipe as mp
import numpy as np
from typing import Tuple, Optional
from app.verification.models_loader import get_models
from app.utils.errors import ReasonCode
from app.verification.constants import REQUIRED_POSE_LANDMARKS, MIN_POSE_PRESENCE_SCORE

def check_person_and_face(image: mp.Image, position: str) -> Tuple[bool, Optional[float], Optional[str], Optional[str]]:
    """
    Checks face count and full body pose.
    Returns: (passed, score, reason_code, message)
    """
    models = get_models()
    
    # 1. Face Detection
    face_result = models.face_detector.detect(image)
    face_count = len(face_result.detections) if face_result.detections else 0
    
    if face_count > 1:
        return False, float(face_count), ReasonCode.GROUP_PHOTO_DETECTED, "Only one person should be in the frame. Please upload a solo photo."
        
    if face_count == 0:
        if position in ["front", "left", "right"]:
            return False, 0.0, ReasonCode.NO_FACE_DETECTED, "No face detected in the photo. Please ensure your face is clearly visible."
        # For 'full_body' and 'back', 0 faces might be acceptable (soft warn or checked differently)

    # 2. Full Body Pose Check (only for full_body)
    if position == "full_body":
        pose_result = models.pose_landmarker.detect(image)
        if not pose_result.pose_landmarks:
            return False, 0.0, ReasonCode.NOT_FULL_BODY, "Could not detect a person's pose. Please ensure your full body is in the frame."
            
        landmarks = pose_result.pose_landmarks[0]
        missing_count = 0
        min_score_found = 1.0
        
        for idx in REQUIRED_POSE_LANDMARKS:
            lm = landmarks[idx]
            if lm.visibility < MIN_POSE_PRESENCE_SCORE or lm.presence < MIN_POSE_PRESENCE_SCORE:
                missing_count += 1
            else:
                # Also check if it's within frame bounds (x, y between 0 and 1)
                if not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0):
                    missing_count += 1
            
            min_score_found = min(min_score_found, lm.visibility)

        # Allow slight leniency (e.g., if ankles are just cut off), but generally expect most landmarks
        # If too many are missing, fail it. Let's require at least 7 out of the 9 required landmarks.
        if missing_count > 2:
            return False, float(min_score_found), ReasonCode.NOT_FULL_BODY, "Full body not visible. Ensure your head, shoulders, hips, knees, and feet are in the frame."

    # Back position is handled entirely differently in pipeline or angle_detector
    
    return True, float(face_count), None, None
