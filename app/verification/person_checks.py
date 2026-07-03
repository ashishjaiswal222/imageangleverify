import mediapipe as mp
import numpy as np
import cv2
from typing import Tuple, Optional
from app.verification.models_loader import get_models
from app.utils.errors import ReasonCode
from app.verification.constants import REQUIRED_POSE_LANDMARKS, MIN_POSE_PRESENCE_SCORE, MIN_FACE_DETECTION_SCORE, MAX_FACE_EMBEDDING_NORM

def check_person_and_face(image: mp.Image, position: str) -> Tuple[bool, Optional[float], Optional[str], Optional[str]]:
    """
    Checks face count and full body pose.
    Returns: (passed, score, reason_code, message)
    """
    models = get_models()
    
    # 1. Face Detection
    # Using InsightFace (buffalo_l) instead of MediaPipe to ensure faces are REALISTIC (blocks cartoons)
    # and to perfectly detect group photos even if faces are distant.
    bgr_img = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR)
    detected_faces = models.face_analysis.get(bgr_img)
    
    # Filter out small background faces (e.g. printed on posters)
    valid_faces = []
    if detected_faces:
        # Calculate area for each face: (x2 - x1) * (y2 - y1)
        # InsightFace bbox format is [x1, y1, x2, y2]
        faces_with_area = []
        for f in detected_faces:
            if f.det_score < MIN_FACE_DETECTION_SCORE:
                continue # Skip low confidence faces like cartoons or anime
            area = (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            faces_with_area.append((f, area))
            
        # Sort by area descending (largest face first)
        if faces_with_area:
            faces_with_area.sort(key=lambda x: x[1], reverse=True)
            largest_area = faces_with_area[0][1]
            
            # Keep faces that are at least 15% the size of the largest face
            valid_faces = [f for f, area in faces_with_area if area >= largest_area * 0.15]
        
    face_count = len(valid_faces)
    
    if face_count > 1:
        return False, float(face_count), ReasonCode.GROUP_PHOTO_DETECTED, "Multiple faces detected. Please upload a solo photo."
        
    if face_count == 1:
        # --- CARTOON/ART FILTER ---
        # Real human faces map to a tight embedding manifold (norm 13.0 - 18.5)
        # Highly stylized anime/cartoons stretch this manifold (norm > 20.0)
        face_norm = np.linalg.norm(valid_faces[0].embedding)
        if face_norm > MAX_FACE_EMBEDDING_NORM:
            return False, float(face_norm), ReasonCode.CARTOON_OR_ART_DETECTED, "The detected face appears to be a cartoon, avatar, or highly stylized art. Please upload a real human photograph."
        
    if face_count == 0:
        if position in ["front", "left", "right", "full_body"]:
            return False, 0.0, ReasonCode.NO_FACE_DETECTED, "No realistic human face detected. Please ensure you are uploading a real photo and your face is clearly visible."
        # For 'back', 0 faces might be acceptable

    # 2. Full Body Pose and Mirror Selfie Checks
    pose_result = models.pose_landmarker.detect(image)
    if pose_result and pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks[0]
        
        # --- MIRROR SELFIE DETECTION ---
        # If wrist (15, 16) is visibly raised in front of the torso (above elbows or near shoulders)
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        left_elbow = landmarks[13]
        right_elbow = landmarks[14]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        # Check left wrist
        if left_wrist.presence > MIN_POSE_PRESENCE_SCORE and left_wrist.visibility > MIN_POSE_PRESENCE_SCORE:
            if left_wrist.y < left_elbow.y and left_wrist.y < left_shoulder.y + 0.1:
                return False, float(left_wrist.y), ReasonCode.MIRROR_SELFIE_DETECTED, "Hands/Arms raised. Please do not hold objects, phones, or cameras in front of you. Keep your arms relaxed."
        
        # Check right wrist
        if right_wrist.presence > MIN_POSE_PRESENCE_SCORE and right_wrist.visibility > MIN_POSE_PRESENCE_SCORE:
            if right_wrist.y < right_elbow.y and right_wrist.y < right_shoulder.y + 0.1:
                return False, float(right_wrist.y), ReasonCode.MIRROR_SELFIE_DETECTED, "Hands/Arms raised. Please do not hold objects, phones, or cameras in front of you. Keep your arms relaxed."

        # --- STRICT FULL BODY ANGLE ---
        if position == "full_body":
            missing_count = 0
            min_score_found = 1.0
            
            for idx in REQUIRED_POSE_LANDMARKS:
                lm = landmarks[idx]
                if lm.visibility < MIN_POSE_PRESENCE_SCORE or lm.presence < MIN_POSE_PRESENCE_SCORE:
                    missing_count += 1
                else:
                    if not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0):
                        missing_count += 1
                min_score_found = min(min_score_found, lm.visibility)

            if missing_count > 2:
                return False, float(min_score_found), ReasonCode.NOT_FULL_BODY, "Full body not visible. Ensure your head, shoulders, hips, knees, and feet are in the frame."
                
            # Verify they are facing completely FORWARD by checking depth (z) distance between shoulders
            # If z-distance is very high, they are angled sideways.
            # MediaPipe z is roughly proportional to shoulder width depth.
            depth_diff = abs(left_shoulder.z - right_shoulder.z)
            horizontal_diff = abs(left_shoulder.x - right_shoulder.x)
            
            # If depth difference is too high OR horizontal shoulder width is too small (meaning they are turned sideways)
            if depth_diff > 0.35 or horizontal_diff < 0.08:
                return False, float(depth_diff), ReasonCode.SIDE_PROFILE_FULL_BODY, "Full body photo must be strictly facing the front, not angled sideways."
    else:
        # If there are no pose landmarks at all, and they requested full_body, fail it.
        if position == "full_body":
            return False, 0.0, ReasonCode.NOT_FULL_BODY, "Could not detect a person's pose. Please ensure your full body is in the frame."

    # Back position is handled entirely differently in pipeline or angle_detector
    
    return True, float(face_count), None, None
