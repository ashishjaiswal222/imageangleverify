import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Tuple
from app.verification.models_loader import get_models
from app.schemas.responses import VerificationResult
from pydantic import BaseModel
import structlog

logger = structlog.get_logger()

class IdentityConsistency(BaseModel):
    passed: bool
    face_similarity_pairs: Dict[str, float]
    clothing_consistency_score: float
    reason_code: str | None = None
    message: str | None = None

def compute_cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def get_face_embedding(image_np: np.ndarray) -> np.ndarray | None:
    models = get_models()
    # image_np is RGB from our image_io. Insightface expects BGR.
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    faces = models.face_analysis.get(img_bgr)
    if not faces:
        return None
    # Pick the largest face if multiple
    faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
    return faces[0].embedding

def get_clothing_histogram(image_np: np.ndarray) -> np.ndarray | None:
    models = get_models()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
    seg_result = models.image_segmenter.segment(mp_image)
    if not seg_result.category_mask:
        return None
        
    mask = seg_result.category_mask.numpy_view()
    # Selfie Multiclass label 4 = clothing
    clothing_mask = (mask == 4).astype(np.uint8) * 255
    
    if np.sum(clothing_mask) < 500: # Arbitrary small threshold, if almost no clothing
        return None
        
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    # Extract histogram of the H and S channels in the masked region
    hist = cv2.calcHist([hsv], [0, 1], clothing_mask, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist

def verify_identity_consistency(images_by_pos: Dict[str, np.ndarray]) -> IdentityConsistency:
    # 1. Face Embeddings (Insightface)
    # Cosine distance threshold for InsightFace ArcFace is usually around 0.6. 
    # Smaller distance = more similar.
    COSINE_DIST_THRESHOLD = 0.6 
    
    embeddings = {}
    for pos in ["front", "left", "right", "full_body"]:
        if pos in images_by_pos:
            emb = get_face_embedding(images_by_pos[pos])
            if emb is not None:
                embeddings[pos] = emb
                
    face_similarity_pairs = {}
    passed = True
    reason_code = None
    message = None
    
    # Compare all pairs of embeddings
    keys = list(embeddings.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            k1, k2 = keys[i], keys[j]
            dist = float(compute_cosine_distance(embeddings[k1], embeddings[k2]))
            pair_key = f"{k1}_{k2}"
            face_similarity_pairs[pair_key] = round(dist, 4)
            if dist > COSINE_DIST_THRESHOLD:
                passed = False
                reason_code = "IDENTITY_MISMATCH_ACROSS_PHOTOS"
                message = f"The face in your {k1.replace('_', ' ').title()} photo doesn't match your {k2.replace('_', ' ').title()} photo. Please make sure all photos are of the same person."

    # 2. Clothing Consistency for 'back'
    clothing_consistency_score = 1.0 # Default if we can't compare
    if "back" in images_by_pos and "front" in images_by_pos:
        back_hist = get_clothing_histogram(images_by_pos["back"])
        front_hist = get_clothing_histogram(images_by_pos["front"])
        
        if back_hist is not None and front_hist is not None:
            # Bhattacharyya distance: 0 is perfect match, 1 is mismatch
            dist = cv2.compareHist(front_hist, back_hist, cv2.HISTCMP_BHATTACHARYYA)
            clothing_consistency_score = round(1 - float(dist), 4) # 1.0 = match, 0.0 = mismatch
            
            # Since clothing can change or be noisy, we don't necessarily hard fail unless it's completely different
            # For this MVP, we include it as a signal but won't hard fail just on clothing unless requested.
            # If we wanted to:
            # if dist > 0.8: 
            #     passed = False
            #     reason_code = "IDENTITY_MISMATCH_ACROSS_PHOTOS"
            #     message = "Clothing from the Back View photo differs significantly from the Front View."

    return IdentityConsistency(
        passed=passed,
        face_similarity_pairs=face_similarity_pairs,
        clothing_consistency_score=clothing_consistency_score,
        reason_code=reason_code,
        message=message
    )
