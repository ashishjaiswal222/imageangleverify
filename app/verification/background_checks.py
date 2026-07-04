import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional
from app.utils.errors import ReasonCode
from app.verification.models_loader import get_models

def check_background_clutter(image: mp.Image, image_np: np.ndarray) -> Tuple[bool, Optional[float], Optional[str], Optional[str]]:
    """
    Checks if the background is a plain wall or a cluttered environment.
    Uses MediaPipe ImageSegmenter to isolate the background, then runs an edge detector.
    """
    models = get_models()
    result = models.image_segmenter.segment(image)
    
    if not result.category_mask:
        return True, None, None, None
        
    mask_np = result.category_mask.numpy_view()
    
    # In selfie multiclass, 0 is background
    bg_mask = (mask_np == 0).astype(np.uint8)
    bg_pixels = np.sum(bg_mask)
    
    # If there is basically no background in the frame, pass
    if bg_pixels < (image_np.shape[0] * image_np.shape[1] * 0.05):
        return True, 0.0, None, None
        
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Blur slightly to remove noise/textures (like a slightly bumpy wall)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Run Canny edge detector
    edges = cv2.Canny(blurred, 30, 100)
    
    # Mask the edges to only include the background area
    bg_edges = cv2.bitwise_and(edges, edges, mask=bg_mask)
    
    edge_pixels = np.count_nonzero(bg_edges)
    edge_density = edge_pixels / bg_pixels
    
    # If more than 3% of the background consists of sharp edges, it's cluttered
    if edge_density > 0.03:
        return False, float(edge_density), ReasonCode.BACKGROUND_CLUTTERED, "Background is too busy or cluttered. Please stand in front of a plain wall."
        
    return True, float(edge_density), None, None
