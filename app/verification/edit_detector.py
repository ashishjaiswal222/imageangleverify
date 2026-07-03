import cv2
import numpy as np
from typing import Tuple, Optional
from app.utils.errors import ReasonCode
from app.verification.constants import ELA_JPEG_QUALITY, ELA_VARIANCE_THRESHOLD

def check_heavy_editing(image_np: np.ndarray) -> Tuple[bool, float, Optional[str], Optional[str]]:
    """
    Uses Error Level Analysis (ELA) to detect heavily filtered or edited photos.
    Resaves the image at a known JPEG quality and compares it to the original.
    High variance in the difference image can indicate manipulation.
    """
    # ELA works best if we use the original image, but since we receive it as an array,
    # we simulate the first compression if it was e.g. a PNG or already a JPEG.
    
    # Encode to JPEG
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), ELA_JPEG_QUALITY]
    _, encoded_img = cv2.imencode('.jpg', image_np, encode_param)
    
    # Decode back
    decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    
    # Compute absolute difference
    # Ensure types are int16 to prevent overflow during subtraction before abs
    img1 = image_np.astype(np.int16)
    img2 = decoded_img.astype(np.int16)
    
    diff = np.abs(img1 - img2)
    
    # The variance of this difference map
    ela_variance = np.var(diff)
    
    # heavily edited/smoothed images tend to have unusually low ELA variance in skin regions,
    # or unusually high variance if text/sharp noise was added.
    # A simple threshold is used here, but in production, this requires calibration.
    # We will flag if variance is unusually high (indicating re-compression artifacts mismatch).
    # Since the prompt says: "Heavily/uniformly edited or over-processed images tend to show unusually uniform or unusually high error levels"
    
    if ela_variance > ELA_VARIANCE_THRESHOLD:
        return False, float(ela_variance), ReasonCode.HEAVY_EDITING_DETECTED, "Image appears heavily edited or over-filtered. Please upload an original, unfiltered photo."
        
    return True, float(ela_variance), None, None
