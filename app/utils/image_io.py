import io
import cv2
import numpy as np
from PIL import Image, ImageOps
from app.utils.errors import VerificationError, ReasonCode
from app.verification.constants import MAX_FILE_SIZE_BYTES, MIN_RESOLUTION, ALLOWED_FORMATS

def validate_and_decode_image(file_bytes: bytes, content_type: str) -> np.ndarray:
    """
    Validates file size, format, decodes it, applies EXIF correction, 
    and returns an RGB numpy array.
    """
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise VerificationError(
            code=ReasonCode.FILE_TOO_LARGE.value,
            message=f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES / (1024*1024):.1f}MB."
        )

    if content_type not in ALLOWED_FORMATS:
        raise VerificationError(
            code=ReasonCode.UNSUPPORTED_FORMAT.value,
            message="Unsupported file format. Please upload JPG, PNG, or WEBP."
        )

    try:
        # Load image with Pillow
        pil_img = Image.open(io.BytesIO(file_bytes))
        
        # Verify it's actually an image and intact
        pil_img.verify()
        
        # Re-open because verify() can close or reset the file pointer
        pil_img = Image.open(io.BytesIO(file_bytes))
        
        # Apply EXIF orientation
        pil_img = ImageOps.exif_transpose(pil_img)
        
        # Convert to RGB (in case of RGBA or grayscale)
        pil_img = pil_img.convert("RGB")
        
        # Convert to numpy array
        np_img = np.array(pil_img)
        
        # Check resolution
        height, width = np_img.shape[:2]
        if width < MIN_RESOLUTION[0] or height < MIN_RESOLUTION[1]:
            raise VerificationError(
                code=ReasonCode.RESOLUTION_TOO_LOW.value,
                message=f"Image resolution too low. Minimum required is {MIN_RESOLUTION[0]}x{MIN_RESOLUTION[1]}."
            )
            
        return np_img

    except VerificationError:
        raise
    except Exception as e:
        raise VerificationError(
            code=ReasonCode.CORRUPT_IMAGE.value,
            message="The image file appears to be corrupted or unreadable."
        )
