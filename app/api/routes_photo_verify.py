import time
import asyncio
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from pydantic import ValidationError

from app.schemas.responses import VerificationResult, BatchResults, IdentityConsistency
from app.verification.pipeline import verify_image
from app.verification.identity_checks import verify_identity_consistency, get_face_embedding, compute_cosine_distance
from app.workers.pool import get_pool
from app.config import settings
from app.utils.errors import VerificationError, ReasonCode
from app.utils.image_io import validate_and_decode_image
import structlog

logger = structlog.get_logger()
router = APIRouter()

ALLOWED_POSITIONS = {"full_body", "front", "left", "right", "back"}

@router.post("/check-single", response_model=VerificationResult)
async def check_single(
    file: UploadFile = File(...),
    position: str = Form(...)
):
    if position not in {"front", "full_body"}:
        raise HTTPException(status_code=422, detail="Invalid position for single check. Must be 'front' or 'full_body'")

    file_bytes = await file.read()
    content_type = file.content_type

    pool = get_pool()
    loop = asyncio.get_running_loop()

    try:
        # Run CPU-bound task in ProcessPoolExecutor
        result: VerificationResult = await asyncio.wait_for(
            loop.run_in_executor(pool, verify_image, file_bytes, content_type, position),
            timeout=settings.batch_timeout_seconds
        )
        
        return result
    except VerificationError as e:
        raise HTTPException(status_code=422, detail={"msg": e.message, "type": "value_error", "loc": ["body", "file"]})
    except asyncio.TimeoutError:
        logger.error("check_single_timeout", position=position)
        raise HTTPException(status_code=504, detail="Request timed out during processing.")
    except Exception as e:
        logger.exception("check_single_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/check-batch", response_model=BatchResults)
async def check_batch(
    session_id: Optional[str] = Form(None),
    full_body: Optional[UploadFile] = File(None),
    front: Optional[UploadFile] = File(None),
    left: Optional[UploadFile] = File(None),
    right: Optional[UploadFile] = File(None),
    back: Optional[UploadFile] = File(None),
):
    start_time = time.time()
    pool = get_pool()
    loop = asyncio.get_running_loop()
    
    files = {
        "full_body": full_body,
        "front": front,
        "left": left,
        "right": right,
        "back": back
    }
    
    tasks = {}
    results = {}
    
    # Read files concurrently into memory
    for pos, file in files.items():
        if file is not None:
            file_bytes = await file.read()
            tasks[pos] = loop.run_in_executor(pool, verify_image, file_bytes, file.content_type, pos)
        else:
            results[pos] = {"status": "MISSING", "message": f"{pos.replace('_', ' ').title()} image was not provided."}
            
    # Execute verification in parallel via ProcessPoolExecutor
    if tasks:
        try:
            done, pending = await asyncio.wait(
                tasks.values(),
                timeout=settings.batch_timeout_seconds,
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Map back to positions
            task_to_pos = {t: pos for pos, t in tasks.items()}
            
            for t in done:
                pos = task_to_pos[t]
                try:
                    res = t.result()
                    results[pos] = res
                except VerificationError as e:
                    # Treat malformed files as missing/error in batch or specific reason
                    results[pos] = {"status": "ERROR", "message": e.message}
                except Exception as e:
                    logger.exception("batch_item_error", position=pos, error=str(e))
                    results[pos] = {"status": "ERROR", "message": "Failed to process this image."}
                    
            for t in pending:
                pos = task_to_pos[t]
                results[pos] = {"status": "TIMEOUT", "message": "Processing timed out."}
                t.cancel()
                
        except Exception as e:
            logger.exception("batch_execution_error", error=str(e))
            raise HTTPException(status_code=500, detail="Error executing batch verification")
            
    # Calculate overall passed and collect valid images for identity check
    overall_passed = True
    valid_images_for_identity = {}
    
    for pos in ALLOWED_POSITIONS:
        res = results.get(pos)
        if isinstance(res, VerificationResult):
            if not res.passed:
                overall_passed = False
            else:
                # We need the numpy array of the image. Wait, the result doesn't return the numpy array to save memory.
                # I should decode it again here, or wait, it's just bytes. I can use image_io.
                from app.utils.image_io import validate_and_decode_image
                file = files[pos]
                if file is not None:
                    try:
                        file.file.seek(0)
                        img_bytes = await file.read()
                        valid_images_for_identity[pos] = validate_and_decode_image(img_bytes, file.content_type)
                    except Exception as e:
                        logger.error(f"Failed to re-decode {pos} for identity check: {e}")
        else:
            # Missing or error -> overall fails
            overall_passed = False

    identity_consistency = None
    # Only check identity if we have multiple valid images
    if overall_passed and len(valid_images_for_identity) > 1:
        try:
            # Identity check can be run in executor as well
            identity_result = await loop.run_in_executor(
                pool, 
                verify_identity_consistency, 
                valid_images_for_identity
            )
            # Re-map Pydantic model
            identity_consistency = IdentityConsistency(**identity_result.model_dump())
            if not identity_consistency.passed:
                overall_passed = False
        except Exception as e:
            logger.exception("identity_check_error", error=str(e))
            overall_passed = False
            identity_consistency = IdentityConsistency(
                passed=False, 
                face_similarity_pairs={}, 
                clothing_consistency_score=0.0,
                reason_code="IDENTITY_CHECK_FAILED", 
                message="Failed to verify identity across photos."
            )

    processed_in_ms = int((time.time() - start_time) * 1000)
    
    return BatchResults(
        session_id=session_id,
        overall_passed=overall_passed,
        identity_consistency=identity_consistency,
        results=results,
        processed_in_ms=processed_in_ms
    )
