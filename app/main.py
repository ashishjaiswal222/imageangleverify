import os

# Suppress MediaPipe's C++ clearcut telemetry and general TF logging spam
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from fastapi import FastAPI, APIRouter
from app.api.routes_photo_verify import router as photo_verify_router
from app.workers.pool import get_pool, shutdown_pool
import structlog
import asyncio

import numpy as np

logger = structlog.get_logger()

# Setup router
api_router = APIRouter()
api_router.include_router(photo_verify_router, prefix="/api/v1/photo-verify", tags=["verification"])

IS_READY = False

async def warmup_pool():
    global IS_READY
    logger.info("Warming up worker pool with dummy inference...")
    pool = get_pool()
    loop = asyncio.get_running_loop()
    
    # Create a dummy 480x480 blue image
    dummy_img = np.zeros((480, 480, 3), dtype=np.uint8)
    dummy_img[:, :] = (255, 0, 0)
    
    import cv2
    _, encoded = cv2.imencode('.jpg', dummy_img)
    dummy_bytes = encoded.tobytes()
    
    from app.verification.pipeline import verify_image
    
    try:
        # Run dummy inference to force model loading
        await loop.run_in_executor(pool, verify_image, dummy_bytes, "image/jpeg", "front")
    except Exception as e:
        # A VerificationError for NO_FACE_DETECTED is expected here, so we just catch and log
        logger.debug(f"Warmup inference completed with expected error: {e}")
        
    IS_READY = True
    logger.info("Models are loaded. API is ready.")

app = FastAPI(
    title="Photo Verification API",
    description="Microservice for checking photo quality, angle, and occlusions.",
    version="1.0.0"
)

@app.on_event("startup")
async def on_startup():
    logger.info("Starting up API...")
    get_pool() # Initialize pool
    # Start warmup in background to not block uvicorn bind
    asyncio.create_task(warmup_pool())

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down API...")
    shutdown_pool()

app.include_router(api_router)

@app.get("/api/v1/photo-verify/health")
def health_check():
    from fastapi import HTTPException
    if not IS_READY:
        raise HTTPException(status_code=503, detail="Service is starting up and warming models.")
    return {"status": "ready"}
