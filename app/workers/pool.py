import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from app.config import settings

# Global reference to the pool
_pool = None

def init_worker():
    """
    Initializer for the process pool. 
    This forces the MediaPipe models to load once per worker process.
    """
    # Import inside the worker so it doesn't load in the main thread prematurely
    from app.verification.models_loader import get_models
    get_models()

def get_pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(
            max_workers=settings.worker_pool_size,
            initializer=init_worker
        )
    return _pool

def shutdown_pool():
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=True)
        _pool = None
