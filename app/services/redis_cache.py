import redis.asyncio as redis
import structlog
import numpy as np
from app.config import settings

logger = structlog.get_logger()

class RedisCacheManager:
    _instance = None
    
    def __init__(self):
        self.client = None
        self.connected = False
        
    async def connect(self):
        try:
            self.client = redis.from_url(settings.redis_url)
            await self.client.ping()
            self.connected = True
            logger.info("Connected to Redis for identity caching.")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis. Real-time caching will be disabled. Error: {e}")
            self.connected = False
            self.client = None
            
    async def disconnect(self):
        if self.client:
            await self.client.aclose()
            
    async def save_embedding(self, session_id: str, position: str, embedding: np.ndarray):
        if not self.connected or not self.client:
            return
            
        try:
            # Serialize numpy array to bytes
            emb_bytes = embedding.tobytes()
            key = f"session:{session_id}:embeddings"
            
            # Store in a Redis Hash map where key is the position
            await self.client.hset(key, position, emb_bytes)
            # Ensure TTL is set
            await self.client.expire(key, settings.identity_session_ttl_seconds)
        except Exception as e:
            logger.error(f"Failed to save embedding to Redis: {e}")

    async def get_embeddings(self, session_id: str) -> dict[str, np.ndarray]:
        if not self.connected or not self.client:
            return {}
            
        try:
            key = f"session:{session_id}:embeddings"
            cached_data = await self.client.hgetall(key)
            
            embeddings = {}
            for pos_bytes, emb_bytes in cached_data.items():
                pos_str = pos_bytes.decode('utf-8')
                # 128-d float32 array
                emb_array = np.frombuffer(emb_bytes, dtype=np.float32)
                embeddings[pos_str] = emb_array
            return embeddings
        except Exception as e:
            logger.error(f"Failed to fetch embeddings from Redis: {e}")
            return {}

_redis_cache_instance = RedisCacheManager()

def get_redis_cache() -> RedisCacheManager:
    return _redis_cache_instance
