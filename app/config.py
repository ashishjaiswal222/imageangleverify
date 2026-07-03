from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Photo Verification API"
    worker_pool_size: int = 4
    batch_timeout_seconds: int = 45
    
    # Redis configuration
    redis_url: str = "redis://localhost:6379/0"
    identity_session_ttl_seconds: int = 3600
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
