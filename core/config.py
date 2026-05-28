from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # External APIs
    openrouter_api_key: str
    groq_api_key: str
    google_api_key: str
    hf_token: str = ""

    # Model strings
    visual_model: str
    synthesis_model: str
    critic_model: str
    orchestrator_model: str
    ollama_base_url: str
    finetuned_model_id: str = ""

    # PostgreSQL
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_url: str

    # Redis
    redis_url: str

    # Chroma
    chroma_host: str
    chroma_port: int

    # JWT
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # App
    app_port: int = 8080
    debug: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
