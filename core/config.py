from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl
from typing import List


class Settings(BaseSettings):
    PROJECT_NAME: str = "Sentiment Analysis API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "API for product review sentiment analysis using BiGRU model"

    # CORS
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
    ]

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()