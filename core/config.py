import os
from pathlib import Path


class Settings:
    # Base settings
    PROJECT_NAME = "PDF RAG System"
    API_V1_STR = "/api/v1"

    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

    # Database
    DB_PATH = "pdf_database.db"
    VECTOR_STORE_PATH = "chroma_db"

    # LLM settings
    DEFAULT_LLM_MODEL = "llama3.1"
    DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

    # File upload
    UPLOAD_DIR = Path("uploads")


settings = Settings()
