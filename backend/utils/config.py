"""
Configuration management using environment variables and defaults
"""

import os
from typing import Optional

class Settings:
    """Application settings"""
    
    # Endee configuration
    endee_host: str = os.getenv("ENDEE_HOST", "localhost")
    endee_port: int = int(os.getenv("ENDEE_PORT", "8080"))
    
    # Embedding model configuration
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # LLM configuration
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # Document processing
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    
    # API configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Storage
    upload_dir: str = os.getenv("UPLOAD_DIR", "/tmp/uploads")

settings = Settings()