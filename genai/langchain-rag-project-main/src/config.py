"""Configuration management for the RAG system."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables and config files."""
    
    # API Keys - Optional for development/testing
    google_api_key: str = Field("", env="GOOGLE_API_KEY")
    qdrant_url: str = Field("", env="QDRANT_URL")  
    qdrant_api_key: str = Field("", env="QDRANT_API_KEY")
    mem0_api_key: str = Field("", env="MEM0_API_KEY")
    
    # Database
    collection_name: str = Field("langchain_docs", env="COLLECTION_NAME")
    
    # API Configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    debug: bool = Field(True, env="DEBUG")
    
    # RAG Configuration
    chunk_size: int = Field(1200, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    top_k_retrieval: int = Field(5, env="TOP_K_RETRIEVAL")
    embedding_model: str = Field("gemini-embedding-001", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(3072, env="EMBEDDING_DIMENSION")
    llm_model: str = Field("gemini-2.0-flash", env="LLM_MODEL")
    max_tokens: int = Field(4000, env="MAX_TOKENS")
    temperature: float = Field(0.1, env="TEMPERATURE")
    
    # Enhanced Prompting Configuration
    use_adaptive_prompts: bool = Field(True, env="USE_ADAPTIVE_PROMPTS")
    enable_context_analysis: bool = Field(True, env="ENABLE_CONTEXT_ANALYSIS")
    structured_response_format: bool = Field(True, env="STRUCTURED_RESPONSE_FORMAT")
    fallback_similarity_threshold: float = Field(0.7, env="FALLBACK_SIMILARITY_THRESHOLD")
    adaptive_prompt_debug: bool = Field(False, env="ADAPTIVE_PROMPT_DEBUG")
    
    # Memory Management Configuration (Mem0)
    memory_enabled: bool = Field(True, env="MEMORY_ENABLED")
    memory_search_limit: int = Field(5, env="MEMORY_SEARCH_LIMIT")
    memory_search_threshold: float = Field(0.7, env="MEMORY_SEARCH_THRESHOLD")
    memory_auto_save: bool = Field(True, env="MEMORY_AUTO_SAVE")
    memory_window_size: int = Field(10, env="MEMORY_WINDOW_SIZE")  # Last N turns
    memory_context_weight: float = Field(0.3, env="MEMORY_CONTEXT_WEIGHT")  # 0-1, how much to weight memory vs documents
    
    # Data Paths
    data_path: str = Field("./data", env="DATA_PATH")
    raw_data_path: str = Field("./data/raw", env="RAW_DATA_PATH")
    processed_data_path: str = Field("./data/processed", env="PROCESSED_DATA_PATH")
    embeddings_path: str = Field("./data/embeddings", env="EMBEDDINGS_PATH")
    
    # LangChain Repository
    langchain_repo_url: str = Field(
        "https://github.com/langchain-ai/langchain.git",
        env="LANGCHAIN_REPO_URL"
    )
    langchain_local_path: str = Field("./data/raw/langchain", env="LANGCHAIN_LOCAL_PATH")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        return {}
    
    with open(config_file, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


# Global settings instance
settings = Settings()
config = load_config()