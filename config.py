"""
Central configuration module for Aviation RAG System.

This module loads all configuration from environment variables and provides
a centralized configuration object for the entire application.

Usage:
    from config import config

    # Access configuration
    qdrant_host = config.QDRANT_HOST
    model_name = config.EMBEDDING_MODEL
"""

from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings

from dotenv import load_dotenv
from os import getenv

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    load_dotenv()
    # ========================================
    # API Security
    # ========================================
    API_KEY: str = getenv('API_KEY')
    CORS_ORIGINS: str = getenv('CORS_ORIGINS')
    RATE_LIMIT: int = getenv('RATE_LIMIT')



    # ========================================
    # Qdrant Configuration
    # ========================================
    QDRANT_HOST: str = getenv('QDRANT_HOST')
    QDRANT_PORT: int = getenv('QDRANT_PORT')
    QDRANT_COLLECTION_NAME: str = getenv('QDRANT_COLLECTION_NAME')
    QDRANT_API_KEY: Optional[str] = getenv('QDRANT_API_KEY')

    # ========================================
    # Ollama Configuration
    # ========================================
    OLLAMA_HOST: str = getenv('OLLAMA_HOST')
    OLLAMA_MODEL: str = getenv('OLLAMA_MODEL')

    # LLM parameters
    LLM_TEMPERATURE: float = getenv('LLM_TEMPERATURE')
    LLM_TOP_P: float = getenv('LLM_TOP_P')
    LLM_MAX_TOKENS: int = getenv('LLM_MAX_TOKENS')

    # ========================================
    # Embedding Model
    # ========================================
    EMBEDDING_MODEL: str = getenv('EMBEDDING_MODEL')
    EMBEDDING_BATCH_SIZE: int = getenv('EMBEDDING_BATCH_SIZE')
    EMBEDDING_MAX_LENGTH: int = getenv('EMBEDDING_MAX_LENGTH')
    EMBEDDING_DIMENSION: int = getenv('EMBEDDING_DIMENSION')

    # ========================================
    # Search Configuration
    # ========================================
    SEARCH_TOP_K: int = getenv('SEARCH_TOP_K')
    SEARCH_SCORE_THRESHOLD: float = getenv('SEARCH_SCORE_THRESHOLD')
    SEARCH_DENSE_ENABLED: bool = getenv('SEARCH_DENSE_ENABLED', 'true')
    SEARCH_SPARSE_ENABLED: bool = getenv('SEARCH_SPARSE_ENABLED', 'false')
    # RRF prefetch pool size (hybrid only) = limit * SEARCH_PREFETCH_MULTIPLIER.
    SEARCH_PREFETCH_MULTIPLIER: int = getenv('SEARCH_PREFETCH_MULTIPLIER', '3')
    # When a sub-query carries a sort, fetch limit * this many candidates
    # before in-memory sorting; the deduped pool is then capped back to limit.
    SEARCH_SORT_FETCH_MULTIPLIER: int = getenv('SEARCH_SORT_FETCH_MULTIPLIER', '3')
    SPARSE_EMBEDDING_MODEL: str = getenv('SPARSE_EMBEDDING_MODEL', 'Qdrant/bm25')
    HNSW_EF_SEARCH: int = getenv('HNSW_EF_SEARCH')
    HNSW_M: int = getenv('HNSW_M')
    HNSW_EF_CONSTRUCT: int = getenv('HNSW_EF_CONSTRUCT')

    # ========================================
    # Chunking Configuration
    # ========================================
    CHUNK_MAX_TOKENS: int = getenv('CHUNK_MAX_TOKENS')
    CHUNK_OVERLAP: int = getenv('CHUNK_OVERLAP')

    # ========================================
    # Temporal Extraction
    # ========================================
    DEFAULT_EFFECTIVE_DAYS: int = getenv('DEFAULT_EFFECTIVE_DAYS')

    # ========================================
    # LexML Scraper Configuration
    # ========================================
    LEXML_API_URL: str = getenv('LEXML_API_URL')
    LEXML_MAX_RECORDS_PER_PAGE: int = getenv('LEXML_MAX_RECORDS_PER_PAGE')
    LEXML_KEYWORDS: str = getenv('LEXML_KEYWORDS')
    LEXML_MAX_RATE: Optional[int] = getenv('LEXML_MAX_RATE')

    # ========================================
    # SISLAER Scraper Configuration
    # ========================================
    SISLAER_BASE_URL: str = getenv(
        'SISLAER_BASE_URL',
        'https://www.sislaer.fab.mil.br/TerminalWebCENDOC',
    )
    SISLAER_MAX_RATE: float = float(getenv('SISLAER_MAX_RATE', '10'))
    SISLAER_CONCURRENCY: int = int(getenv('SISLAER_CONCURRENCY', '10'))
    SISLAER_TIMEOUT: int = int(getenv('SISLAER_TIMEOUT', '30'))
    SISLAER_START_ID: int = int(getenv('SISLAER_START_ID', '1'))
    SISLAER_END_ID: int = int(getenv('SISLAER_END_ID', '0'))
    SISLAER_DOC_TYPES: str = getenv(
        'SISLAER_DOC_TYPES',
        'ICA,DCA,FCA,MCA,NSCA,PCA,RCA,TCA,OCA,ROCA,RICA,RIMA,RMA,NPA,PTA,'
        'BCA,BMA,IMA,NOPREP,AVISO,COMUNICADO,NOTA,ORDEM TÉCNICA,'
        'ORIENTAÇÃO NORMATIVA,NORMAS DO COMPREP,MANUAL ELETRÔNICO,MANUAL - OUTROS,'
        'LEI,LEI COMPLEMENTAR,DECRETO,DECRETO - LEI,'
        'RESOLUÇÃO,INSTRUÇÃO NORMATIVA,MEDIDA PROVISÓRIA,CONSTITUIÇÃO FEDERAL,'
        'BOLETIM EXTERNO',
    )

    # ========================================
    # PDF Parser Configuration
    # ========================================
    ENABLE_OCR: bool = getenv('ENABLE_OCR')
    OCR_LANGUAGE: str = getenv('OCR_LANGUAGE')

    # ========================================
    # Logging
    # ========================================
    LOG_LEVEL: str = getenv('LOG_LEVEL')
    LOG_FILE: str = getenv('LOG_FILE')
    LOG_ROTATION: str = getenv('LOG_ROTATION')
    LOG_RETENTION: str = getenv('LOG_RETENTION')

    # ========================================
    # Cache Configuration (Optional)
    # ========================================
    REDIS_HOST: Optional[str] = getenv('REDIS_HOST')
    REDIS_PORT: int = getenv('REDIS_PORT')
    REDIS_DB: int = getenv('REDIS_DB')
    CACHE_TTL: int = getenv('CACHE_TTL')

    # ========================================
    # Database Configuration (Optional)
    # ========================================
    DATABASE_URL: Optional[str] = getenv('DATABASE_URL')

    # ========================================
    # Monitoring (Optional)
    # ========================================
    ENABLE_METRICS: bool = getenv('ENABLE_METRICS')
    PROMETHEUS_PORT: int = getenv('PROMETHEUS_PORT')

    # ========================================
    # Development Settings
    # ========================================
    ENVIRONMENT: str = getenv('ENVIRONMENT')
    DEBUG: bool = getenv('DEBUG')
    RELOAD: bool = getenv('RELOAD')

    # ========================================
    # Performance Tuning
    # ========================================
    API_WORKERS: int = getenv('API_WORKERS')
    CUDA_VISIBLE_DEVICES: str = getenv('CUDA_VISIBLE_DEVICES')

    # ========================================
    # Data Paths
    # ========================================
    DATA_DIR: str = getenv('DATA_DIR')
    PROCESSED_DIR: str = getenv('PROCESSED_DIR')
    MODEL_CACHE_DIR: str = getenv('MODEL_CACHE_DIR')

    # ========================================
    # Pipeline Storage (3-phase architecture)
    # ========================================
    STORE_DB_PATH: str = getenv('STORE_DB_PATH', './data/store.db')
    EMBEDDINGS_DIR: str = getenv('EMBEDDINGS_DIR', './data/embeddings')
    DEFAULT_EMBEDDING_MODE: str = getenv('DEFAULT_EMBEDDING_MODE', 'dense')

    # ========================================
    # API Configuration
    # ========================================
    API_HOST: str = getenv('API_HOST')
    API_PORT: int = getenv('API_PORT')
    API_TITLE: str = getenv('API_TITLE')
    API_VERSION: str = getenv('API_VERSION')
    API_DESCRIPTION: str = getenv('API_DESCRIPTION')

    # ========================================
    # Ingestion Configuration
    # ========================================
    INGESTION_BATCH_SIZE: int = getenv('INGESTION_BATCH_SIZE')
    ENABLE_PARALLEL_PROCESSING: bool = getenv('ENABLE_PARALLEL_PROCESSING')
    NUM_WORKERS: int = getenv('NUM_WORKERS')

    # ========================================
    # RAG Pipeline — Rewriter
    # ========================================
    REWRITER_ENABLED: bool = getenv('REWRITER_ENABLED', 'false').lower() in ('true', '1', 'yes')
    REWRITER_MODEL: str = getenv('REWRITER_MODEL', 'llama3.2:3b')
    REWRITER_MAX_QUERIES: int = int(getenv('REWRITER_MAX_QUERIES', '3'))
    REWRITER_MAX_QUERY_LENGTH: int = int(getenv('REWRITER_MAX_QUERY_LENGTH', '500'))
    REWRITER_TEMPERATURE: float = float(getenv('REWRITER_TEMPERATURE', '0.3'))
    REWRITER_TIMEOUT: int = int(getenv('REWRITER_TIMEOUT', '60'))
    REWRITER_PROMPT_VERSION: str = getenv('REWRITER_PROMPT_VERSION', 'v1')

    # ========================================
    # RAG Pipeline — Evaluator (Cross-Encoder)
    # ========================================
    EVALUATOR_ENABLED: bool = getenv('EVALUATOR_ENABLED', 'true').lower() in ('true', '1', 'yes')
    CROSS_ENCODER_MODEL: str = getenv(
        'CROSS_ENCODER_MODEL', 'BAAI/bge-reranker-base',
    )
    EVALUATOR_THRESHOLD: int = int(getenv('EVALUATOR_THRESHOLD', '55'))
    EVALUATOR_BATCH_SIZE: int = int(getenv('EVALUATOR_BATCH_SIZE', '32'))
    EVALUATOR_MAX_TOKENS: int = int(getenv('EVALUATOR_MAX_TOKENS', '480'))

    # ========================================
    # RAG Pipeline — Generator
    # ========================================
    GENERATOR_MODEL: str = getenv('GENERATOR_MODEL', '')
    GENERATOR_MAX_RESPONSE_TOKENS: int = int(getenv('GENERATOR_MAX_RESPONSE_TOKENS', '1024'))
    GENERATOR_MAX_DOC_CHARS: int = int(getenv('GENERATOR_MAX_DOC_CHARS', '2000'))
    GENERATOR_MAX_DOCS: int = int(getenv('GENERATOR_MAX_DOCS', '7'))
    GENERATOR_GROUNDED_ONLY: bool = getenv('GENERATOR_GROUNDED_ONLY', 'true').lower() in ('true', '1', 'yes')
    GENERATOR_TIMEOUT: int = int(getenv('GENERATOR_TIMEOUT', '120'))

    # ========================================
    # RAG Pipeline — General
    # ========================================
    PIPELINE_DEBUG: bool = getenv('PIPELINE_DEBUG', 'false').lower() in ('true', '1', 'yes')

    # ========================================
    # GPU Inference Mode
    # ========================================
    # "local"  — load models in-process (uses GPU if available, falls back to CPU)
    # "remote" — call the GPU inference server via HTTP
    # "cpu"    — force CPU-only execution (no CUDA, no remote)
    INFERENCE_MODE: str = getenv('INFERENCE_MODE', 'local')
    GPU_SERVER_URL: str = getenv('GPU_SERVER_URL', 'http://localhost:8090')
    GPU_SERVER_API_KEY: str = getenv('GPU_SERVER_API_KEY', '')
    GPU_SERVER_TIMEOUT: int = int(getenv('GPU_SERVER_TIMEOUT', '120'))

    # ========================================
    # Advanced Settings
    # ========================================
    LOG_QUERIES: bool = getenv('LOG_QUERIES')
    ENABLE_PROFILING: bool = getenv('ENABLE_PROFILING')
    LLM_TIMEOUT: int = getenv('LLM_TIMEOUT')
    SEARCH_TIMEOUT: int = getenv('SEARCH_TIMEOUT')

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    # ========================================
    # Derived Properties
    # ========================================
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    @property
    def lexml_keywords_list(self) -> List[str]:
        """Parse LexML keywords from comma-separated string."""
        return [kw.strip() for kw in self.LEXML_KEYWORDS.split(",")]

    @property
    def cuda_devices_list(self) -> List[int]:
        """Parse CUDA device IDs from comma-separated string."""
        if not self.CUDA_VISIBLE_DEVICES:
            return []
        try:
            return [int(d.strip()) for d in self.CUDA_VISIBLE_DEVICES.split(",")]
        except ValueError:
            return []

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT.lower() == "development"

    # ========================================
    # Path Helpers
    # ========================================
    def get_data_path(self, filename: str = "") -> Path:
        """Get path in data directory."""
        path = Path(self.DATA_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path / filename if filename else path

    def get_processed_path(self, filename: str = "") -> Path:
        """Get path in processed directory."""
        path = Path(self.PROCESSED_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path / filename if filename else path

    def get_model_cache_path(self, filename: str = "") -> Path:
        """Get path in model cache directory."""
        path = Path(self.MODEL_CACHE_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path / filename if filename else path

    def get_log_path(self) -> Path:
        """Get log file path."""
        log_path = Path(self.LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return log_path


# ========================================
# Global Configuration Instance
# ========================================
try:
    config = Settings()
except Exception as e:
    # If .env file doesn't exist or has errors, provide helpful message
    print(f"Error loading configuration: {e}")
    print("\nPlease ensure:")
    print("1. You have copied .env.example to .env")
    print("2. All required environment variables are set in .env")
    print("3. The .env file is in the project root directory")
    raise


# ========================================
# Configuration Validation
# ========================================
def validate_config() -> bool:
    """
    Validate configuration settings.

    Returns:
        bool: True if configuration is valid, False otherwise
    """
    issues = []

    # Check API key
    if config.API_KEY == "your-secret-api-key-here-generate-random-string":
        issues.append("API_KEY is still using default value. Please set a secure random key.")

    # Check Qdrant connection
    if config.QDRANT_HOST == "localhost" and config.is_production:
        issues.append("Warning: Using localhost for Qdrant in production environment.")

    # Check embedding model
    if not config.EMBEDDING_MODEL:
        issues.append("EMBEDDING_MODEL is not set.")

    # Check LLM configuration
    if config.INFERENCE_MODE != "remote" and (not config.OLLAMA_HOST or not config.OLLAMA_MODEL):
        issues.append("Ollama configuration (OLLAMA_HOST, OLLAMA_MODEL) is incomplete.")

    # Check GPU server when remote mode
    if config.INFERENCE_MODE == "remote" and not config.GPU_SERVER_URL:
        issues.append("GPU_SERVER_URL is required when INFERENCE_MODE=remote.")

    # Check data directories
    for dir_name, dir_path in [
        ("DATA_DIR", config.DATA_DIR),
        ("PROCESSED_DIR", config.PROCESSED_DIR),
        ("MODEL_CACHE_DIR", config.MODEL_CACHE_DIR),
    ]:
        path = Path(dir_path)
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create {dir_name} at {dir_path}: {e}")

    # Print issues
    if issues:
        print("\n⚠️  Configuration Issues Found:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        return False

    print("✓ Configuration validated successfully")
    return True


# ========================================
# Helper Functions
# ========================================
def print_config():
    """Print current configuration (excluding sensitive values)."""
    print("\n" + "=" * 60)
    print("Aviation RAG System Configuration")
    print("=" * 60)

    sections = {
        "Environment": [
            ("Environment", config.ENVIRONMENT),
            ("Debug", config.DEBUG),
            ("Inference Mode", config.INFERENCE_MODE),
            ("GPU Server URL", config.GPU_SERVER_URL if config.INFERENCE_MODE == "remote" else "N/A"),
        ],
        "Qdrant": [
            ("Host", config.QDRANT_HOST),
            ("Port", config.QDRANT_PORT),
            ("Collection", config.QDRANT_COLLECTION_NAME),
        ],
        "Ollama/LLM": [
            ("Host", config.OLLAMA_HOST),
            ("Model", config.OLLAMA_MODEL),
            ("Temperature", config.LLM_TEMPERATURE),
        ],
        "Embeddings": [
            ("Model", config.EMBEDDING_MODEL),
            ("Batch Size", config.EMBEDDING_BATCH_SIZE),
            ("Dimension", config.EMBEDDING_DIMENSION),
        ],
        "Search": [
            ("Top-K", config.SEARCH_TOP_K),
            ("Score Threshold", config.SEARCH_SCORE_THRESHOLD),
            ("Dense (semantic)", config.SEARCH_DENSE_ENABLED),
            ("Sparse (BM25)", config.SEARCH_SPARSE_ENABLED),
        ],
        "API": [
            ("Host", config.API_HOST),
            ("Port", config.API_PORT),
            ("Workers", config.API_WORKERS),
            ("Rate Limit", f"{config.RATE_LIMIT}/min"),
        ],
    }

    for section_name, items in sections.items():
        print(f"\n{section_name}:")
        for key, value in items:
            print(f"  {key:20s}: {value}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    """Run configuration validation and print config when executed directly."""
    print_config()
    validate_config()
