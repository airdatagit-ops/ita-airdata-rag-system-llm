"""Configuration for Web Application."""

from pathlib import Path

from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from os import getenv

load_dotenv()

# Project root is two parents up from this file: web/app/config.py -> <root>.
# Used to resolve default paths for operational data (e.g. app.db) regardless
# of the cwd the web service is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_APP_DB = _PROJECT_ROOT / "data" / "app.db"

_ENV = getenv('ENVIRONMENT', 'development').lower()
_IS_PROD = _ENV == 'production'

_DEFAULTS = {
    'development': {
        'HOST': '127.0.0.1',
        'PORT': '8082',
        'RELOAD': 'True',
        'ROOT_PATH': '',
        'API_BASE_URL': 'http://127.0.0.1:8083',
    },
    'production': {
        'HOST': '127.0.0.1',
        'PORT': '8082',
        'RELOAD': 'False',
        'ROOT_PATH': '',
        'API_BASE_URL': 'http://127.0.0.1:8083',
    },
}

def _get(key: str, fallback: str = '') -> str:
    """Read from env, falling back to environment-aware defaults.

    Uses ``os.getenv(key)`` first (None means unset).  An explicit empty
    value in .env (e.g. ``ROOT_PATH=``) is honoured as-is so that
    development can deliberately clear ROOT_PATH.
    """
    val = getenv(key)
    if val is not None:
        return val
    return _DEFAULTS.get(_ENV, _DEFAULTS['development']).get(key, fallback)


class Settings(BaseSettings):
    """Application settings."""
    ENVIRONMENT: str = _ENV

    # Web Server
    HOST: str = _get('HOST')
    PORT: int = int(_get('PORT', '8082'))
    RELOAD: bool = _get('RELOAD', 'False').lower() in ('true', '1', 'yes')
    ROOT_PATH: str = _get('ROOT_PATH')

    # API Configuration
    API_BASE_URL: str = _get('API_BASE_URL')
    API_KEY: str = _get('API_KEY')

    # Application
    APP_NAME: str = _get('APP_NAME', 'Aviation RAG Web Interface')
    APP_VERSION: str = _get('APP_VERSION', '1.0.0')

    # Operational SQLite database (feedback, and future users/flags/config).
    # Served by the Datasette explorer alongside data/store.db.
    APP_DB_PATH: str = _get('APP_DB_PATH', str(_DEFAULT_APP_DB))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Tolerate unknown keys in .env so a rollback to an older revision (or a
        # forward deploy that introduces new keys before the code lands) does
        # not crash Settings() at import time. Pydantic's default is "forbid",
        # which previously turned a stale web/.env into a 502 on the web tier.
        extra = "ignore"


settings = Settings()
