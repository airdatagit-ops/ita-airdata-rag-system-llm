"""Configuration for Web Application."""

from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from os import getenv

load_dotenv()

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
    AUTH_MODE: str = _get('AUTH_MODE', 'api_key')

    # Drupal OAuth2
    SESSION_SECRET_KEY: str = _get('SESSION_SECRET_KEY', _get('API_KEY', 'change-me'))
    DRUPAL_OAUTH_BASE_URL: str = _get('DRUPAL_OAUTH_BASE_URL').rstrip('/')
    DRUPAL_OAUTH_CLIENT_ID: str = _get('DRUPAL_OAUTH_CLIENT_ID')
    DRUPAL_OAUTH_CLIENT_SECRET: str = _get('DRUPAL_OAUTH_CLIENT_SECRET')
    DRUPAL_OAUTH_AUTHORIZE_URL: str = _get('DRUPAL_OAUTH_AUTHORIZE_URL')
    DRUPAL_OAUTH_TOKEN_URL: str = _get('DRUPAL_OAUTH_TOKEN_URL')
    DRUPAL_OAUTH_USERINFO_URL: str = _get('DRUPAL_OAUTH_USERINFO_URL')
    DRUPAL_OAUTH_SCOPES: str = _get('DRUPAL_OAUTH_SCOPES', 'openid profile email')
    DRUPAL_OAUTH_CALLBACK_PATH: str = _get('DRUPAL_OAUTH_CALLBACK_PATH', '/auth/callback')
    SESSION_COOKIE_SECURE: bool = _get('SESSION_COOKIE_SECURE', 'False').lower() in ('true', '1', 'yes')

    # Temporary local web login for demos/presentations
    WEB_LOGIN_ENABLED: bool = _get('WEB_LOGIN_ENABLED', 'False').lower() in ('true', '1', 'yes')
    WEB_LOGIN_USERNAME: str = _get('WEB_LOGIN_USERNAME', 'airdata')
    WEB_LOGIN_PASSWORD: str = _get('WEB_LOGIN_PASSWORD')

    # Application
    APP_NAME: str = _get('APP_NAME', 'Aviation RAG Web Interface')
    APP_VERSION: str = _get('APP_VERSION', '1.0.0')

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
