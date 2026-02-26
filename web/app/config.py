"""Configuration for Web Application."""

from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from os import getenv


class Settings(BaseSettings):
    """Application settings."""
    load_dotenv()
    # Web Server
    HOST: str = getenv('HOST')
    PORT: int = int(getenv('PORT'))
    RELOAD: bool = bool(getenv('RELOAD'))
    ROOT_PATH: str = getenv('ROOT_PATH', '')
    
    # API Configuration
    API_BASE_URL: str = getenv('API_BASE_URL')
    API_KEY: str = getenv('API_KEY')
    
    # Application
    APP_NAME: str = getenv('APP_NAME')
    APP_VERSION: str = getenv('APP_VERSION')
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
