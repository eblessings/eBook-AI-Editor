"""
Configuration management for the eBook Editor application.
Handles environment variables, AI model settings, and application configuration.
"""

import os
from typing import Optional, List, Literal
from pydantic import BaseSettings, Field, validator
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Server Configuration
    APP_NAME: str = "eBook Editor Pro"
    APP_VERSION: str = "1.0.0"
    HOST: str = Field(default="localhost", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENVIRONMENT: Literal["development", "staging", "production"] = Field(
        default="development", env="ENVIRONMENT"
    )
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"],
        env="CORS_ORIGINS"
    )
    
    # Security
    SECRET_KEY: str = Field(
        default="your-secret-key-change-this-in-production", 
        env="SECRET_KEY"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # AI Model Configuration
    USE_LOCAL_MODEL: bool = Field(default=True, env="USE_LOCAL_MODEL")
    LOCAL_MODEL_NAME: str = Field(
        default="microsoft/DialoGPT-medium", 
        env="LOCAL_MODEL_NAME"
    )
    LOCAL_MODEL_PATH: str = Field(default="./models", env="LOCAL_MODEL_PATH")
    MODEL_CACHE_DIR: str = Field(default="./model_cache", env="MODEL_CACHE_DIR")
    
    # Advanced AI Models
    GRAMMAR_MODEL: str = Field(
        default="textattack/roberta-base-CoLA", 
        env="GRAMMAR_MODEL"
    )
    STYLE_MODEL: str = Field(
        default="microsoft/DialoGPT-large", 
        env="STYLE_MODEL"
    )
    SUMMARIZATION_MODEL: str = Field(
        default="facebook/bart-large-cnn", 
        env="SUMMARIZATION_MODEL"
    )
    
    # External AI API Configuration
    EXTERNAL_AI_ENABLED: bool = Field(default=False, env="EXTERNAL_AI_ENABLED")
    EXTERNAL_AI_BASE_URL: str = Field(
        default="http://localhost:1234/v1", 
        env="EXTERNAL_AI_BASE_URL"
    )
    EXTERNAL_AI_MODEL: str = Field(default="local-model", env="EXTERNAL_AI_MODEL")
    EXTERNAL_AI_API_KEY: str = Field(default="", env="EXTERNAL_AI_API_KEY")
    EXTERNAL_AI_TIMEOUT: int = Field(default=30, env="EXTERNAL_AI_TIMEOUT")
    
    # File Processing Configuration
    MAX_FILE_SIZE: int = Field(default=50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB
    MAX_CONTENT_LENGTH: int = Field(default=1000000, env="MAX_CONTENT_LENGTH")  # 1M chars
    TEMP_DIR: str = Field(default="./temp", env="TEMP_DIR")
    UPLOAD_DIR: str = Field(default="./uploads", env="UPLOAD_DIR")
    EXPORT_DIR: str = Field(default="./exports", env="EXPORT_DIR")
    
    # Supported file types
    SUPPORTED_UPLOAD_TYPES: List[str] = Field(
        default=[
            "text/plain",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/pdf",
            "application/epub+zip",
            "text/markdown",
            "text/html"
        ],
        env="SUPPORTED_UPLOAD_TYPES"
    )
    
    # NLP Configuration
    SPACY_MODEL: str = Field(default="en_core_web_sm", env="SPACY_MODEL")
    LANGUAGE_TOOL_LANGUAGE: str = Field(default="en-US", env="LANGUAGE_TOOL_LANGUAGE")
    ENABLE_GRAMMAR_CHECK: bool = Field(default=True, env="ENABLE_GRAMMAR_CHECK")
    ENABLE_STYLE_CHECK: bool = Field(default=True, env="ENABLE_STYLE_CHECK")
    ENABLE_READABILITY_CHECK: bool = Field(default=True, env="ENABLE_READABILITY_CHECK")
    
    # Database Configuration (optional)
    DATABASE_URL: str = Field(
        default="sqlite:///./ebook_editor.db", 
        env="DATABASE_URL"
    )
    DB_ECHO: bool = Field(default=False, env="DB_ECHO")
    
    # Redis Configuration (for caching and background tasks)
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    CACHE_ENABLED: bool = Field(default=True, env="CACHE_ENABLED")
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    
    # Background Task Configuration
    ENABLE_BACKGROUND_TASKS: bool = Field(default=True, env="ENABLE_BACKGROUND_TASKS")
    TASK_QUEUE_NAME: str = Field(default="ebook_editor_tasks", env="TASK_QUEUE_NAME")
    
    # Monitoring and Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    
    # eBook Generation Settings
    DEFAULT_EPUB_LANGUAGE: str = Field(default="en", env="DEFAULT_EPUB_LANGUAGE")
    DEFAULT_EPUB_PUBLISHER: str = Field(
        default="eBook Editor Pro", 
        env="DEFAULT_EPUB_PUBLISHER"
    )
    CHAPTER_DETECTION_METHOD: Literal["ai", "heading", "pagebreak"] = Field(
        default="ai", 
        env="CHAPTER_DETECTION_METHOD"
    )
    
    # Performance Settings
    MAX_WORKERS: int = Field(default=4, env="MAX_WORKERS")
    ENABLE_GPU: bool = Field(default=False, env="ENABLE_GPU")
    MODEL_DEVICE: str = Field(default="cpu", env="MODEL_DEVICE")
    
    @validator("TEMP_DIR", "UPLOAD_DIR", "EXPORT_DIR", "LOCAL_MODEL_PATH", "MODEL_CACHE_DIR")
    def create_directories(cls, v):
        """Ensure directories exist."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string if needed."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("SUPPORTED_UPLOAD_TYPES", pre=True)
    def parse_upload_types(cls, v):
        """Parse supported upload types from string if needed."""
        if isinstance(v, str):
            return [file_type.strip() for file_type in v.split(",")]
        return v
    
    @validator("MODEL_DEVICE")
    def validate_model_device(cls, v):
        """Validate and auto-detect model device."""
        import torch
        if v == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


class DevelopmentSettings(Settings):
    """Development-specific settings."""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    DB_ECHO: bool = True
    ENABLE_METRICS: bool = False


class ProductionSettings(Settings):
    """Production-specific settings."""
    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"
    DB_ECHO: bool = False
    ENABLE_METRICS: bool = True


class TestingSettings(Settings):
    """Testing-specific settings."""
    DEBUG: bool = True
    DATABASE_URL: str = "sqlite:///./test.db"
    TEMP_DIR: str = "./test_temp"
    ENABLE_BACKGROUND_TASKS: bool = False


def get_settings() -> Settings:
    """Get settings based on environment."""
    environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "development":
        return DevelopmentSettings()
    elif environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return Settings()


# Global settings instance
settings = get_settings()


# AI Model Configurations
AI_MODEL_CONFIGS = {
    "text-generation": {
        "small": {
            "model": "microsoft/DialoGPT-small",
            "max_length": 512,
            "temperature": 0.7
        },
        "medium": {
            "model": "microsoft/DialoGPT-medium", 
            "max_length": 1024,
            "temperature": 0.7
        },
        "large": {
            "model": "microsoft/DialoGPT-large",
            "max_length": 2048,
            "temperature": 0.7
        }
    },
    "grammar-checking": {
        "default": {
            "model": "textattack/roberta-base-CoLA",
            "threshold": 0.5
        }
    },
    "style-analysis": {
        "default": {
            "model": "facebook/bart-large-cnn",
            "max_length": 1024
        }
    },
    "summarization": {
        "short": {
            "model": "facebook/bart-base",
            "max_length": 142,
            "min_length": 30
        },
        "long": {
            "model": "facebook/bart-large-cnn",
            "max_length": 512,
            "min_length": 100
        }
    }
}


# eBook Format Configurations
EBOOK_FORMATS = {
    "epub": {
        "extension": ".epub",
        "mime_type": "application/epub+zip",
        "supports_images": True,
        "supports_toc": True,
        "supports_metadata": True
    },
    "pdf": {
        "extension": ".pdf",
        "mime_type": "application/pdf",
        "supports_images": True,
        "supports_toc": True,
        "supports_metadata": True
    },
    "docx": {
        "extension": ".docx",
        "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "supports_images": True,
        "supports_toc": True,
        "supports_metadata": True
    },
    "html": {
        "extension": ".html",
        "mime_type": "text/html",
        "supports_images": True,
        "supports_toc": True,
        "supports_metadata": False
    },
    "txt": {
        "extension": ".txt",
        "mime_type": "text/plain",
        "supports_images": False,
        "supports_toc": False,
        "supports_metadata": False
    }
}


# Readability Grade Levels
READABILITY_LEVELS = {
    "very_easy": {"min_score": 90, "grade": "5th grade", "color": "#10B981"},
    "easy": {"min_score": 80, "grade": "6th grade", "color": "#34D399"},
    "fairly_easy": {"min_score": 70, "grade": "7th grade", "color": "#6EE7B7"},
    "standard": {"min_score": 60, "grade": "8th-9th grade", "color": "#FDE047"},
    "fairly_difficult": {"min_score": 50, "grade": "10th-12th grade", "color": "#FBBF24"},
    "difficult": {"min_score": 30, "grade": "College level", "color": "#F59E0B"},
    "very_difficult": {"min_score": 0, "grade": "Graduate level", "color": "#EF4444"}
}
