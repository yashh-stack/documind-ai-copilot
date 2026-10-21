import json
import logging
import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class _JSONFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", ""),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _make_logger(name: str, level: str = "INFO") -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    handler = logging.StreamHandler()
    handler.setFormatter(_JSONFormatter())
    log.addHandler(handler)
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    log.propagate = False
    return log


logger = _make_logger("documind", os.getenv("LOG_LEVEL", "INFO"))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    api_title: str = "DocuMind AI Copilot"
    api_version: str = "4.0.0"
    data_dir: str = "data"
    log_level: str = "INFO"
    port: int = 8000

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "deepseek/deepseek-chat-v3-0324:free"
    llm_temperature: float = 0.15
    llm_max_retries: int = 3

    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 5

    memory_window_size: int = 10
    max_upload_size_mb: int = 50
    max_pages: int = 100
    rate_limit_per_minute: int = 30

    cors_origins: str = "*"
    frontend_api_url: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
