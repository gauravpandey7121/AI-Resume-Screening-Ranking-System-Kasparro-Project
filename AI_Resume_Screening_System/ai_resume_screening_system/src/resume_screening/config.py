from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock").lower()
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    github_token: str | None = os.getenv("GITHUB_TOKEN")
    github_enabled: bool = os.getenv("GITHUB_ENABLED", "true").lower() not in {"0", "false", "no"}
    github_timeout_seconds: float = float(os.getenv("GITHUB_TIMEOUT_SECONDS", "8"))
    github_recent_days: int = int(os.getenv("GITHUB_RECENT_DAYS", "30"))
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))


settings = Settings()

WEIGHTS = {
    "ai_project_depth": 40,
    "python_backend": 30,
    "cloud_fullstack": 15,
    "github": 10,
    "engineering_depth": 5,
}
