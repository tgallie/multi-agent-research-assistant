"""Environment-backed application configuration."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent.schemas import BudgetLimits


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    tavily_api_key: str | None = None
    serpapi_api_key: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4.1-mini"
    agent_log_path: Path = Path("var/runs.jsonl")
    request_timeout_seconds: float = Field(default=15.0, ge=1.0, le=60.0)
    model_cost_per_million_tokens: float = Field(default=0.0, ge=0.0)

    def budget_limits(self) -> BudgetLimits:
        """Return the default production-safe run limits."""

        return BudgetLimits()
