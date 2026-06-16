from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfigurationError(RuntimeError):
    pass


class AzureOpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    azure_openai_endpoint: AnyHttpUrl
    azure_openai_api_key: SecretStr
    azure_openai_deployment: str = Field(min_length=1)
    azure_openai_model: str = "gpt-5"
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_max_completion_tokens: int = Field(default=3000, gt=0)
    azure_openai_reasoning_effort: Literal["minimal", "low", "medium", "high"] = "low"
    azure_openai_timeout_seconds: float = Field(default=240, gt=0)
    azure_openai_max_retries: int = Field(default=2, ge=0)


@lru_cache(maxsize=1)
def get_settings() -> AzureOpenAISettings:
    try:
        return AzureOpenAISettings()  # type: ignore[call-arg]
    except ValidationError as exc:
        required = (
            "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and "
            "AZURE_OPENAI_DEPLOYMENT"
        )
        raise LLMConfigurationError(
            f"Azure OpenAI is not configured. Set {required} in .env."
        ) from exc
