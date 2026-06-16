from functools import lru_cache

from langchain_openai import AzureChatOpenAI

from core.config import get_settings


@lru_cache(maxsize=1)
def get_chat_model() -> AzureChatOpenAI:
    """Return the shared Azure OpenAI GPT-5 chat model."""
    settings = get_settings()
    return AzureChatOpenAI(
        azure_endpoint=str(settings.azure_openai_endpoint),
        api_key=settings.azure_openai_api_key.get_secret_value(),
        azure_deployment=settings.azure_openai_deployment,
        api_version=settings.azure_openai_api_version,
        model=settings.azure_openai_model,
        max_completion_tokens=settings.azure_openai_max_completion_tokens,
        reasoning_effort=settings.azure_openai_reasoning_effort,
        max_retries=settings.azure_openai_max_retries,
        timeout=settings.azure_openai_timeout_seconds,
    )
