from core.config import AzureOpenAISettings


def test_openai_timeout_defaults_to_long_running_value() -> None:
    settings = AzureOpenAISettings(
        azure_openai_endpoint="https://example.openai.azure.com/",
        azure_openai_api_key="test-key",
        azure_openai_deployment="gpt-5",
    )

    assert settings.azure_openai_timeout_seconds == 240
    assert settings.azure_openai_max_retries == 2
