import pytest

from config.settings import MissingAPIKeyError, Settings
from orchestration.pipeline import build_client


def test_build_client_raises_when_api_key_missing() -> None:
    settings = Settings(anthropic_api_key=None)
    with pytest.raises(MissingAPIKeyError):
        build_client(settings)


def test_build_client_raises_when_api_key_empty_string() -> None:
    settings = Settings(anthropic_api_key="")
    with pytest.raises(MissingAPIKeyError):
        build_client(settings)


def test_build_client_succeeds_with_api_key() -> None:
    settings = Settings(anthropic_api_key="sk-ant-test-key")
    client = build_client(settings)
    assert client is not None
