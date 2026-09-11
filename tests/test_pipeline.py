from datetime import UTC, datetime

import pytest

from config.settings import MissingAPIKeyError, Settings
from domain.intents import Intent
from domain.models import ClassificationResult, Conversation, Message
from orchestration.pipeline import build_client, build_precedent_index


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


class _FakeClassifier:
    def __init__(self, intent_by_text: dict[str, Intent], failing_texts: frozenset[str] = frozenset()) -> None:
        self._intent_by_text = intent_by_text
        self._failing_texts = failing_texts

    def classify(self, message: Message) -> ClassificationResult:
        if message.text in self._failing_texts:
            raise ValueError("simulated malformed LLM response")
        return ClassificationResult(intent=self._intent_by_text[message.text], confidence=0.9, rationale="fake")


def _conversation(conversation_id: str, customer_text: str, brand_text: str) -> Conversation:
    now = datetime.now(UTC)
    return Conversation(
        conversation_id=conversation_id,
        messages=[
            Message(id=f"{conversation_id}-c", author_id="customer", text=customer_text, created_at=now, in_reply_to_id=None),
            Message(id=f"{conversation_id}-b", author_id="SpotifyCares", text=brand_text, created_at=now, in_reply_to_id=f"{conversation_id}-c"),
        ],
    )


def test_build_precedent_index_groups_by_intent() -> None:
    conversations = [
        _conversation("1", "my app crashed", "sorry about that, please update the app"),
        _conversation("2", "cancel my plan", "we're sad to see you go"),
    ]
    classifier = _FakeClassifier({"my app crashed": Intent.PLAYBACK_BUG, "cancel my plan": Intent.CANCELLATION_RETENTION})
    settings = Settings()

    index = build_precedent_index(conversations, classifier, settings)

    assert len(index.by_intent[Intent.PLAYBACK_BUG]) == 1
    assert index.by_intent[Intent.PLAYBACK_BUG][0].resolution == "sorry about that, please update the app"
    assert len(index.by_intent[Intent.CANCELLATION_RETENTION]) == 1


def test_build_precedent_index_skips_conversations_that_fail_to_classify() -> None:
    conversations = [
        _conversation("1", "good one", "resolved"),
        _conversation("2", "bad one", "also resolved"),
    ]
    classifier = _FakeClassifier(
        {"good one": Intent.PLAYBACK_BUG, "bad one": Intent.PLAYBACK_BUG},
        failing_texts=frozenset({"bad one"}),
    )
    settings = Settings()

    index = build_precedent_index(conversations, classifier, settings)

    assert len(index.by_intent[Intent.PLAYBACK_BUG]) == 1
    assert index.by_intent[Intent.PLAYBACK_BUG][0].message.text == "good one"
