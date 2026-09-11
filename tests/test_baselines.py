from datetime import UTC, datetime

from domain.intents import Intent
from domain.models import Message
from eval.baselines import CannedReplyDrafter, KeywordClassifier


def _message(text: str) -> Message:
    return Message(id="1", author_id="customer", text=text, created_at=datetime.now(UTC), in_reply_to_id=None)


def test_classifies_billing_dispute() -> None:
    result = KeywordClassifier().classify(_message("I was charged twice for my subscription"))
    assert result.intent == Intent.BILLING_DISPUTE


def test_classifies_playback_bug() -> None:
    result = KeywordClassifier().classify(_message("The app keeps buffering and songs won't play"))
    assert result.intent == Intent.PLAYBACK_BUG


def test_classifies_account_access() -> None:
    result = KeywordClassifier().classify(_message("I'm locked out and can't reset my password"))
    assert result.intent == Intent.ACCOUNT_ACCESS


def test_classifies_cancellation_retention() -> None:
    result = KeywordClassifier().classify(_message("I want to cancel my subscription"))
    assert result.intent == Intent.CANCELLATION_RETENTION


def test_classifies_feature_question() -> None:
    result = KeywordClassifier().classify(_message("How do I create a collaborative playlist?"))
    assert result.intent == Intent.FEATURE_QUESTION


def test_no_keyword_match_falls_back_to_other() -> None:
    result = KeywordClassifier().classify(_message("Just wanted to say I love the app!"))
    assert result.intent == Intent.OTHER


def test_keyword_inside_unrelated_word_does_not_match() -> None:
    result = KeywordClassifier().classify(_message("My phone charger died, can't play music"))
    assert result.intent != Intent.BILLING_DISPUTE


def test_keyword_substring_inside_another_word_does_not_match() -> None:
    result = KeywordClassifier().classify(_message("Can you flag this for the team?"))
    assert result.intent != Intent.PLAYBACK_BUG


def test_whole_word_keyword_still_matches() -> None:
    result = KeywordClassifier().classify(_message("I was charged twice this month"))
    assert result.intent == Intent.BILLING_DISPUTE


def test_canned_reply_has_no_grounding() -> None:
    reply = CannedReplyDrafter().draft(_message("anything"), Intent.PLAYBACK_BUG)
    assert reply.grounded_on == []
    assert len(reply.text) > 0


def test_canned_reply_covers_every_intent() -> None:
    drafter = CannedReplyDrafter()
    for intent in Intent:
        reply = drafter.draft(_message("anything"), intent)
        assert reply.text
