from pathlib import Path

import pytest

from domain.escalation import EscalationPolicy
from domain.intents import Intent
from domain.models import ClassificationResult, DraftReply, JudgeVerdict, Message
from eval.judge import GoldenExample, evaluate_system, load_golden_set

POLICY = EscalationPolicy(
    confidence_threshold=0.7,
    high_risk_intents=frozenset({Intent.BILLING_DISPUTE, Intent.CANCELLATION_RETENTION}),
)


class _FakeClassifier:
    def __init__(self, intent: Intent, confidence: float) -> None:
        self._intent = intent
        self._confidence = confidence

    def classify(self, message: Message) -> ClassificationResult:
        return ClassificationResult(
            intent=self._intent, confidence=self._confidence, rationale="fake"
        )


class _FakeDrafter:
    def draft(self, message: Message, intent: Intent) -> DraftReply:
        return DraftReply(text=f"reply for {intent.value}", grounded_on=[])


class _FakeJudge:
    def evaluate(self, message: Message, reply: DraftReply) -> JudgeVerdict:
        return JudgeVerdict(score=4.0, rationale="fake")


class _FlakyJudge:
    def evaluate(self, message: Message, reply: DraftReply) -> JudgeVerdict:
        if message.id == "bad":
            raise ValueError("simulated malformed LLM response")
        return JudgeVerdict(score=4.0, rationale="fake")


def _example(conversation_id: str = "1") -> GoldenExample:
    return GoldenExample(
        conversation_id=conversation_id,
        customer_message="my playback keeps crashing",
        true_intent=Intent.PLAYBACK_BUG,
        true_escalate=False,
        notes="",
    )


def test_evaluate_system_applies_escalation_policy() -> None:
    results, failures = evaluate_system(
        [_example()],
        _FakeClassifier(Intent.PLAYBACK_BUG, 0.95),
        _FakeDrafter(),
        _FakeJudge(),
        POLICY,
    )
    assert failures == []
    assert results[0].predicted_escalate is False
    assert results[0].predicted_intent == Intent.PLAYBACK_BUG
    assert results[0].judge_score == 4.0


def test_evaluate_system_high_risk_intent_escalates() -> None:
    results, failures = evaluate_system(
        [_example()],
        _FakeClassifier(Intent.BILLING_DISPUTE, 0.99),
        _FakeDrafter(),
        _FakeJudge(),
        POLICY,
    )
    assert failures == []
    assert results[0].predicted_escalate is True


def test_evaluate_system_none_policy_never_escalates() -> None:
    results, failures = evaluate_system(
        [_example()],
        _FakeClassifier(Intent.BILLING_DISPUTE, 0.99),
        _FakeDrafter(),
        _FakeJudge(),
        None,
    )
    assert failures == []
    assert results[0].predicted_escalate is False
    assert results[0].escalation_reason == "baseline: always auto-handled"


def test_evaluate_system_isolates_one_failure_from_the_rest() -> None:
    examples = [_example("good-1"), _example("bad"), _example("good-2")]
    results, failures = evaluate_system(
        examples, _FakeClassifier(Intent.PLAYBACK_BUG, 0.95), _FakeDrafter(), _FlakyJudge(), POLICY
    )
    assert {r.conversation_id for r in results} == {"good-1", "good-2"}
    assert len(failures) == 1
    assert failures[0].conversation_id == "bad"
    assert "simulated malformed LLM response" in failures[0].error


def test_load_golden_set(tmp_path: Path) -> None:
    golden_path = tmp_path / "golden.csv"
    golden_path.write_text(
        "conversation_id,customer_message,true_intent,true_escalate,notes,labeled_by,labeled_at\n"
        "1,help my app crashed,playback_bug,False,minor bug,tester,2026-01-01\n"
        "2,cancel my plan,cancellation_retention,True,wants refund,tester,2026-01-01\n",
        encoding="utf-8",
    )
    examples = load_golden_set(golden_path)
    assert len(examples) == 2
    assert examples[0].true_intent == Intent.PLAYBACK_BUG
    assert examples[0].true_escalate is False
    assert examples[1].true_escalate is True


def test_load_golden_set_rejects_unrecognized_escalate_value(tmp_path: Path) -> None:
    golden_path = tmp_path / "golden.csv"
    golden_path.write_text(
        "conversation_id,customer_message,true_intent,true_escalate,notes,labeled_by,labeled_at\n"
        "1,help my app crashed,playback_bug,yes,typo'd value,tester,2026-01-01\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="true_escalate"):
        load_golden_set(golden_path)


def test_load_golden_set_handles_utf8_bom(tmp_path: Path) -> None:
    golden_path = tmp_path / "golden.csv"
    content = (
        "conversation_id,customer_message,true_intent,true_escalate,notes,labeled_by,labeled_at\n"
        "1,help my app crashed,playback_bug,False,,tester,2026-01-01\n"
    )
    golden_path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))

    examples = load_golden_set(golden_path)
    assert len(examples) == 1
    assert examples[0].conversation_id == "1"
