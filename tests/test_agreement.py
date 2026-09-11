import pytest

from domain.intents import Intent
from eval.agreement import compute_agreement
from eval.judge import GoldenEvalResult


def _result(
    conversation_id: str,
    true_intent: Intent,
    predicted_intent: Intent,
    true_escalate: bool,
    predicted_escalate: bool,
    judge_score: float,
) -> GoldenEvalResult:
    return GoldenEvalResult(
        conversation_id=conversation_id,
        customer_message="text",
        true_intent=true_intent,
        true_escalate=true_escalate,
        predicted_intent=predicted_intent,
        predicted_confidence=0.9,
        predicted_escalate=predicted_escalate,
        escalation_reason="reason",
        draft_text="reply",
        judge_score=judge_score,
        judge_rationale="rationale",
    )


def test_compute_agreement_perfect_match() -> None:
    results = [
        _result("1", Intent.PLAYBACK_BUG, Intent.PLAYBACK_BUG, False, False, 5.0),
        _result("2", Intent.BILLING_DISPUTE, Intent.BILLING_DISPUTE, True, True, 3.0),
    ]
    report = compute_agreement(results)
    assert report.n == 2
    assert report.intent_raw_agreement == pytest.approx(1.0)
    assert report.intent_kappa == pytest.approx(1.0)
    assert report.escalate_raw_agreement == pytest.approx(1.0)
    assert report.escalate_kappa == pytest.approx(1.0)
    assert report.mean_judge_score == pytest.approx(4.0)


def test_compute_agreement_partial_mismatch() -> None:
    results = [
        _result("1", Intent.PLAYBACK_BUG, Intent.PLAYBACK_BUG, False, False, 4.0),
        _result("2", Intent.BILLING_DISPUTE, Intent.OTHER, True, False, 2.0),
    ]
    report = compute_agreement(results)
    assert report.intent_raw_agreement == pytest.approx(0.5)
    assert report.escalate_raw_agreement == pytest.approx(0.5)
    assert report.mean_judge_score == pytest.approx(3.0)


def test_compute_agreement_rejects_empty_results() -> None:
    with pytest.raises(ValueError):
        compute_agreement([])
