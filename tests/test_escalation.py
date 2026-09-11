from domain.escalation import EscalationPolicy, decide_escalation
from domain.intents import Intent
from domain.models import ClassificationResult

POLICY = EscalationPolicy(
    confidence_threshold=0.7,
    high_risk_intents=frozenset({Intent.BILLING_DISPUTE, Intent.CANCELLATION_RETENTION}),
)


def _classification(intent: Intent, confidence: float) -> ClassificationResult:
    return ClassificationResult(intent=intent, confidence=confidence, rationale="test")


def test_low_confidence_escalates() -> None:
    result = decide_escalation(_classification(Intent.PLAYBACK_BUG, 0.5), POLICY)
    assert result.escalate is True
    assert "confidence" in result.reason


def test_high_confidence_low_risk_auto_handles() -> None:
    result = decide_escalation(_classification(Intent.PLAYBACK_BUG, 0.95), POLICY)
    assert result.escalate is False


def test_high_risk_intent_escalates_even_at_high_confidence() -> None:
    result = decide_escalation(_classification(Intent.BILLING_DISPUTE, 0.99), POLICY)
    assert result.escalate is True
    assert "high-risk" in result.reason


def test_confidence_exactly_at_threshold_does_not_escalate_on_confidence_alone() -> None:
    result = decide_escalation(_classification(Intent.FEATURE_QUESTION, 0.7), POLICY)
    assert result.escalate is False


def test_both_reasons_are_reported() -> None:
    result = decide_escalation(_classification(Intent.CANCELLATION_RETENTION, 0.4), POLICY)
    assert result.escalate is True
    assert "confidence" in result.reason
    assert "high-risk" in result.reason
