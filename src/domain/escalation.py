from dataclasses import dataclass

from domain.intents import Intent
from domain.models import ClassificationResult, EscalationDecision


@dataclass(frozen=True)
class EscalationPolicy:
    confidence_threshold: float
    high_risk_intents: frozenset[Intent]


def decide_escalation(
    classification: ClassificationResult, policy: EscalationPolicy
) -> EscalationDecision:
    reasons: list[str] = []
    if classification.confidence < policy.confidence_threshold:
        reasons.append(
            f"confidence {classification.confidence:.2f} below threshold "
            f"{policy.confidence_threshold:.2f}"
        )
    if classification.intent in policy.high_risk_intents:
        reasons.append(f"intent {classification.intent.value} is high-risk")

    if reasons:
        return EscalationDecision(escalate=True, reason="; ".join(reasons))
    return EscalationDecision(escalate=False, reason="confident classification, low-risk intent")
