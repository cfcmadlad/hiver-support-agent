import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from domain.escalation import EscalationPolicy, decide_escalation
from domain.intents import Intent
from domain.models import Classifier, EscalationDecision, Judge, Message, ReplyDrafter


@dataclass(frozen=True)
class GoldenExample:
    conversation_id: str
    customer_message: str
    true_intent: Intent
    true_escalate: bool
    notes: str


@dataclass(frozen=True)
class GoldenEvalResult:
    conversation_id: str
    customer_message: str
    true_intent: Intent
    true_escalate: bool
    predicted_intent: Intent
    predicted_confidence: float
    predicted_escalate: bool
    escalation_reason: str
    draft_text: str
    judge_score: float
    judge_rationale: str


@dataclass(frozen=True)
class EvalFailure:
    conversation_id: str
    error: str


def _parse_bool(raw: str, conversation_id: str, field: str) -> bool:
    lowered = raw.strip().lower()
    if lowered not in ("true", "false"):
        raise ValueError(f"conversation {conversation_id}: unrecognized {field} value {raw!r}")
    return lowered == "true"


def load_golden_set(path: Path) -> list[GoldenExample]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return [
            GoldenExample(
                conversation_id=row["conversation_id"],
                customer_message=row["customer_message"],
                true_intent=Intent(row["true_intent"]),
                true_escalate=_parse_bool(
                    row["true_escalate"], row["conversation_id"], "true_escalate"
                ),
                notes=row["notes"],
            )
            for row in csv.DictReader(f)
        ]


def _evaluate_one(
    example: GoldenExample,
    classifier: Classifier,
    drafter: ReplyDrafter,
    judge: Judge,
    escalation_policy: EscalationPolicy | None,
) -> GoldenEvalResult | EvalFailure:
    message = Message(
        id=example.conversation_id,
        author_id="golden",
        text=example.customer_message,
        created_at=datetime.now(UTC),
        in_reply_to_id=None,
    )
    try:
        classification = classifier.classify(message)
        draft = drafter.draft(message, classification.intent)
        escalation: EscalationDecision
        if escalation_policy is None:
            escalation = EscalationDecision(escalate=False, reason="baseline: always auto-handled")
        else:
            escalation = decide_escalation(classification, escalation_policy)
        verdict = judge.evaluate(message, draft)
    except Exception as e:  # noqa: BLE001
        # Broad on purpose: isolates one bad example (malformed LLM response, transient API
        # error) from the rest of the batch instead of crashing an 80+ example run over it.
        return EvalFailure(conversation_id=example.conversation_id, error=str(e))

    return GoldenEvalResult(
        conversation_id=example.conversation_id,
        customer_message=example.customer_message,
        true_intent=example.true_intent,
        true_escalate=example.true_escalate,
        predicted_intent=classification.intent,
        predicted_confidence=classification.confidence,
        predicted_escalate=escalation.escalate,
        escalation_reason=escalation.reason,
        draft_text=draft.text,
        judge_score=verdict.score,
        judge_rationale=verdict.rationale,
    )


def evaluate_system(
    examples: list[GoldenExample],
    classifier: Classifier,
    drafter: ReplyDrafter,
    judge: Judge,
    escalation_policy: EscalationPolicy | None,
    max_workers: int = 8,
) -> tuple[list[GoldenEvalResult], list[EvalFailure]]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        outcomes = list(
            executor.map(
                lambda example: _evaluate_one(
                    example, classifier, drafter, judge, escalation_policy
                ),
                examples,
            )
        )

    results = [o for o in outcomes if isinstance(o, GoldenEvalResult)]
    failures = [o for o in outcomes if isinstance(o, EvalFailure)]
    return results, failures
