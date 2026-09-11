from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from anthropic import Anthropic

from config.settings import MissingAPIKeyError, Settings
from domain.escalation import EscalationPolicy, decide_escalation
from domain.intents import Intent
from domain.models import (
    ClassificationResult,
    Classifier,
    Conversation,
    DraftReply,
    EscalationDecision,
    Message,
    Precedent,
    PrecedentIndex,
    ReplyDrafter,
)


def build_client(settings: Settings) -> Anthropic:
    if not settings.anthropic_api_key:
        raise MissingAPIKeyError("ANTHROPIC_API_KEY is not set")
    return Anthropic(api_key=settings.anthropic_api_key)


def build_escalation_policy(settings: Settings) -> EscalationPolicy:
    return EscalationPolicy(
        confidence_threshold=settings.confidence_threshold,
        high_risk_intents=settings.high_risk_intents,
    )


def _classify_or_none(classifier: Classifier, message: Message) -> ClassificationResult | None:
    try:
        return classifier.classify(message)
    except Exception:  # noqa: BLE001
        # Isolates one bad candidate (malformed LLM response, transient API error) from the
        # rest of the pool instead of losing every already-dispatched concurrent call over it.
        # A skipped candidate just means one fewer precedent in that intent's bucket, which is
        # low-stakes (precedents are few-shot examples, not a reported result), so a silent
        # skip plus a one-line count is proportionate here, unlike evaluate_system's results.
        return None


def build_precedent_index(
    conversations: list[Conversation], classifier: Classifier, settings: Settings
) -> PrecedentIndex:
    candidates: list[tuple[Message, Message]] = []
    for conversation in conversations:
        root_message = conversation.root_customer_message(settings.brand_author_id)
        resolution_message = conversation.last_brand_message(settings.brand_author_id)
        if root_message is None or resolution_message is None:
            continue
        candidates.append((root_message, resolution_message))

    root_messages = [root_message for root_message, _ in candidates]
    with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
        classifications = list(
            executor.map(lambda message: _classify_or_none(classifier, message), root_messages)
        )

    by_intent: dict[Intent, list[Precedent]] = {intent: [] for intent in Intent}
    skipped = 0
    for (root_message, resolution_message), classification in zip(candidates, classifications):
        if classification is None:
            skipped += 1
            continue
        by_intent[classification.intent].append(
            Precedent(message=root_message, resolution=resolution_message.text)
        )
    if skipped:
        print(f"build_precedent_index: skipped {skipped} candidate(s) that failed to classify")
    return PrecedentIndex(by_intent=by_intent)


@dataclass(frozen=True)
class PipelineResult:
    classification: ClassificationResult
    draft: DraftReply
    escalation: EscalationDecision


class Pipeline:
    def __init__(
        self,
        classifier: Classifier,
        drafter: ReplyDrafter,
        escalation_policy: EscalationPolicy,
    ) -> None:
        self._classifier = classifier
        self._drafter = drafter
        self._escalation_policy = escalation_policy

    def process_message(self, message: Message) -> PipelineResult:
        classification = self._classifier.classify(message)
        draft = self._drafter.draft(message, classification.intent)
        escalation = decide_escalation(classification, self._escalation_policy)
        return PipelineResult(classification=classification, draft=draft, escalation=escalation)
