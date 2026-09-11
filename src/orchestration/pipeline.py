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


def build_precedent_index(
    conversations: list[Conversation], classifier: Classifier, settings: Settings
) -> PrecedentIndex:
    by_intent: dict[Intent, list[Precedent]] = {intent: [] for intent in Intent}
    for conversation in conversations:
        root_message = conversation.root_customer_message(settings.brand_author_id)
        resolution_message = conversation.last_brand_message(settings.brand_author_id)
        if root_message is None or resolution_message is None:
            continue
        classification = classifier.classify(root_message)
        by_intent[classification.intent].append(
            Precedent(message=root_message, resolution=resolution_message.text)
        )
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
