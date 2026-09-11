from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, Field

from domain.intents import Intent


class Message(BaseModel):
    id: str
    author_id: str
    text: str
    created_at: datetime
    in_reply_to_id: str | None


class Conversation(BaseModel):
    conversation_id: str
    messages: list[Message]

    def root_customer_message(self, brand_author_id: str) -> "Message | None":
        for message in self.messages:
            if message.author_id != brand_author_id:
                return message
        return None

    def last_brand_message(self, brand_author_id: str) -> "Message | None":
        for message in reversed(self.messages):
            if message.author_id == brand_author_id:
                return message
        return None


class ClassificationResult(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0, le=1)
    rationale: str


class DraftReply(BaseModel):
    text: str
    grounded_on: list[str]


class EscalationDecision(BaseModel):
    escalate: bool
    reason: str


class Precedent(BaseModel):
    message: Message
    resolution: str


class PrecedentIndex(BaseModel):
    by_intent: dict[Intent, list[Precedent]]

    def similar(self, intent: Intent, k: int) -> list[Precedent]:
        return self.by_intent.get(intent, [])[:k]


class JudgeVerdict(BaseModel):
    score: float
    rationale: str


class Classifier(Protocol):
    def classify(self, message: Message) -> ClassificationResult: ...


class ReplyDrafter(Protocol):
    def draft(self, message: Message, intent: Intent) -> DraftReply: ...


class Judge(Protocol):
    def evaluate(self, message: Message, reply: DraftReply) -> JudgeVerdict: ...
