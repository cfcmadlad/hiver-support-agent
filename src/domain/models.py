from datetime import datetime
from typing import Protocol

from pydantic import BaseModel

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


class ClassificationResult(BaseModel):
    intent: Intent
    confidence: float
    rationale: str


class DraftReply(BaseModel):
    text: str
    grounded_on: list[str]


class EscalationDecision(BaseModel):
    escalate: bool
    reason: str


class JudgeVerdict(BaseModel):
    score: float
    rationale: str


class Classifier(Protocol):
    def classify(self, message: Message) -> ClassificationResult: ...


class ReplyDrafter(Protocol):
    def draft(self, message: Message, intent: Intent) -> DraftReply: ...


class Judge(Protocol):
    def evaluate(self, message: Message, reply: DraftReply) -> JudgeVerdict: ...
