from anthropic import Anthropic
from pydantic import BaseModel

from adapters.cache import ResponseCache, hash_request
from config.settings import Settings
from domain.intents import Intent
from domain.models import ClassificationResult, DraftReply, JudgeVerdict, Message, PrecedentIndex

_CLASSIFY_MAX_TOKENS = 400
_DRAFT_MAX_TOKENS = 700
_JUDGE_MAX_TOKENS = 500


class LLMResponseError(Exception):
    pass


class _DraftOutput(BaseModel):
    text: str


def _cached_parse[T: BaseModel](
    client: Anthropic,
    cache: ResponseCache,
    *,
    model: str,
    system: str,
    user_content: str,
    max_tokens: int,
    output_format: type[T],
    error_context: str,
) -> T:
    key = hash_request(
        model,
        system,
        {"user_content": user_content, "schema": output_format.__name__, "max_tokens": max_tokens},
    )

    def compute() -> str:
        response = client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_content}],
            output_format=output_format,
        )
        result = response.parsed_output
        if result is None:
            raise LLMResponseError(error_context)
        return result.model_dump_json()

    return output_format.model_validate_json(cache.get_or_set(key, compute))


class ClaudeClassifier:
    def __init__(self, client: Anthropic, settings: Settings, cache: ResponseCache) -> None:
        self._client = client
        self._settings = settings
        self._cache = cache

    def classify(self, message: Message) -> ClassificationResult:
        system = (
            "You classify customer support messages sent to SpotifyCares into exactly one intent. "
            f"Valid intents: {', '.join(intent.value for intent in Intent)}. "
            "Report your own confidence in the classification as a number between 0 and 1, "
            "with a rationale of one concise sentence."
        )
        return _cached_parse(
            self._client,
            self._cache,
            model=self._settings.haiku_model,
            system=system,
            user_content=message.text,
            max_tokens=_CLASSIFY_MAX_TOKENS,
            output_format=ClassificationResult,
            error_context=f"classifier returned no parseable output for message {message.id}",
        )


class ClaudeReplyDrafter:
    def __init__(
        self,
        client: Anthropic,
        settings: Settings,
        cache: ResponseCache,
        precedents: PrecedentIndex,
    ) -> None:
        self._client = client
        self._settings = settings
        self._cache = cache
        self._precedents = precedents

    def draft(self, message: Message, intent: Intent) -> DraftReply:
        examples = self._precedents.similar(intent, self._settings.precedent_k)
        precedent_text = "\n\n".join(
            f"Customer: {precedent.message.text}\nSpotifyCares: {precedent.resolution}"
            for precedent in examples
        )
        system = (
            "You are SpotifyCares customer support. Draft a helpful, on-brand reply to the "
            "customer's message below. Ground your reply in the style and substance of how "
            "SpotifyCares actually resolved similar issues in the past, shown here:\n\n"
            f"{precedent_text or '(no historical precedent available for this intent)'}"
        )
        output = _cached_parse(
            self._client,
            self._cache,
            model=self._settings.sonnet_model,
            system=system,
            user_content=message.text,
            max_tokens=_DRAFT_MAX_TOKENS,
            output_format=_DraftOutput,
            error_context=f"drafter returned no parseable output for message {message.id}",
        )
        return DraftReply(text=output.text, grounded_on=[precedent.message.id for precedent in examples])


class ClaudeJudge:
    def __init__(self, client: Anthropic, settings: Settings, cache: ResponseCache) -> None:
        self._client = client
        self._settings = settings
        self._cache = cache

    def evaluate(self, message: Message, reply: DraftReply) -> JudgeVerdict:
        system = (
            "You are an impartial evaluator of SpotifyCares customer support replies. "
            "Score how well the reply addresses the customer's message from 1 (poor) to 5 "
            "(excellent), with a rationale of one or two sentences."
        )
        user_content = f"Customer message: {message.text}\n\nSupport reply: {reply.text}"
        return _cached_parse(
            self._client,
            self._cache,
            model=self._settings.sonnet_model,
            system=system,
            user_content=user_content,
            max_tokens=_JUDGE_MAX_TOKENS,
            output_format=JudgeVerdict,
            error_context=f"judge returned no parseable output for message {message.id}",
        )
