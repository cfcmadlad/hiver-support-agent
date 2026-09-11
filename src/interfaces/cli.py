import random
from datetime import UTC, datetime

import click

from adapters.cache import ResponseCache
from adapters.dataset_twitter import load_conversations
from adapters.llm_claude import ClaudeClassifier, ClaudeReplyDrafter
from config.settings import Settings
from domain.models import Message
from orchestration.pipeline import (
    Pipeline,
    build_client,
    build_escalation_policy,
    build_precedent_index,
)


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("text")
def classify(text: str) -> None:
    settings = Settings.from_env()
    client = build_client(settings)
    cache = ResponseCache(settings.cache_dir)
    classifier = ClaudeClassifier(client, settings, cache)

    message = Message(
        id="adhoc", author_id="cli", text=text, created_at=datetime.now(UTC), in_reply_to_id=None
    )
    result = classifier.classify(message)
    click.echo(f"intent: {result.intent.value} (confidence {result.confidence:.2f})")
    click.echo(f"rationale: {result.rationale}")


@cli.command()
@click.option(
    "--count",
    default=3,
    show_default=True,
    help="Number of seeded subsample conversations to run end-to-end.",
)
@click.option(
    "--precedent-pool-size",
    default=20,
    show_default=True,
    help="Conversations classified to build reply-grounding precedents from. "
    "Disjoint from --count so a conversation can never be shown its own resolution.",
)
def run(count: int, precedent_pool_size: int) -> None:
    settings = Settings.from_env()
    client = build_client(settings)
    cache = ResponseCache(settings.cache_dir)

    all_conversations = load_conversations(settings.subsample_path)
    shuffled = random.Random(settings.seed).sample(all_conversations, len(all_conversations))
    conversations = shuffled[:count]
    precedent_pool = shuffled[count : count + precedent_pool_size]

    classifier = ClaudeClassifier(client, settings, cache)
    precedents = build_precedent_index(precedent_pool, classifier, settings)
    drafter = ClaudeReplyDrafter(client, settings, cache, precedents)
    policy = build_escalation_policy(settings)
    pipeline = Pipeline(classifier, drafter, policy)

    for conversation in conversations:
        message = conversation.root_customer_message(settings.brand_author_id)
        if message is None:
            continue
        result = pipeline.process_message(message)

        click.echo(f"--- conversation {conversation.conversation_id} ---")
        click.echo(f"customer: {message.text}")
        click.echo(
            f"intent: {result.classification.intent.value} "
            f"(confidence {result.classification.confidence:.2f})"
        )
        click.echo(f"draft reply: {result.draft.text}")
        click.echo(f"escalate: {result.escalation.escalate} ({result.escalation.reason})")


if __name__ == "__main__":
    cli()
