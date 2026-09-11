import csv
import random
from datetime import UTC, datetime
from pathlib import Path

import click

from adapters.dataset_twitter import load_conversations
from config.settings import Settings
from domain.intents import Intent

_GOLDEN_SET_SIZE = 200
_FIELDNAMES = [
    "conversation_id",
    "customer_message",
    "true_intent",
    "true_escalate",
    "notes",
    "labeled_by",
    "labeled_at",
]


def _already_labeled(golden_path: Path) -> set[str]:
    if not golden_path.exists():
        return set()
    with golden_path.open(newline="", encoding="utf-8") as f:
        return {row["conversation_id"] for row in csv.DictReader(f)}


def _prompt_intent() -> Intent:
    intents = list(Intent)
    click.echo("Intent:")
    for i, intent in enumerate(intents, start=1):
        click.echo(f"  {i}. {intent.value}")
    choice = click.prompt("Choose", type=click.IntRange(1, len(intents)))
    return intents[choice - 1]


@click.command()
@click.option("--labeled-by", default="aditya1325.raps@gmail.com", show_default=True)
def label(labeled_by: str) -> None:
    settings = Settings()
    settings.golden_path.parent.mkdir(parents=True, exist_ok=True)

    conversations = load_conversations(settings.subsample_path)
    sample = random.Random(settings.seed).sample(
        conversations, min(_GOLDEN_SET_SIZE, len(conversations))
    )

    done = _already_labeled(settings.golden_path)
    remaining = [c for c in sample if c.conversation_id not in done]

    write_header = not settings.golden_path.exists()
    click.echo(f"{len(done)} already labeled, {len(remaining)} remaining out of {len(sample)}.")

    with settings.golden_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if write_header:
            writer.writeheader()

        try:
            for conversation in remaining:
                root_message = conversation.root_customer_message(settings.brand_author_id)
                if root_message is None:
                    continue
                customer_message = root_message.text

                click.echo("\n" + "=" * 60)
                click.echo(f"conversation {conversation.conversation_id}")
                click.echo(customer_message)
                click.echo("=" * 60)

                intent = _prompt_intent()
                escalate = click.confirm("Should this have been escalated to a human?")
                notes = click.prompt("Notes (optional)", default="", show_default=False)

                writer.writerow(
                    {
                        "conversation_id": conversation.conversation_id,
                        "customer_message": customer_message,
                        "true_intent": intent.value,
                        "true_escalate": escalate,
                        "notes": notes,
                        "labeled_by": labeled_by,
                        "labeled_at": datetime.now(UTC).isoformat(),
                    }
                )
                f.flush()
        except click.exceptions.Abort:
            click.echo("\nStopped. Progress is saved, run again to resume.")


if __name__ == "__main__":
    label()
