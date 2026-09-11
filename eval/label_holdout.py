import csv
import random
from datetime import UTC, datetime
from pathlib import Path

import click

from adapters.dataset_twitter import load_conversations
from config.settings import Settings
from domain.intents import Intent

_HOLDOUT_SIZE = 40
_HOLDOUT_SEED = 4242
_FIELDNAMES = [
    "conversation_id",
    "customer_message",
    "true_intent",
    "true_escalate",
    "notes",
    "labeled_by",
    "labeled_at",
]


def _already_labeled(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
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
@click.option("--count", default=_HOLDOUT_SIZE, show_default=True)
def label_holdout(labeled_by: str, count: int) -> None:
    """Label a fresh sample disjoint from data/golden/golden_set.csv.

    Exists to validate the escalation-policy fix out of sample: the account_access
    high-risk change was chosen by looking at golden_set.csv's own failure analysis,
    so re-measuring on golden_set.csv again would not show whether it generalizes.
    """
    settings = Settings()
    holdout_path = settings.golden_path.parent / "holdout_set.csv"
    holdout_path.parent.mkdir(parents=True, exist_ok=True)

    conversations = load_conversations(settings.subsample_path)
    golden_ids = _already_labeled(settings.golden_path)
    eligible = [c for c in conversations if c.conversation_id not in golden_ids]
    sample = random.Random(_HOLDOUT_SEED).sample(eligible, min(count, len(eligible)))

    done = _already_labeled(holdout_path)
    remaining = [c for c in sample if c.conversation_id not in done]

    write_header = not holdout_path.exists()
    click.echo(f"{len(done)} already labeled, {len(remaining)} remaining out of {len(sample)}.")

    with holdout_path.open("a", newline="", encoding="utf-8") as f:
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
    label_holdout()
