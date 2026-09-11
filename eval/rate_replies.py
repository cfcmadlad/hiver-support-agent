import csv
from pathlib import Path

import click
import numpy as np
import pandas as pd

from domain.metrics import cohens_kappa, raw_agreement

_RATE_COUNT = 30
_SEED = 123
_FIELDNAMES = ["conversation_id", "customer_message", "draft_text", "human_score"]


def _already_rated(ratings_path: Path) -> set[str]:
    if not ratings_path.exists():
        return set()
    with ratings_path.open(newline="", encoding="utf-8") as f:
        return {row["conversation_id"] for row in csv.DictReader(f)}


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option(
    "--full-agent-csv", default=Path("data/eval/full_agent.csv"), type=click.Path(path_type=Path)
)
@click.option(
    "--ratings-csv",
    default=Path("data/eval/human_reply_ratings.csv"),
    type=click.Path(path_type=Path),
)
@click.option("--count", default=_RATE_COUNT, show_default=True, help="How many replies to rate.")
def rate(full_agent_csv: Path, ratings_csv: Path, count: int) -> None:
    """Rate a sample of full-agent draft replies 1-5, blind to the judge's own score.

    Independence from the LLM judge is the entire point, so this never shows judge_score
    or judge_rationale while you rate.
    """
    df = pd.read_csv(full_agent_csv)
    sample = df.sample(n=min(count, len(df)), random_state=_SEED)

    done = _already_rated(ratings_csv)
    remaining = sample[~sample["conversation_id"].astype(str).isin(done)]

    write_header = not ratings_csv.exists()
    click.echo(f"{len(done)} already rated, {len(remaining)} remaining out of {len(sample)}.")

    ratings_csv.parent.mkdir(parents=True, exist_ok=True)
    with ratings_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if write_header:
            writer.writeheader()

        try:
            for _, row in remaining.iterrows():
                click.echo("\n" + "=" * 60)
                click.echo(f"customer: {row['customer_message']}")
                click.echo("-" * 60)
                click.echo(f"reply:    {row['draft_text']}")
                click.echo("=" * 60)

                score = click.prompt(
                    "Reply quality, 1 (poor) to 5 (excellent)", type=click.IntRange(1, 5)
                )
                writer.writerow(
                    {
                        "conversation_id": row["conversation_id"],
                        "customer_message": row["customer_message"],
                        "draft_text": row["draft_text"],
                        "human_score": score,
                    }
                )
                f.flush()
        except click.exceptions.Abort:
            click.echo("\nStopped. Progress is saved, run again to resume.")


@cli.command()
@click.option(
    "--full-agent-csv", default=Path("data/eval/full_agent.csv"), type=click.Path(path_type=Path)
)
@click.option(
    "--ratings-csv",
    default=Path("data/eval/human_reply_ratings.csv"),
    type=click.Path(path_type=Path),
)
def report(full_agent_csv: Path, ratings_csv: Path) -> None:
    """Compute agreement between human reply-quality ratings and the LLM judge's scores."""
    ratings = pd.read_csv(ratings_csv)
    judged = pd.read_csv(full_agent_csv)[["conversation_id", "judge_score"]]
    merged = ratings.merge(judged, on="conversation_id", how="inner")

    if len(merged) < len(ratings):
        click.echo(
            f"warning: {len(ratings) - len(merged)} rated conversation(s) not found in {full_agent_csv}"
        )

    n = len(merged)
    human = merged["human_score"].astype(float).to_numpy()
    judge = merged["judge_score"].astype(float).to_numpy()

    correlation = float(np.corrcoef(human, judge)[0, 1])
    mean_abs_diff = float(np.mean(np.abs(human - judge)))
    within_one = float(np.mean(np.abs(human - judge) <= 1.0))

    human_binary = [h >= 4 for h in human]
    judge_binary = [j >= 4 for j in judge]
    binary_raw = raw_agreement(human_binary, judge_binary)
    binary_kappa = cohens_kappa(human_binary, judge_binary)

    click.echo(f"n={n}")
    click.echo(f"Pearson correlation (human vs judge, 1-5 scores): {correlation:.3f}")
    click.echo(f"mean absolute difference: {mean_abs_diff:.2f}")
    click.echo(f"agree within 1 point: {within_one:.3f}")
    click.echo(
        f"'good reply' (score>=4) raw agreement: {binary_raw:.3f}  kappa: {binary_kappa:.3f}"
    )


if __name__ == "__main__":
    cli()
