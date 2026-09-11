import csv
import random
from dataclasses import asdict
from pathlib import Path

import click

from adapters.cache import ResponseCache
from adapters.dataset_twitter import load_conversations
from adapters.llm_claude import ClaudeClassifier, ClaudeJudge, ClaudeReplyDrafter
from config.settings import Settings
from domain.escalation import EscalationPolicy
from domain.models import Classifier, Judge, PrecedentIndex, ReplyDrafter
from eval.agreement import compute_agreement
from eval.baselines import CannedReplyDrafter, KeywordClassifier
from eval.judge import (
    EvalFailure,
    GoldenEvalResult,
    GoldenExample,
    evaluate_system,
    load_golden_set,
)
from orchestration.pipeline import build_client, build_escalation_policy, build_precedent_index

_RESULT_FIELDNAMES = list(GoldenEvalResult.__dataclass_fields__)
_FAILURE_FIELDNAMES = list(EvalFailure.__dataclass_fields__)


def _write_results(path: Path, results: list[GoldenEvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_RESULT_FIELDNAMES)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["true_intent"] = result.true_intent.value
            row["predicted_intent"] = result.predicted_intent.value
            writer.writerow(row)


def _write_failures(path: Path, failures: list[EvalFailure]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FAILURE_FIELDNAMES)
        writer.writeheader()
        for failure in failures:
            writer.writerow(asdict(failure))


def _run_and_report(
    label: str,
    filename: str,
    examples: list[GoldenExample],
    classifier: Classifier,
    drafter: ReplyDrafter,
    judge: Judge,
    policy: EscalationPolicy | None,
    max_workers: int,
    results_dir: Path,
) -> None:
    click.echo(f"Evaluating {label} on {len(examples)} golden examples...")
    results, failures = evaluate_system(examples, classifier, drafter, judge, policy, max_workers)

    failures_path = results_dir / f"{filename}_failures.csv"
    if failures:
        failed_ids = ", ".join(f.conversation_id for f in failures)
        click.echo(f"  {len(failures)} example(s) failed and were excluded: {failed_ids}")
        _write_failures(failures_path, failures)
    elif failures_path.exists():
        # Clear a stale failures file from a previous run so it never misrepresents this run.
        failures_path.unlink()

    if not results:
        click.echo(f"  no successful results for {label}")
        return
    click.echo(f"--- {label} ---")
    report = compute_agreement(results)
    click.echo(f"n={report.n}")
    click.echo(
        f"intent raw agreement: {report.intent_raw_agreement:.3f}  kappa: {report.intent_kappa:.3f}"
    )
    click.echo(
        f"escalate raw agreement: {report.escalate_raw_agreement:.3f}  kappa: {report.escalate_kappa:.3f}"
    )
    click.echo(f"mean judge score: {report.mean_judge_score:.2f}")
    _write_results(results_dir / f"{filename}.csv", results)


@click.command()
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Evaluate only the first N golden examples, for a cheap smoke test before a full run.",
)
@click.option(
    "--precedent-pool-size",
    default=50,
    show_default=True,
    help="Non-golden conversations classified to build the full agent's reply-grounding precedents.",
)
@click.option(
    "--results-dir",
    default=Path("data/eval"),
    type=click.Path(path_type=Path),
    show_default=True,
)
@click.option(
    "--golden-path",
    default=None,
    type=click.Path(path_type=Path),
    help="Override the golden set to evaluate against, e.g. a held-out validation set. "
    "Defaults to settings.golden_path.",
)
def evaluate(
    limit: int | None, precedent_pool_size: int, results_dir: Path, golden_path: Path | None
) -> None:
    settings = Settings.from_env()
    client = build_client(settings)
    cache = ResponseCache(settings.cache_dir)

    examples = load_golden_set(golden_path or settings.golden_path)
    if limit is not None:
        examples = examples[:limit]
    click.echo(f"Loaded {len(examples)} golden examples.")

    golden_ids = {example.conversation_id for example in examples}
    all_conversations = load_conversations(settings.subsample_path)
    non_golden = [c for c in all_conversations if c.conversation_id not in golden_ids]
    precedent_pool = random.Random(settings.seed).sample(
        non_golden, min(precedent_pool_size, len(non_golden))
    )

    classifier: Classifier = ClaudeClassifier(client, settings, cache)
    precedents = build_precedent_index(precedent_pool, classifier, settings)
    drafter: ReplyDrafter = ClaudeReplyDrafter(client, settings, cache, precedents)
    judge = ClaudeJudge(client, settings, cache)
    policy = build_escalation_policy(settings)

    _run_and_report(
        "full agent",
        "full_agent",
        examples,
        classifier,
        drafter,
        judge,
        policy,
        settings.max_workers,
        results_dir,
    )

    ungrounded_settings = settings.model_copy(update={"sonnet_model": settings.haiku_model})
    baseline2_drafter = ClaudeReplyDrafter(
        client, ungrounded_settings, cache, PrecedentIndex(by_intent={})
    )
    _run_and_report(
        "baseline 2: zero-shot ungrounded Haiku",
        "baseline2",
        examples,
        classifier,
        baseline2_drafter,
        judge,
        policy,
        settings.max_workers,
        results_dir,
    )

    _run_and_report(
        "baseline 1: keyword + canned reply",
        "baseline1",
        examples,
        KeywordClassifier(),
        CannedReplyDrafter(),
        judge,
        None,
        settings.max_workers,
        results_dir,
    )


if __name__ == "__main__":
    evaluate()
