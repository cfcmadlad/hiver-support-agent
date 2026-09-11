from collections.abc import Hashable, Sequence
from pathlib import Path

import click
import numpy as np
import pandas as pd

from domain.escalation import EscalationPolicy, decide_escalation
from domain.intents import Intent
from domain.metrics import cohens_kappa
from domain.models import ClassificationResult

_N_RESAMPLES = 2000
_SEED = 7
_OLD_HIGH_RISK = frozenset({Intent.BILLING_DISPUTE, Intent.CANCELLATION_RETENTION})
_NEW_HIGH_RISK = frozenset(
    {Intent.BILLING_DISPUTE, Intent.CANCELLATION_RETENTION, Intent.ACCOUNT_ACCESS}
)


def _bootstrap_kappa[T: Hashable](
    true_labels: Sequence[T], pred_labels: Sequence[T], rng: np.random.Generator
) -> list[float]:
    n = len(true_labels)
    kappas = []
    for _ in range(_N_RESAMPLES):
        idx = rng.integers(0, n, size=n)
        t = [true_labels[i] for i in idx]
        p = [pred_labels[i] for i in idx]
        try:
            kappas.append(cohens_kappa(t, p))
        except ValueError:
            continue
    return kappas


def _ci(values: list[float]) -> tuple[float, float, float]:
    arr = np.array(values)
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _escalate_for_policy(df: pd.DataFrame, high_risk: frozenset[Intent]) -> list[bool]:
    policy = EscalationPolicy(confidence_threshold=0.7, high_risk_intents=high_risk)
    results = []
    for _, row in df.iterrows():
        classification = ClassificationResult(
            intent=Intent(row["predicted_intent"]),
            confidence=row["predicted_confidence"],
            rationale="",
        )
        results.append(decide_escalation(classification, policy).escalate)
    return results


def _report_kappa_ci[T: Hashable](
    label: str, true_labels: list[T], pred_labels: list[T], rng: np.random.Generator
) -> None:
    point = cohens_kappa(true_labels, pred_labels)
    mean, lo, hi = _ci(_bootstrap_kappa(true_labels, pred_labels, rng))
    click.echo(
        f"{label:<36} point {point:.3f}  bootstrap mean {mean:.3f}  95% CI [{lo:.3f}, {hi:.3f}]"
    )


@click.command()
@click.option(
    "--results-dir", default=Path("data/eval"), type=click.Path(path_type=Path), show_default=True
)
def main(results_dir: Path) -> None:
    rng = np.random.default_rng(_SEED)
    full_agent = pd.read_csv(results_dir / "full_agent.csv")
    baseline1 = pd.read_csv(results_dir / "baseline1.csv")

    click.echo(f"Bootstrap 95% CIs, {_N_RESAMPLES} resamples, seed {_SEED}\n")

    _report_kappa_ci(
        "Full agent intent kappa:",
        list(full_agent["true_intent"]),
        list(full_agent["predicted_intent"]),
        rng,
    )
    _report_kappa_ci(
        "Baseline 1 intent kappa:",
        list(baseline1["true_intent"]),
        list(baseline1["predicted_intent"]),
        rng,
    )

    true_escalate = list(full_agent["true_escalate"].astype(bool))
    old_escalate = _escalate_for_policy(full_agent, _OLD_HIGH_RISK)
    new_escalate = _escalate_for_policy(full_agent, _NEW_HIGH_RISK)

    _report_kappa_ci("Escalate kappa, old policy:", true_escalate, old_escalate, rng)
    _report_kappa_ci("Escalate kappa, new policy:", true_escalate, new_escalate, rng)

    n = len(full_agent)
    diffs = []
    for _ in range(_N_RESAMPLES):
        idx = rng.integers(0, n, size=n)
        t = [true_escalate[i] for i in idx]
        o = [old_escalate[i] for i in idx]
        w = [new_escalate[i] for i in idx]
        try:
            diffs.append(cohens_kappa(t, w) - cohens_kappa(t, o))
        except ValueError:
            continue
    mean, lo, hi = _ci(diffs)
    click.echo(
        f"Paired difference (new - old policy kappa): mean {mean:.3f}  95% CI [{lo:.3f}, {hi:.3f}]"
    )
    click.echo(
        "(CI excludes 0 means the improvement is unlikely to be sampling noise on this n=200)"
        if lo > 0
        else "(CI includes 0)"
    )


if __name__ == "__main__":
    main()
