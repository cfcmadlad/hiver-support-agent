from collections import Counter
from collections.abc import Hashable
from dataclasses import dataclass

from domain.intents import Intent


@dataclass
class IntentCounts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0


def classification_accuracy(predictions: list[Intent], ground_truth: list[Intent]) -> float:
    return raw_agreement(predictions, ground_truth)


def per_intent_counts(
    predictions: list[Intent], ground_truth: list[Intent]
) -> dict[Intent, IntentCounts]:
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must be the same length")
    counts: dict[Intent, IntentCounts] = {intent: IntentCounts() for intent in Intent}
    for predicted, actual in zip(predictions, ground_truth):
        if predicted == actual:
            counts[predicted].true_positive += 1
        else:
            counts[predicted].false_positive += 1
            counts[actual].false_negative += 1
    return counts


def raw_agreement[T: Hashable](rater_a: list[T], rater_b: list[T]) -> float:
    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a and rater_b must be the same length")
    if not rater_a:
        raise ValueError("cannot compute agreement over an empty sequence")
    return sum(1 for a, b in zip(rater_a, rater_b) if a == b) / len(rater_a)


def cohens_kappa[T: Hashable](rater_a: list[T], rater_b: list[T]) -> float:
    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a and rater_b must be the same length")
    if not rater_a:
        raise ValueError("cannot compute kappa over an empty sequence")

    n = len(rater_a)
    observed_agreement = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n

    counts_a = Counter(rater_a)
    counts_b = Counter(rater_b)
    labels = set(counts_a) | set(counts_b)
    expected_agreement = sum((counts_a[label] / n) * (counts_b[label] / n) for label in labels)

    if expected_agreement >= 1.0:
        return 1.0
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)
