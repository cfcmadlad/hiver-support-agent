import pytest

from domain.intents import Intent
from domain.metrics import classification_accuracy, cohens_kappa, per_intent_counts, raw_agreement


def test_classification_accuracy() -> None:
    predictions = [Intent.BILLING_DISPUTE, Intent.PLAYBACK_BUG, Intent.BILLING_DISPUTE, Intent.OTHER]
    ground_truth = [Intent.BILLING_DISPUTE, Intent.PLAYBACK_BUG, Intent.PLAYBACK_BUG, Intent.OTHER]
    assert classification_accuracy(predictions, ground_truth) == pytest.approx(0.75)


def test_classification_accuracy_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        classification_accuracy([Intent.OTHER], [Intent.OTHER, Intent.OTHER])


def test_classification_accuracy_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        classification_accuracy([], [])


def test_per_intent_counts() -> None:
    predictions = [Intent.BILLING_DISPUTE, Intent.PLAYBACK_BUG, Intent.BILLING_DISPUTE]
    ground_truth = [Intent.BILLING_DISPUTE, Intent.BILLING_DISPUTE, Intent.BILLING_DISPUTE]
    counts = per_intent_counts(predictions, ground_truth)
    assert counts[Intent.BILLING_DISPUTE].true_positive == 2
    assert counts[Intent.BILLING_DISPUTE].false_negative == 1
    assert counts[Intent.PLAYBACK_BUG].false_positive == 1


def test_raw_agreement() -> None:
    rater_a = [True, True, False, False]
    rater_b = [True, False, False, False]
    assert raw_agreement(rater_a, rater_b) == pytest.approx(0.75)


def test_raw_agreement_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        raw_agreement([True], [True, False])


def test_raw_agreement_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        raw_agreement([], [])


def test_cohens_kappa_perfect_agreement() -> None:
    labels = [True, False, True, True, False]
    assert cohens_kappa(labels, labels) == pytest.approx(1.0)


def test_cohens_kappa_partial_agreement() -> None:
    rater_a = [True, True, True, False, False]
    rater_b = [True, True, False, False, False]
    assert cohens_kappa(rater_a, rater_b) == pytest.approx(0.6153846153846154)


def test_cohens_kappa_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        cohens_kappa([True], [True, False])


def test_cohens_kappa_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        cohens_kappa([], [])


def test_cohens_kappa_works_on_categorical_labels() -> None:
    rater_a = [Intent.BILLING_DISPUTE, Intent.PLAYBACK_BUG, Intent.OTHER]
    rater_b = [Intent.BILLING_DISPUTE, Intent.PLAYBACK_BUG, Intent.PLAYBACK_BUG]
    kappa = cohens_kappa(rater_a, rater_b)
    assert 0.0 < kappa < 1.0
