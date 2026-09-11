from dataclasses import dataclass

from domain.metrics import classification_accuracy, cohens_kappa, raw_agreement
from eval.judge import GoldenEvalResult


@dataclass(frozen=True)
class AgreementReport:
    n: int
    intent_raw_agreement: float
    intent_kappa: float
    escalate_raw_agreement: float
    escalate_kappa: float
    mean_judge_score: float


def compute_agreement(results: list[GoldenEvalResult]) -> AgreementReport:
    predicted_intents = [r.predicted_intent for r in results]
    true_intents = [r.true_intent for r in results]
    predicted_escalates = [r.predicted_escalate for r in results]
    true_escalates = [r.true_escalate for r in results]

    return AgreementReport(
        n=len(results),
        intent_raw_agreement=classification_accuracy(predicted_intents, true_intents),
        intent_kappa=cohens_kappa(predicted_intents, true_intents),
        escalate_raw_agreement=raw_agreement(predicted_escalates, true_escalates),
        escalate_kappa=cohens_kappa(predicted_escalates, true_escalates),
        mean_judge_score=sum(r.judge_score for r in results) / len(results),
    )
