# Results

Golden set: 200 hand-labeled SpotifyCares conversations (intent, escalate, notes). Full methodology, taxonomy, and every design decision behind these numbers is in DECISIONS.md; this file is the results and what they actually mean.

Three systems evaluated against the same 200 examples:

- **Full agent**: Haiku classification, Sonnet reply drafting grounded in same-intent historical precedents, confidence/intent-based escalation policy.
- **Baseline 2**: zero-shot Haiku classify + draft, no retrieval, no grounding, same classifier and same escalation policy as the full agent.
- **Baseline 1**: keyword/regex classifier, fixed canned reply per intent, no LLM at all, never escalates.

| System | Intent raw agreement | Intent kappa | Escalate raw agreement | Escalate kappa | Mean judge score |
|---|---|---|---|---|---|
| Full agent | 0.530 | 0.412 | 0.590 | 0.099 | 4.14 |
| Baseline 2 (ungrounded Haiku) | 0.528 | 0.409 | 0.588 | 0.098 | 4.16 |
| Baseline 1 (keyword + canned) | 0.472 | 0.185 | 0.573 | 0.000 | 2.04 |

n=200 for the full agent, n=199 for both baselines (one judge call each failed on malformed JSON and was excluded, not silently scored; see Failure analysis).

## What's misleading about these headline numbers

**Escalate kappa is close to zero for every system, including the full agent, despite raw agreement in the high 50s.** Raw agreement looks passable. Chance-corrected agreement says none of these systems' escalation calls track human judgment meaningfully better than guessing off the label distribution. This is not a small-sample artifact: it holds at n=200, and it holds identically for the full agent and baseline 2, which share the same escalation policy. If this project reported only "59% escalation agreement" as its headline number, that would overstate the system's real reliability substantially. The reason is structural, not statistical noise, see Failure analysis.

**Self-reported classifier confidence is badly miscalibrated.** Mean self-reported confidence across all 200 predictions is 0.83. Actual intent accuracy is 0.53. Worse: mean confidence on the 94 *wrong* classifications is 0.81, barely below the 0.85 mean confidence on the 106 correct ones. The model is nearly as confident when it is wrong as when it is right. A number like "mean confidence 0.83" sounds like a calibrated reliability estimate; it is not one, and using it to gate escalation (see below) inherits that miscalibration directly.

**Baseline 2 scored marginally *higher* than the full agent on judge quality (4.16 vs 4.14).** Grounding in historical precedent plus a stronger drafting model (Sonnet vs Haiku) was expected to win clearly on reply quality. It did not, at this sample size. Two honest readings: grounding genuinely is not adding much reply-quality value the judge can detect, or the judge and the drafter are both Claude models and may share blind spots that an independent human rater would not share. There is no ground-truth reply-quality label in this golden set (deliberate, to avoid anchoring labelers on the historical reply) to check the judge against, so "mean judge score" is an internally-consistent LLM opinion, not a validated quality measure. Neither system's score should be read as "customers would rate this reply a 4.1/5."

## Results vs. baselines: what the comparison actually isolates

**Baseline 1 to baseline 2/full agent (using an LLM at all):** intent kappa goes from 0.185 to ~0.41, and mean judge score goes from 2.04 to ~4.15. Using an LLM instead of keyword matching and canned replies is a real, large improvement on both classification and reply quality. This is the clearest, least ambiguous result in the whole evaluation.

**Baseline 2 to full agent (grounding and a stronger drafting model specifically):** essentially no measurable difference. Intent numbers are nearly identical by construction, both share the same classifier and classifier output is cached, so this is not a meaningful second data point on classification. The judge-score comparison is the one part of this pairing that actually isolates grounding's effect, and it shows no win, see above. Escalation numbers are also nearly identical between these two, again because both use the same classifier feeding the same escalation policy; this comparison was never going to show a difference in escalation behavior and the near-identical rows confirm that rather than revealing anything new.

## Failure analysis

### The escalation policy has a structural blind spot, and it explains almost the entire miss rate

Full agent escalation confusion matrix (n=200):

| | Predicted: don't escalate | Predicted: escalate |
|---|---|---|
| **True: don't escalate** | 98 | 16 |
| **True: escalate** | 66 | 20 |

Recall on true escalations is 20/86, 23%. Of the 66 missed escalations, **all 66 (100%)** had classifier confidence at or above the 0.7 threshold and a predicted intent outside the two hardcoded high-risk categories (billing_dispute, cancellation_retention). The policy is confidence-threshold-or-high-risk-intent by design (DECISIONS.md); every single miss falls exactly into the gap that design leaves open. Breaking down what the policy missed:

- **31/66 were predicted account_access**, a category not in the high-risk set. Several of these look like genuinely urgent access/lockout issues by content (e.g. "it's been like a month since I first applied... what's happening?" on an account access problem, confidence 0.75) that a human labeler judged worth escalating and the policy structurally could not catch, since account_access carries no special weight regardless of confidence.
- **29/66 were predicted playback_bug**, where human escalation judgment appears driven by per-message severity or customer frustration, something a fixed two-category intent list cannot represent.
- **10/66 (9 billing_dispute + 1 cancellation_retention by true label)** should have been caught by the high-risk-intent rule but weren't, because the classifier had already misclassified the intent into something else. The escalation policy trusts the *predicted* intent, not the true one; a classification error here silently becomes an escalation error too, compounding rather than independent failures.

On the other side, 16 false-positive escalations (true: don't escalate, predicted: escalate) skew toward low-confidence borderline calls (mean confidence 0.70 for this group) and toward the two hardcoded high-risk intents (5 of 16), including a message that reads as a mild complaint with an Apple Music mention ("could you get involved with this please? Thinking of going to Apple Music otherwise", confidence 0.85, classified cancellation_retention) that a human did not think needed a human. The high-risk-intent rule fires on category membership alone with no read of actual severity, so it also over-triggers in the other direction.

**Net finding:** a fixed confidence-threshold-or-two-categories policy is the wrong shape for this problem. Human escalation judgment in this golden set is driven by message-level severity and urgency cues that cut across all six intent categories, not by intent category membership. Two concrete next steps this data supports: broadening the high-risk set to include account_access given it accounts for nearly half the misses, and decoupling escalation from the *classifier's* predicted intent so a misclassification cannot also silently suppress an escalation.

### Intent confusion concentrates in the OTHER bucket

Per-intent accuracy, full agent:

| True intent | n | Accuracy |
|---|---|---|
| playback_bug | 26 | 0.85 |
| account_access | 25 | 0.72 |
| feature_question | 47 | 0.57 |
| billing_dispute | 21 | 0.57 |
| other | 78 | 0.35 |
| cancellation_retention | 3 | 0.00 |

`other` is both the single largest true-intent class in this golden set (78/200, 39%) and the worst-performing one (35% accuracy), mostly misclassified as feature_question (29 times) or playback_bug (12 times), for example "@147836 When will Spotify come to India?" (a general question with no clear feature reference, true label `other`, predicted `feature_question`). The classifier appears biased toward assigning a specific-sounding category even when a human judged the message as not clearly fitting any of the five substantive intents. This is partly a taxonomy-design finding, not purely a model-quality one: `other` was defined as a catch-all escape hatch, and 39% of real traffic landing there suggests the five substantive categories may be under-covering the actual distribution of SpotifyCares conversations, or that `other` genuinely contains a heterogeneous mix the model has no consistent signal to detect.

`cancellation_retention` has only 3 examples in the golden set with 0% accuracy; too small a sample to draw any real conclusion from beyond "this category is rare in the sampled data and worth a larger targeted sample before trusting any number about it."

### Residual eval-harness failure mode: judge truncation, now rare but not eliminated

2 of 600 evaluations (1 in baseline 2, 1 in baseline 1) failed with the same malformed-JSON truncation that was caught and partially fixed earlier in this project (`_JUDGE_MAX_TOKENS` raised 300 to 500, prompt tightened to ask for a shorter rationale). An earlier 83-example run under the same fix saw 2 failures in 249 evaluations; this 200-example run saw 2 in 600, consistent with a low, roughly-stable residual rate rather than a fluke. Both failures were isolated cleanly by the harness (excluded from the reported n, not silently scored as something else) rather than crashing the run. The underlying risk (a sufficiently long model response can still truncate mid-JSON at any fixed token cap) is mitigated, not eliminated. Worth noting as a caveat on the n=199 rows rather than treating those denominators as if nothing was excluded.

## What I'd do differently with more time

- Recalibrate confidence (self-consistency sampling or logprobs) rather than trusting the model's self-report, given how flat confidence is between correct and incorrect predictions.
- Rebuild the escalation policy around message content directly rather than predicted-intent category membership, informed by the failure analysis above (account_access looks materially under-weighted; predicted-intent-only logic silently compounds classification errors into escalation errors).
- Get an independent human quality rating on a subsample of drafted replies, since the judge score currently has nothing external to validate against.
- Revisit the `other` bucket, either by subdividing the taxonomy or by specifically studying what `other` messages actually are, since it is both the largest and least accurately classified category.
