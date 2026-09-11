# Results

The golden set has 200 hand-labeled SpotifyCares conversations. Intent, escalate, and a note, on each one. The full methodology lives in DECISIONS.md. This file is about what the numbers actually mean, and where not to trust them.

## Problem framing

**What "good" means here.** Not classification accuracy on its own. Trustworthy autonomy. Handle as much routine volume as you can without a human, and still catch the messages where a wrong auto-response costs something real: money moved incorrectly, someone who wanted to cancel getting ignored, an access lockout left hanging. That's why escalation is its own graded decision here, not a side effect of classification confidence. It's why the harness reports escalate-kappa and intent-kappa separately instead of blending them into one number. A system that classifies well but escalates badly isn't mostly good. It's dangerous in exactly the cases that matter. The biggest finding in this report is an escalation policy that looked fine on paper and had a real blind spot underneath it.

**What I chose not to build.** Real decisions, not things I ran out of time for.
- No embedding or vector retrieval. Same-intent historical precedent is a cheap, good-enough similarity signal, and it doesn't pull in a new dependency.
- No calibrated confidence. The model just self-reports a number. It isn't logprob-based or sampled. The miscalibration is what the misleading-number section below is for, not something to quietly fix before anyone sees it.
- No open-ended taxonomy. Six fixed categories. That's what makes kappa well-defined, and it's what makes the OTHER-bucket finding read as a finding instead of noise.
- No multi-turn context when drafting. The drafter only sees the customer's first message.
- No tuning against the golden set, with one exception. The 0.7 confidence threshold was fixed before the golden set existed and stayed fixed. The high-risk intent list got changed once, after this golden set's own failure analysis pointed at it. That's a real break from the rule, so I checked it on a fresh, disjoint set instead of trusting the same 200 examples that produced the fix. More on that under Failure analysis.
- No production concerns. No rate limiting, no PII redaction, no monitoring, no handoff UI. Out of scope for a take-home.

Three systems, all tested against the same 200 examples.

- **Full agent.** Haiku classifies. Sonnet drafts a reply grounded in same-intent historical precedent. A confidence and intent-based policy decides whether to escalate.
- **Baseline 2.** Zero-shot Haiku. No retrieval. Same classifier and same policy as the full agent.
- **Baseline 1.** Keyword and regex matching. One canned reply per intent. No LLM anywhere. Never escalates.

| System | Intent raw agreement | Intent kappa | Escalate raw agreement | Escalate kappa | Mean judge score |
|---|---|---|---|---|---|
| Full agent | 0.530 | 0.412 | 0.720 | 0.417 | 4.16 |
| Baseline 2 (ungrounded Haiku) | 0.530 | 0.412 | 0.720 | 0.417 | 4.16 |
| Baseline 1 (keyword + canned) | 0.470 | 0.183 | 0.570 | 0.000 | 2.04 |

n=200 for all three. Nothing failed in this run, though there's a rare judge-truncation issue covered under Failure analysis. Bootstrap 95% confidence intervals, 2000 resamples: full agent intent kappa lands in [0.32, 0.49], escalate kappa in [0.29, 0.54].

One thing before the numbers mean anything: the escalate columns reflect a policy I changed partway through. I added `account_access` to the high-risk set after the failure analysis below showed it was behind nearly half of missed escalations. The before and after is there too.

## What is misleading about my headline number?

The old escalation kappa wasn't just low. It was statistically indistinguishable from chance. Before the fix, escalate kappa was 0.099, with a bootstrap interval of [-0.02, 0.23]. That range includes zero. Raw agreement, 0.590, looked passable. Kappa said you couldn't actually rule out no-better-than-guessing. After the fix, kappa climbed to 0.417, interval [0.29, 0.54]. A paired bootstrap run directly on the improvement gives [0.21, 0.42], which excludes zero too. That's the real evidence the fix helped, not a lucky reshuffle of 200 examples. Raw agreement alone, 0.590 to 0.720, would never have shown you that. And 0.417 is still moderate, not strong. The rest of the gap has its own explanation further down.

Self-reported confidence is badly miscalibrated. Mean confidence across all 200 predictions is 0.83. Actual accuracy is 0.53. Mean confidence on the 94 wrong classifications is 0.81, barely below the 0.85 on the 106 correct ones. The model is almost as confident when it's wrong as when it's right. That number gates escalation, so the miscalibration goes straight through with it.

Baseline 2 and the full agent tied on judge quality. 4.16 each. Grounding plus a stronger drafting model was supposed to win clearly here. It didn't win at all. Either grounding isn't adding much reply quality the judge can detect, or the judge and the drafter are similar enough as models that they share blind spots a person wouldn't. Don't read 4.16 as customers would rate this a 4.2 out of 5. Check that against an actual person first, which is what the next part does.

The judge's mean matching a human's mean is not the same thing as the judge being a good judge of any single reply. I had someone rate 30 full-agent replies blind, no visibility into the judge's own score. Their average was 4.17. The judge's average on the same 30 was 4.16. Almost a perfect match. But agreement on individual replies is only moderate. Correlation 0.41. Treat a score of 4 or above as "good" and the binary kappa is 0.36, fair, well short of strong. If this report had stopped at "the means match, the judge is validated," that would be the exact mistake this section is warning about. Two averages can agree while the calls underneath them mostly don't. Three disagreements make it concrete. The judge gave a 2 out of 5 to a reply with a genuinely garbled sentence, and the human gave it a 5, reading straight past the broken grammar. The judge gave a 5 to a reply that claimed "we've sent you a DM," an action the pipeline never actually verified, and the human gave that one a 2. A third pair, judge 4, human 2, looks like the judge rewarding a procedurally correct answer even though it reads as tone-deaf next to a frustrated customer. This is the strongest reason in the whole report not to read "mean judge score" as "what a human would say," not without a check like this one.

## Results vs. baselines

Baseline 1 against the other two answers one question: does using an LLM at all matter. Intent kappa goes from 0.183 to 0.412. Mean judge score goes from 2.04 to 4.16. A real, large win on both classification and reply quality. This is the clearest result in the whole evaluation.

Baseline 2 against the full agent answers a different one: does grounding and a stronger drafting model matter. No measurable difference. Intent and escalate numbers are exactly identical, not just close, because both share a classifier and that classifier's output is cached. So this pairing was never going to say much about classification either way. The judge score is the one part that actually isolates grounding, and it's a dead tie. 4.16 to 4.16.

## Failure analysis

### 1. The escalation policy had a structural blind spot. Adding account_access to the high-risk set fixed most of it.

Before, the policy escalated on confidence below 0.7, or when the intent was billing_dispute or cancellation_retention.

| | Predicted: don't escalate | Predicted: escalate |
|---|---|---|
| **True: don't escalate** | 98 | 16 |
| **True: escalate** | 66 | 20 |

Recall on true escalations: 20 out of 86, 23%. Every one of the 66 misses had confidence at or above 0.7, and a predicted intent outside the high-risk set. 31 were predicted account_access. 29 were playback_bug. 3 feature_question, 3 other. account_access alone was nearly half the miss rate, the biggest single contributor, and a lot of these read as genuinely urgent.

So I added account_access to the high-risk set, in `src/config/settings.py`, and re-derived escalation from predictions that were already classified and already cached. No new API calls. Escalation is just a function sitting on top of classification.

| | Predicted: don't escalate | Predicted: escalate |
|---|---|---|
| **True: don't escalate** | 93 | 21 |
| **True: escalate** | 35 | 51 |

Recall: 51 out of 86, 59%. Precision improved too, 71% against 56%, so it wasn't a trade, both numbers moved together. Kappa went from 0.099 to 0.417. Statistically real, not just a shifted point estimate, see the bootstrap intervals above.

What's left isn't a smaller version of the same problem. It's a different one. 29 of the remaining 35 misses, 83%, are predicted playback_bug, which was never going to be caught since it isn't a high-risk category. The other 8 were all misclassified away from their true high-risk intent before the policy ever saw them, so the classifier's error was the actual failure, not the policy. playback_bug was left off the high-risk list on purpose, since most playback bugs really are low-stakes. The ones that aren't need a signal for severity, not another category. False positives moved too, 16 up to 21, and 10 of those are now driven by a high-risk intent firing regardless of how severe the message actually is. That's the visible cost of a rule based on category alone.

I labeled 40 more conversations, disjoint from the golden set, specifically so this check couldn't be contaminated by the same data that produced the fix. Escalate kappa on that set: 0.557, recall 88%, precision 68%. Intent kappa: 0.402. Both at or above the numbers that produced and then validated the fix in the first place. At 40 examples these carry real uncertainty on their own. But the direction is unambiguous. This wasn't the model overfitting to the golden set.

### 2. billing_dispute and account_access were tangled up at the classification level.

billing_dispute is 57% accurate. Six of the nine errors get misclassified as account_access, the biggest confusion pair in the whole set. Two real examples show why it's genuinely ambiguous, not just a model slip. "I am being charged monthly but unable to access my premium." "Same credit card, not expired, logged out and in, still not working." Both mix payment language with access language. My guess is that the taxonomy treats these two intents as separated by root cause, but a customer just describes what they're feeling, and what they're feeling is often access-shaped even when the real cause is billing. All six of these misclassified messages are also true-escalate cases. So before the fix, every one of them silently missed its escalation too. Now that account_access is high-risk, getting misclassified into it still triggers escalation. The mix-up costs a classification point, not an escalation anymore. That's a second win from the same fix, not something new.

### 3. Intent confusion piles up in the OTHER bucket.

| True intent | n | Accuracy |
|---|---|---|
| playback_bug | 26 | 0.85 |
| account_access | 25 | 0.72 |
| feature_question | 47 | 0.57 |
| billing_dispute | 21 | 0.57 |
| other | 78 | 0.35 |
| cancellation_retention | 3 | 0.00 |

other is the biggest class in this golden set, 78 of 200, and the worst-performing one, 35% accurate. Mostly mistaken for feature_question, 29 times, or playback_bug, 12 times. One message just asked when Spotify was coming to India. True label other. The model called it feature_question. It looks like the classifier reaches for something specific-sounding even when a human clearly judged the message didn't fit anywhere. Part of this is a taxonomy problem, not a model problem. If 39% of real traffic lands in a catch-all, the five substantive categories are probably under-covering what SpotifyCares actually sees. cancellation_retention only has 3 examples here, all wrong, too small a sample to conclude much beyond needing a bigger, targeted one.

### 4. Reply quality is worst on OTHER too, not just classification accuracy.

16 of 200 full-agent replies score 3 or below from the judge. Four of those, the actual bottom, are simple, the message was misclassified, so the reply answered the wrong problem. But eight of the remaining twelve were classified correctly as other and still came out mediocre. other averages 3.90 across 35 replies. Every other intent sits between 4.03 and 4.31. That's a pattern, not bad luck. My guess is that other's own historical precedent pool is just as mixed as the category itself, so even a correctly-routed message gets grounded on precedent that might be about something else entirely. That's a second, independent piece of evidence that other is under-specified, and this time it hits drafting, not classification.

### 5. A residual harness failure: judge truncation, better but not gone.

Zero failures in this run. All 600 evaluations scored cleanly. That doesn't mean the failure mode is fixed. The same judge truncation, a response cut off mid-JSON when the rationale runs long, has shown up twice before under this exact same fix. Two failures in 249 evaluations at 83 examples. Two in 600 at an earlier run of 200. A low, steady rate across three runs, then zero on the fourth, reads as genuinely rare, not resolved. When it does happen, the harness excludes it cleanly, records the error, never miscounts it as something it isn't. That's the actual fix. The underlying cause, a long enough response can still get cut off at any fixed limit, doesn't fully go away without retrying on a parse failure.

## Is this agent good enough to trust?

Short answer. Not for full autonomy, not today. Yes, for a specific, limited slice of the traffic.

What I'd trust it to do: the high-risk categories always reach a human, by policy, and that holds up on two separate checks, kappa 0.417 on the golden set, 0.557 on held-out data, precision in the high 60s to low 70s both times. The pipeline also clearly beats both baselines. The intent kappa intervals don't even overlap with the keyword baseline's.

What I wouldn't trust yet: auto-handling anything predicted other. That's 39% of traffic, the worst accuracy, and now the worst reply quality too. It should go to a human by default. I also wouldn't trust the escalation policy to catch a genuinely severe playback_bug message. 83% of the remaining misses live there, and the policy has no idea how to tell a severe one from a routine one, it only knows category membership. And I wouldn't report mean judge score or mean confidence to anyone as a reliable number. Both were checked against real ground truth. Both came up short.

What I'd actually ship, if this were a real decision: auto-handle only playback_bug and feature_question predictions that clear the escalation policy. Route every other prediction to a person, no exceptions. Add some real severity signal to playback_bug before trusting its escalation calls at all. Run the whole thing in shadow mode first, the agent recommends, a person decides, before it ever runs on its own. The 40-example held-out check in this report is a genuinely good sign. It's still one check. A live rollout deserves ongoing validation, not a single number to clear once.

## What I'd do next with one more week

- Label a much bigger held-out set. The 40-example check says the fix generalizes, and the direction is clear, but those individual numbers carry real noise of their own at that size.
- Decouple escalation from the classifier's predicted intent, so a misclassification can't quietly swallow an escalation no matter which categories count as high-risk.
- Model severity directly instead of growing the high-risk list. The remaining misses are 83% playback_bug, and that category will never make sense as blanket high-risk, since most playback bugs really are low-stakes.
- Recalibrate confidence, self-consistency sampling or logprobs, instead of trusting the model's own self-report, given how flat it is between right and wrong.
- Rate more than 30 replies for the judge-human check, and get a second rater too, to see whether the agreement and the three disagreement patterns hold up at a bigger sample.
- Take a real look at the other bucket. It's the biggest category, the least accurate, and now also the worst on reply quality.
