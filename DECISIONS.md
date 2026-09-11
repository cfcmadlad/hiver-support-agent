# Decisions

## Brand: SpotifyCares
Chosen over comcastcares/TMobileHelp/UPSHelp/AmazonHelp/AppleSupport. Single digital product with genuine intent variety (billing disputes, playback bugs, account access, cancellation/retention conversations) without ISP-level ground truth noise. Chosen for failure analysis and escalation reasoning depth given grading weights eval rigor over agent polish, not for domain size.

## LLM: Claude Haiku for classification and baselines, Sonnet for drafting and judging
One provider, one adapter interface. Haiku cheap enough for full subsample passes during iteration; Sonnet reserved for reply drafting and the LLM-judge where reasoning quality matters more than cost.

## Dependency manager: venv + pip (reversed from an initial uv choice)
Started with uv for a single-tool venv/lockfile/install workflow. Reversed once uv, then plain pip, both hit the same broken TLS chain on this machine (`SSL: CERTIFICATE_VERIFY_FAILED`, a local network/cert interception issue, not a PyPI problem) and needed non-default flags (`--native-tls`, then a hand-exported Windows CA bundle) to install anything. `python -m venv` + `pip install -e ".[dev]"` is the more universally-installed baseline: no extra tool for a grader to install, and `pyproject.toml` stays the single source of truth for dependencies either way (`[project.dependencies]` plus a `dev` extra instead of uv's `[dependency-groups]`). Traded uv's lockfile-pinned reproducibility for lower setup friction on an unknown grading machine.

## Build backend: hatchling over uv_build
uv_build expects a single package under `src/<project_name>/`. This project's locked structure puts multiple top-level packages directly under `src/` (domain, adapters, orchestration, interfaces, config) plus a sibling `eval/` package at the repo root. hatchling's `packages = [...]` list supports that directly with an editable install, so `from domain.models import X` works unqualified from anywhere in the repo without path hacks.

## Intent taxonomy: closed 6-value enum, not open classification
BILLING_DISPUTE, PLAYBACK_BUG, ACCOUNT_ACCESS, CANCELLATION_RETENTION, FEATURE_QUESTION, OTHER. `str, Enum` so it serializes cleanly through Pydantic without a custom encoder. A closed set keeps the classifier's output space fixed and makes agreement stats (judge vs human) well-defined; OTHER is the escape hatch for anything that doesn't fit rather than letting the model invent new categories.

## Escalation rule is a pure function, not an injected Protocol
Escalation is deterministic given a `ClassificationResult` (confidence threshold, intent-specific rules), not an LLM call. Per the project's own rule (no interface for a single implementation that will never have two), it lives as a plain function in `domain/escalation.py`, unlike `Classifier`, `ReplyDrafter`, and `Judge`, which are Protocols because their implementations are swappable adapters.

## Escalation policy: confidence threshold OR high-risk intent, not confidence alone
Escalates if classifier confidence < 0.7, or if intent is `BILLING_DISPUTE` or `CANCELLATION_RETENTION` regardless of confidence. Confidence-only would auto-handle a confidently-wrong billing dispute; money and churn issues get a human even when the classifier is sure, since a wrong auto-reply there is more costly than on a playback bug. The threshold and high-risk set are `EscalationPolicy` fields (`domain/escalation.py`), constructed from `Settings` at the orchestration layer, not passed into domain as the whole settings object — domain only sees the two primitives it actually needs, keeping it decoupled from adapter/config-level concerns like model names and file paths.

## Golden set schema: intent + escalate + notes, not a written reference reply
Each hand-labeled example captures true intent, whether it should have been escalated, and free-text notes. The actual historical SpotifyCares reply is already in the subsample data for free, so it's not something the labeler writes. Labeling shows only the customer's message, not the historical reply, to avoid anchoring the escalate judgment on what the brand actually did.

## Reply grounding: precedent retrieval by classified intent
The whole subsample is classified once (Haiku, cached — one-time cost, free on every rerun after), then grouped into a `PrecedentIndex` by intent. At draft time, a few same-intent historical (customer message, actual SpotifyCares reply) pairs are pulled as few-shot context for the Sonnet draft call. No embedding/vector-search dependency needed, stays inside the locked dependency list, and directly matches "grounded in how the brand historically resolved similar issues" rather than approximating it via keyword overlap.

## Baselines: rule-based keyword classifier vs. zero-shot ungrounded Haiku
Baseline 1 is a keyword/regex classifier (no LLM) with a fixed canned per-intent reply, always auto-handled. Baseline 2 is zero-shot Haiku classify+draft with no retrieval or historical grounding. Comparing the full agent against both isolates two different sources of uplift: using an LLM at all (baseline 1 to baseline 2), and grounding plus Sonnet specifically (baseline 2 to the full agent). Not built yet — deferred to the next session, after the golden set and pipeline are verified.

## Structured LLM output via `client.messages.parse(output_format=...)`, not manual tool-use JSON
The installed Anthropic SDK (1.4.0) has a native `messages.parse` method that takes a Pydantic model as `output_format`, derives a JSON schema from it, and returns a `parsed_output` already validated against that model. Used directly for `ClassificationResult` and `JudgeVerdict`. For `DraftReply`, the LLM only produces `text` (a small adapter-internal schema); `grounded_on` is filled in from the precedent ids actually retrieved, not self-reported by the model, since we know that deterministically and don't want to trust the model's own account of what it used. A malformed or missing `parsed_output` raises `LLMResponseError` explicitly rather than falling back to a guessed answer.

## Classifier confidence is self-reported, not calibrated
`ClassificationResult.confidence` comes from asking the model to report its own confidence 0-1; it isn't a statistically calibrated probability from logprobs or repeated sampling. Flagging this now because it's exactly the kind of thing that becomes a "what's misleading about my headline number" caveat later — a mean self-reported confidence isn't the same claim as a mean calibrated one. Constrained to `[0, 1]` via a Pydantic `Field`, both so the JSON schema handed to the model is precise and so the escalation confidence-threshold check can't silently misfire on an out-of-range value.

## Precedent pool is sampled disjoint from whatever's being classified/drafted
An 8-angle review of the day-2 diff caught a real leakage bug: the CLI's `run` command originally built the precedent index from the same small `--count` sample it then drafted replies for, so a conversation could retrieve its own actual historical resolution as its own "precedent" — the model was shown the answer to the question it was being asked. Fixed by shuffling the full subsample once per run (seeded) and slicing two disjoint ranges: `conversations` (what gets processed) and `precedent_pool` (what the precedent index is built from), with a `--precedent-pool-size` option (default 20) so a cheap smoke test doesn't require classifying the whole 1,000-conversation subsample to stay leakage-free. The same review also found "find the customer's root message and the brand's resolution" reimplemented three times (pipeline, CLI, labeling tool) with inconsistent rules — consolidated into `Conversation.root_customer_message()` / `Conversation.last_brand_message()` on the domain model so there's one definition.

## Dataset subsample: 1,000 conversations, seed 42
Verified against the raw 2.81M-row `twcs.csv` before committing to the number: union-find over `in_response_to_tweet_id` / `response_tweet_id` finds 28,280 distinct connected conversation threads with at least one `SpotifyCares` reply, so 1,000 is a genuine random sample (3.5% of the eligible pool), not the whole population. Each "conversation" is the full connected thread, not just a single inbound/outbound pair, so multi-turn resolutions stay intact for reply-grounding retrieval later. The sampled 1,000 conversations expand to 3,436 total tweet rows. Size picked to leave headroom for a stratified 150-250 golden set while staying cheap to reprocess with Haiku on every iteration.

## Subsample extraction lives outside src/, not in adapters/dataset_twitter.py
The filtering and sampling script that produced `data/subsample/` is a one-time data-prep step, not the runtime dataset adapter. `adapters/dataset_twitter.py` stays an empty module this session; it gets implemented in session 2 against the already-fixed subsample, keeping today's scope to scaffolding plus data prep and none of the LLM-calling pipeline.
