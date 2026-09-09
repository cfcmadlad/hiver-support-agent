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
Escalation is deterministic given a `ClassificationResult` and `DraftReply` (confidence thresholds, intent-specific rules), not an LLM call. Per the project's own rule (no interface for a single implementation that will never have two), it lives as a plain function in `domain/escalation.py`, unlike `Classifier`, `ReplyDrafter`, and `Judge`, which are Protocols because their implementations are swappable adapters.

## Dataset subsample: 1,000 conversations, seed 42
Verified against the raw 2.81M-row `twcs.csv` before committing to the number: union-find over `in_response_to_tweet_id` / `response_tweet_id` finds 28,280 distinct connected conversation threads with at least one `SpotifyCares` reply, so 1,000 is a genuine random sample (3.5% of the eligible pool), not the whole population. Each "conversation" is the full connected thread, not just a single inbound/outbound pair, so multi-turn resolutions stay intact for reply-grounding retrieval later. The sampled 1,000 conversations expand to 3,436 total tweet rows. Size picked to leave headroom for a stratified 150-250 golden set while staying cheap to reprocess with Haiku on every iteration.

## Subsample extraction lives outside src/, not in adapters/dataset_twitter.py
The filtering and sampling script that produced `data/subsample/` is a one-time data-prep step, not the runtime dataset adapter. `adapters/dataset_twitter.py` stays an empty module this session; it gets implemented in session 2 against the already-fixed subsample, keeping today's scope to scaffolding plus data prep and none of the LLM-calling pipeline.
