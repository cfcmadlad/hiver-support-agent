# hiver-support-agent

An AI customer-support agent built over the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset, scoped to one brand: **SpotifyCares**.

The agent does three things:

1. **Classify** an incoming customer message into a self-defined intent set.
2. **Draft a reply**, grounded in how SpotifyCares historically resolved similar issues.
3. **Decide auto-handle vs escalate**, with a stated reason.

See [DECISIONS.md](DECISIONS.md) for what was chosen, rejected, and why. See [REPORT.md](REPORT.md) for results once the eval pipeline exists.

## Why SpotifyCares

A single digital product with genuine intent variety (billing disputes, playback bugs, account access, cancellation/retention) without the ground-truth noise of an ISP or telecom brand. Full reasoning in [DECISIONS.md](DECISIONS.md).

## Architecture

Layered, one-directional dependency flow:

```
interfaces (CLI) -> orchestration -> domain -> adapters -> data
```

Domain logic (`src/domain/`) never imports an SDK, a file path, or a model name. LLM calls, dataset reads, and disk writes live behind adapter interfaces (`src/adapters/`) defined by the domain and injected at the composition root.

```
src/
  interfaces/     CLI entry point
  orchestration/  pipeline wiring
  domain/         models, intents, escalation rules, metrics (pure, no side effects)
  adapters/       Claude API, dataset loading, response caching, storage
  config/         typed settings
eval/             hand-labeling tool, LLM-judge, human/judge agreement stats
tests/            metric computation, escalation rules, agreement stats
data/
  raw/            full Kaggle dataset (gitignored)
  subsample/      fixed-seed working subsample (gitignored)
  golden/         hand-labeled eval set (committed, this is a deliverable)
```

## Setup

Requires Python 3.12+.

```bash
python -m venv .venv
.venv/Scripts/activate   # .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
```

This installs `anthropic`, `pydantic`, `pandas`, `click` as runtime dependencies, `ruff`, `mypy`, `pytest` as the `dev` extra.

Set your Anthropic API key before running anything that calls Claude (session 2 onward):

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

`Settings.from_env()` loads `.env` automatically via `python-dotenv`. Note: `load_dotenv()` does not override an `ANTHROPIC_API_KEY` already set in your shell environment — if you previously exported one, unset it (`unset ANTHROPIC_API_KEY`) so `.env` takes effect.

## Dataset

The full `twcs.csv` from the Kaggle dataset is expected at `data/raw/archive/twcs/twcs.csv` (not committed, 2.81M rows across all brands).

`data/subsample/spotifycares_subsample.csv` is a fixed-seed working subsample scoped to SpotifyCares:

- Filtered to the 28,280 conversations in the raw dataset that contain at least one `SpotifyCares` reply (a "conversation" is the full connected thread via `in_response_to_tweet_id` / `response_tweet_id`, not just one inbound/outbound pair).
- Sampled down to **1,000 conversations, seed 42** (3,436 total tweet rows; see [DECISIONS.md](DECISIONS.md) for why this size).
- Gitignored, not committed. Anyone reproducing this needs the raw Kaggle CSV; the subsample is regenerated deterministically from it.

`data/golden/` holds the hand-labeled evaluation set (150-250 examples) and is committed since it's a project deliverable.

## Running the pipeline

Not yet implemented. The domain interfaces (`Classifier`, `ReplyDrafter`, `Judge` in [src/domain/models.py](src/domain/models.py)) are defined; the Claude-backed adapters and orchestration wiring land in the next session.

```bash
python -m interfaces.cli
```

Target: once the pipeline and eval harness exist, reproducing the headline results from a clean clone takes under 15 minutes.

## Development

```bash
ruff check .
mypy src eval tests
pytest
```
