import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from domain.intents import Intent


class MissingAPIKeyError(Exception):
    pass


class Settings(BaseModel):
    anthropic_api_key: str | None = None
    haiku_model: str = "claude-haiku-4-5-20251001"
    sonnet_model: str = "claude-sonnet-5"
    subsample_path: Path = Path("data/subsample/spotifycares_subsample.csv")
    golden_path: Path = Path("data/golden/golden_set.csv")
    cache_dir: Path = Path("data/cache")
    brand_author_id: str = "SpotifyCares"
    seed: int = 42
    confidence_threshold: float = 0.7
    high_risk_intents: frozenset[Intent] = frozenset(
        {Intent.BILLING_DISPUTE, Intent.CANCELLATION_RETENTION, Intent.ACCOUNT_ACCESS}
    )
    precedent_k: int = 3
    max_workers: int = 8

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"))
