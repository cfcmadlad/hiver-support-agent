import random
from pathlib import Path
from typing import cast

import pandas as pd

RAW_PATH = Path("data/raw/archive/twcs/twcs.csv")
OUT_PATH = Path("data/subsample/spotifycares_subsample.csv")
BRAND_AUTHOR_ID = "SpotifyCares"
SAMPLE_SIZE = 1000
SEED = 42


def _find(parent: dict[int, int], x: int) -> int:
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


def _union(parent: dict[int, int], a: int, b: int) -> None:
    root_a, root_b = _find(parent, a), _find(parent, b)
    if root_a != root_b:
        parent[root_b] = root_a


def _ensure(parent: dict[int, int], tweet_id: int) -> None:
    if tweet_id not in parent:
        parent[tweet_id] = tweet_id


def main() -> None:
    df = pd.read_csv(RAW_PATH, dtype={"response_tweet_id": "string"})

    parent: dict[int, int] = {int(tid): int(tid) for tid in df["tweet_id"]}
    for row in df.itertuples(index=False):
        tweet_id = int(cast(float, row.tweet_id))
        if pd.notna(row.in_response_to_tweet_id):
            parent_id = int(cast(float, row.in_response_to_tweet_id))
            _ensure(parent, parent_id)
            _union(parent, tweet_id, parent_id)
        if isinstance(row.response_tweet_id, str):
            for response_id in row.response_tweet_id.split(","):
                response_id_int = int(response_id)
                _ensure(parent, response_id_int)
                _union(parent, tweet_id, response_id_int)

    roots = {tweet_id: _find(parent, tweet_id) for tweet_id in parent}
    brand_tweet_ids = set(df.loc[df["author_id"] == BRAND_AUTHOR_ID, "tweet_id"].astype(int))
    brand_roots = {roots[tid] for tid in brand_tweet_ids}
    print(f"{len(brand_roots)} eligible conversations contain a {BRAND_AUTHOR_ID} reply")

    sampled_roots = set(
        random.Random(SEED).sample(sorted(brand_roots), min(SAMPLE_SIZE, len(brand_roots)))
    )
    print(f"sampled {len(sampled_roots)} conversations, seed {SEED}")

    df["_root"] = df["tweet_id"].astype(int).map(roots)
    subsample = df[df["_root"].isin(sampled_roots)].drop(columns="_root")
    subsample = subsample.sort_values("tweet_id")
    print(f"{len(subsample)} tweet rows across {len(sampled_roots)} conversations")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    subsample.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
