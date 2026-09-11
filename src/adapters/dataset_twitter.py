from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd

from domain.models import Conversation, Message


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


def load_conversations(csv_path: Path) -> list[Conversation]:
    df = pd.read_csv(csv_path, dtype={"response_tweet_id": "string"})

    parent: dict[int, int] = {int(tid): int(tid) for tid in df["tweet_id"]}
    messages_by_id: dict[int, Message] = {}

    for row in df.itertuples(index=False):
        tweet_id = int(cast(float, row.tweet_id))

        parent_id: int | None = None
        if pd.notna(row.in_response_to_tweet_id):
            parent_id = int(cast(float, row.in_response_to_tweet_id))
            _ensure(parent, parent_id)
            _union(parent, tweet_id, parent_id)
        if isinstance(row.response_tweet_id, str):
            for response_id in row.response_tweet_id.split(","):
                response_id_int = int(response_id)
                _ensure(parent, response_id_int)
                _union(parent, tweet_id, response_id_int)

        messages_by_id[tweet_id] = Message(
            id=str(tweet_id),
            author_id=str(row.author_id),
            text="" if pd.isna(row.text) else str(row.text),
            created_at=datetime.strptime(str(row.created_at), "%a %b %d %H:%M:%S %z %Y"),
            in_reply_to_id=str(parent_id) if parent_id is not None else None,
        )

    groups: dict[int, list[Message]] = {}
    for tweet_id, message in messages_by_id.items():
        root = _find(parent, tweet_id)
        groups.setdefault(root, []).append(message)

    conversations = [
        Conversation(
            conversation_id=str(root),
            messages=sorted(messages, key=lambda m: m.created_at),
        )
        for root, messages in groups.items()
    ]
    conversations.sort(key=lambda c: c.conversation_id)
    return conversations
