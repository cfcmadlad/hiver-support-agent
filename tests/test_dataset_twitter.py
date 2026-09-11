from pathlib import Path

from adapters.dataset_twitter import load_conversations

SUBSAMPLE_PATH = Path("data/subsample/spotifycares_subsample.csv")


def test_load_conversations_count() -> None:
    conversations = load_conversations(SUBSAMPLE_PATH)
    assert len(conversations) == 1000


def test_every_conversation_has_a_spotifycares_reply() -> None:
    conversations = load_conversations(SUBSAMPLE_PATH)
    for conversation in conversations:
        authors = {message.author_id for message in conversation.messages}
        assert "SpotifyCares" in authors


def test_messages_sorted_by_created_at() -> None:
    conversations = load_conversations(SUBSAMPLE_PATH)
    for conversation in conversations:
        timestamps = [message.created_at for message in conversation.messages]
        assert timestamps == sorted(timestamps)


def test_conversation_ids_are_unique() -> None:
    conversations = load_conversations(SUBSAMPLE_PATH)
    ids = [c.conversation_id for c in conversations]
    assert len(ids) == len(set(ids))
