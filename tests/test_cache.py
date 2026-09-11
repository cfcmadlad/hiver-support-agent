import threading
from pathlib import Path

from adapters.cache import ResponseCache


def test_get_or_set_returns_cached_value_without_recomputing(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    calls = []

    def compute() -> str:
        calls.append(1)
        return "computed"

    first = cache.get_or_set("key", compute)
    second = cache.get_or_set("key", compute)

    assert first == "computed"
    assert second == "computed"
    assert len(calls) == 1


def test_get_or_set_serializes_concurrent_calls_on_the_same_key(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    call_count = 0
    call_count_lock = threading.Lock()

    def compute() -> str:
        nonlocal call_count
        with call_count_lock:
            call_count += 1
        return "computed"

    threads = [
        threading.Thread(target=cache.get_or_set, args=("shared-key", compute)) for _ in range(20)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert call_count == 1
    assert cache.get("shared-key") == "computed"


def test_get_or_set_different_keys_do_not_block_each_other(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    assert cache.get_or_set("a", lambda: "value-a") == "value-a"
    assert cache.get_or_set("b", lambda: "value-b") == "value-b"
