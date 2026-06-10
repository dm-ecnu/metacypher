"""Per-query timing and counting instrumentation.

The efficiency analysis in the paper reports per-query latency, LLM-call
counts, and COUNT-probe counts; this module makes those quantities measurable
instead of estimated. It is opt-in and zero-overhead when inactive: nothing is
recorded unless the caller is inside ``track_query()``.

Usage::

    from metacypher.instrumentation import track_query, instrumented_count_fn

    with track_query() as stats:
        result = text_to_cypher(question, graph)
    print(stats.as_dict())
    # {"total_seconds": ..., "stage_seconds": {"analysis": ..., ...},
    #  "llm_calls": ..., "llm_seconds": ..., "probe_count": ..., ...}

For ValidateRank, wrap the injected ``count_fn`` so every COUNT probe is
counted and timed::

    validate_rank(candidates, instrumented_count_fn(count_fn), catalog, ...)

The collector is thread-local, so concurrent queries on different threads do
not mix their numbers.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional

_local = threading.local()


@dataclass
class QueryStats:
    """Counters for one tracked query."""

    stage_seconds: Dict[str, float] = field(default_factory=dict)
    llm_calls: int = 0
    llm_seconds: float = 0.0
    probe_count: int = 0
    probe_seconds: float = 0.0
    prompt_count: int = 0
    prompt_chars: int = 0
    prompt_tokens_est: int = 0
    total_seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_seconds": self.total_seconds,
            "stage_seconds": dict(self.stage_seconds),
            "llm_calls": self.llm_calls,
            "llm_seconds": self.llm_seconds,
            "probe_count": self.probe_count,
            "probe_seconds": self.probe_seconds,
            "prompt_count": self.prompt_count,
            "prompt_chars": self.prompt_chars,
            "prompt_tokens_est": self.prompt_tokens_est,
        }


def current() -> Optional[QueryStats]:
    """The active collector for this thread, or None when not tracking."""
    return getattr(_local, "stats", None)


@contextmanager
def track_query() -> Iterator[QueryStats]:
    """Collect stats for everything executed inside the block."""
    stats = QueryStats()
    previous = current()
    _local.stats = stats
    start = time.perf_counter()
    try:
        yield stats
    finally:
        stats.total_seconds = time.perf_counter() - start
        _local.stats = previous


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Attribute wall-clock time inside the block to pipeline stage *name*."""
    stats = current()
    start = time.perf_counter()
    try:
        yield
    finally:
        if stats is not None:
            elapsed = time.perf_counter() - start
            stats.stage_seconds[name] = stats.stage_seconds.get(name, 0.0) + elapsed


def record_llm_call(seconds: float) -> None:
    """Called by the LLM client around each chat-completion request."""
    stats = current()
    if stats is not None:
        stats.llm_calls += 1
        stats.llm_seconds += seconds


def estimate_tokens(text: str) -> int:
    """Cheap model-free token estimate for prompt-size accounting.

    max(word count, chars/4) tracks BPE token counts within ~15% for the
    mixed English/Cypher prompts used here; good enough for the
    EX-vs-context-length figure, which compares relative sizes.
    """
    if not text:
        return 0
    return max(len(text.split()), len(text) // 4)


def record_prompt(chars: int, tokens_est: int) -> None:
    """Called where a generation prompt is assembled (fig:context x-axis)."""
    stats = current()
    if stats is not None:
        stats.prompt_count += 1
        stats.prompt_chars += int(chars)
        stats.prompt_tokens_est += int(tokens_est)


def instrumented_count_fn(count_fn: Callable[[str], int]) -> Callable[[str], int]:
    """Wrap a ValidateRank ``count_fn`` so each COUNT probe is counted/timed."""

    def wrapper(cypher: str) -> int:
        start = time.perf_counter()
        try:
            return count_fn(cypher)
        finally:
            stats = current()
            if stats is not None:
                stats.probe_count += 1
                stats.probe_seconds += time.perf_counter() - start

    return wrapper
