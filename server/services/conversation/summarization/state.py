from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class LogEntry:
    """Snapshot of a single conversation log entry."""

    tag: str
    payload: str
    index: int = -1
    timestamp: Optional[str] = None


@dataclass
class SummaryState:
    """Persisted working-memory summary state."""

    summary_text: str = ""
    last_index: int = -1
    updated_at: Optional[datetime] = None
    unsummarized_entries: List[LogEntry] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "SummaryState":
        return cls()


def omitted_user_entries(entries: Sequence[Tuple[str, str]], payloads: Iterable[str]) -> Set[int]:
    """Indexes of the last `user_message` entry matching each of `payloads` (one entry per payload).

    Used to leave messages the inbox already logged out of the history shown before them as the new message.
    """
    remaining = list(payloads)
    omitted: Set[int] = set()
    for index in range(len(entries) - 1, -1, -1):
        if not remaining:
            break
        tag, payload = entries[index]
        if tag == "user_message" and payload in remaining:
            remaining.remove(payload)
            omitted.add(index)
    return omitted


__all__ = ["LogEntry", "SummaryState", "omitted_user_entries"]
