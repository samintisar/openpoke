"""Execution agent log management with structured XML-style tags.

Logs are keyed by agent id (the registry card id), so two agents whose names slug alike never share a log.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from html import escape, unescape
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from ...logging_config import logger
from ...utils.timezones import now_in_user_timezone


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_EXECUTION_LOG_DIR = _DATA_DIR / "execution_agents"


def _slugify(name: str) -> str:
    """Convert agent name to filesystem-safe slug."""
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in name.strip()).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "agent"


def _encode_payload(payload: str) -> str:
    """Encode payload for storage."""
    normalized = payload.replace("\r\n", "\n").replace("\r", "\n")
    collapsed = normalized.replace("\n", "\\n")
    return escape(collapsed, quote=False)


def _decode_payload(payload: str) -> str:
    """Decode payload from storage."""
    return unescape(payload).replace("\\n", "\n")


_ATTR_PATTERN = re.compile(r"(\w+)\s*=\s*\"([^\"]*)\"")

MEMORY_TAG = "agent_memory"


@dataclass
class AgentMemory:
    """A compacted memory: the summary of the first `upto` raw entries plus the identifiers seen in them."""

    memory: str
    identifiers: List[str] = field(default_factory=list)
    upto: int = 0
    timestamp: str = ""


class ExecutionAgentLogStore:
    """Append-only journal for execution agents with XML-style tags."""

    def __init__(self, base_dir: Path):
        self._base_dir = base_dir
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning(f"Failed to create directory: {exc}")

    def _lock_for(self, agent_name: str) -> threading.Lock:
        """Get or create a lock for an agent."""
        slug = _slugify(agent_name)
        with self._global_lock:
            if slug not in self._locks:
                self._locks[slug] = threading.Lock()
            return self._locks[slug]

    def _log_path(self, agent_name: str) -> Path:
        """Get log file path for an agent."""
        return self._base_dir / f"{_slugify(agent_name)}.log"

    def _append(self, agent_name: str, tag: str, payload: str, attrs: Optional[Dict[str, str]] = None) -> None:
        """Append an entry with the given tag (plus optional extra attributes)."""
        encoded = _encode_payload(str(payload))
        timestamp = now_in_user_timezone("%Y-%m-%d %H:%M:%S")
        extra = "".join(f" {key}=\"{escape(str(value))}\"" for key, value in (attrs or {}).items())
        entry = f"<{tag} timestamp=\"{timestamp}\"{extra}>{encoded}</{tag}>\n"

        with self._lock_for(agent_name):
            try:
                with self._log_path(agent_name).open("a", encoding="utf-8") as handle:
                    handle.write(entry)
            except Exception as exc:
                logger.error(f"Failed to append to log: {exc}")

    def _parse_line(self, line: str) -> Optional[Tuple[str, str, str]]:
        """Parse a single log line."""
        stripped = line.strip()
        if not (stripped.startswith("<") and "</" in stripped):
            return None

        open_end = stripped.find(">")
        close_start = stripped.rfind("</")
        close_end = stripped.rfind(">")

        if open_end == -1 or close_start == -1 or close_end == -1:
            return None

        open_tag_content = stripped[1:open_end]
        if " " in open_tag_content:
            tag, attr_string = open_tag_content.split(" ", 1)
        else:
            tag, attr_string = open_tag_content, ""

        closing_tag = stripped[close_start + 2 : close_end]
        if closing_tag != tag:
            return None

        attributes: Dict[str, str] = {
            match.group(1): match.group(2) for match in _ATTR_PATTERN.finditer(attr_string)
        }
        timestamp = attributes.get("timestamp", "")
        payload = _decode_payload(stripped[open_end + 1 : close_start])
        return tag, timestamp, payload

    def record_request(self, agent_name: str, instructions: str) -> None:
        """Record an incoming request from the interaction agent."""
        self._append(agent_name, "agent_request", instructions)

    def record_action(self, agent_name: str, description: str) -> None:
        """Record an agent action (tool call)."""
        self._append(agent_name, "agent_action", description)

    def record_tool_response(self, agent_name: str, tool_name: str, response: str) -> None:
        """Record the response from a tool."""
        self._append(agent_name, "tool_response", f"{tool_name}: {response}")

    def record_agent_response(self, agent_name: str, response: str) -> None:
        """Record the agent's final response."""
        self._append(agent_name, "agent_response", response)

    def record_memory(self, agent_name: str, memory: str, identifiers: List[str], upto: int) -> None:
        """Record a compacted memory covering the first `upto` raw entries; raw entries are never rewritten."""
        payload = json.dumps({"memory": memory, "identifiers": list(identifiers), "upto": int(upto)}, ensure_ascii=False)
        self._append(agent_name, MEMORY_TAG, payload, {"upto": str(int(upto))})

    def latest_memory(self, agent_name: str) -> Optional[AgentMemory]:
        """The most recent compacted memory, or None when there is none (or it is unreadable)."""
        latest: Optional[AgentMemory] = None
        for tag, timestamp, payload in self._iter_all(agent_name):
            if tag != MEMORY_TAG:
                continue
            try:
                # Stored newlines come back raw (see `_decode_payload`), so allow control characters.
                data = json.loads(payload, strict=False)
                latest = AgentMemory(
                    memory=str(data["memory"]),
                    identifiers=[str(item) for item in data.get("identifiers") or []],
                    upto=int(data.get("upto", 0)),
                    timestamp=timestamp,
                )
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                logger.warning(f"Skipping unreadable memory entry: {exc}")
        return latest

    def iter_entries(self, agent_name: str) -> Iterator[Tuple[str, str, str]]:
        """Iterate over the raw log entries for an agent (compacted memories are skipped)."""
        for entry in self._iter_all(agent_name):
            if entry[0] != MEMORY_TAG:
                yield entry

    def _iter_all(self, agent_name: str) -> Iterator[Tuple[str, str, str]]:
        """Iterate over every log entry, memories included."""
        path = self._log_path(agent_name)
        with self._lock_for(agent_name):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except FileNotFoundError:
                lines = []
            except Exception as exc:
                logger.error(f"Failed to read log: {exc}")
                lines = []

        for line in lines:
            parsed = self._parse_line(line)
            if parsed is not None:
                yield parsed

    def load_transcript(self, agent_name: str) -> str:
        """Load the full transcript for inclusion in system prompt."""
        parts: List[str] = []
        for tag, timestamp, payload in self.iter_entries(agent_name):
            escaped = escape(payload, quote=False)
            if timestamp:
                parts.append(f"<{tag} timestamp=\"{timestamp}\">{escaped}</{tag}>")
            else:
                parts.append(f"<{tag}>{escaped}</{tag}>")
        return "\n".join(parts)

    def load_recent(self, agent_name: str, limit: int = 10) -> list[tuple[str, str, str]]:
        """Load recent log entries."""
        entries = list(self.iter_entries(agent_name))
        return entries[-limit:] if entries else []

    def rename_agent(self, old_key: str, new_key: str) -> bool:
        """Move an agent's log to a new key; never overwrites an existing log."""
        source, target = self._log_path(old_key), self._log_path(new_key)
        if source == target or not source.exists() or target.exists():
            return False
        try:
            source.rename(target)
            return True
        except Exception as exc:
            logger.error(f"Failed to rename log: {exc}")
            return False

    def list_agents(self) -> list[str]:
        """List all agents with logs."""
        try:
            return sorted(path.stem for path in self._base_dir.glob("*.log"))
        except Exception as exc:
            logger.error(f"Failed to list agents: {exc}")
            return []

    def clear_all(self) -> None:
        """Clear all execution agent logs."""
        try:
            for log_file in self._base_dir.glob("*.log"):
                log_file.unlink()
            logger.info("Cleared all execution agent logs")
        except Exception as exc:
            logger.error(f"Failed to clear execution logs: {exc}")


_execution_agent_logs = ExecutionAgentLogStore(_EXECUTION_LOG_DIR)


def get_execution_agent_logs() -> ExecutionAgentLogStore:
    """Get the singleton log store instance."""
    return _execution_agent_logs
