"""Run execution agents: one run at a time per agent, a global cap on concurrent runs, partial results.

Runs for the same agent are serial in FIFO order (a correction sent mid-run queues behind the run in progress).
The global slot is taken inside the agent lock, so a queued run never holds one. Each finished run updates its card's
summary, schedules memory compaction and hands its result to `deliver_result`, which posts it to the inbox. The summary is updated in the background, so the interaction agent's turn on a
just-delivered result routes against the previous summary (deliberate: it never delays delivery).
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ...agents.execution_agent.runtime import ExecutionAgentRuntime, ExecutionResult
from ...agents.execution_agent.tools.gmail import find_key
from ...config import get_settings
from ...logging_config import logger
from ..agents.memory import maybe_compact
from ..agents.registry import get_agent_directory
from ..agents.summaries import refresh_card_summary
from ..inbox import get_inbox
from .log_store import ExecutionAgentLogStore, get_execution_agent_logs


@dataclass(frozen=True)
class RunContext:
    """The execution-agent run in progress: who is running and, for trigger fires, which trigger."""

    agent_id: str
    origin: str
    trigger_id: Optional[int] = None


# Set for the duration of an EA run so tools can tell which agent (and trigger) they act for.
current_run: contextvars.ContextVar[Optional[RunContext]] = contextvars.ContextVar("current_run", default=None)


def _display_name(agent_id: str) -> str:
    """The registry card's display name, falling back to the id."""
    try:
        card = get_agent_directory().get(agent_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"[{agent_id}] Registry lookup failed: {exc}")
        return agent_id
    return card.name if card else agent_id


_BACKGROUND_TASKS: set[asyncio.Task] = set()  # strong refs so background tasks are not collected


def _spawn(coro: Any) -> Optional[asyncio.Task]:
    task = asyncio.get_running_loop().create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return task


def _schedule_compaction(agent_id: str) -> None:
    """Compact the agent's history in the background; never blocks or fails the run."""
    try:
        _spawn(maybe_compact(agent_id))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"[{agent_id}] Could not schedule memory compaction: {exc}")


def _schedule_summary(agent_id: str, instructions: str, result_text: str) -> None:
    """Fold the run into the card's living summary in the background; never blocks or fails the run."""
    try:
        _spawn(refresh_card_summary(agent_id, instructions, result_text))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"[{agent_id}] Could not schedule the summary update: {exc}")


# -- result payload and delivery ------------------------------------------------------------------------------


def format_result(result: ExecutionResult) -> str:
    """Render one result in the interaction-agent format: `[SUCCESS|FAILED|PARTIAL] <name>: <text>`."""
    status = "PARTIAL" if result.partial else ("SUCCESS" if result.success else "FAILED")
    response_text = (result.response or "(no response provided)").strip()
    return f"[{status}] {result.agent_name}: {response_text}"


def deliver_result(result: ExecutionResult, *, origin: str, trigger_id: Optional[int] = None) -> None:
    """Post a finished run's result to the interaction inbox: `trigger_result` for trigger fires, else
    `agent_result`. The inbox coalesces results that finish close together into one interaction-agent turn."""
    kind = "trigger_result" if trigger_id is not None else "agent_result"
    try:
        get_inbox().post(kind, format_result(result), agent_id=result.agent_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(f"[{result.agent_name}] Failed to deliver result ({origin}, trigger={trigger_id}): {exc}")


# -- side effects from the run's log entries ------------------------------------------------------------------


_CALL_PATTERN = re.compile(r"^Calling (\w+) with: (.*)$", re.DOTALL)


def _field(args_text: str, key: str) -> Optional[str]:
    """A string or number argument from a (possibly truncated) JSON argument dump."""
    match = re.search(rf'"{key}"\s*:\s*(?:"([^"]*)"|(-?\d+))', args_text)
    if not match:
        return None
    return match.group(1) if match.group(1) is not None else match.group(2)


def _parse_response(text: str) -> Optional[Any]:
    """The tool's JSON result, `{}` when truncated past parsing, or None when the call failed.

    Successful calls are logged as a JSON dump; a tool that raised is logged as its bare error text.
    """
    stripped = text.strip()
    try:
        data = json.loads(stripped, strict=False)
    except ValueError:
        if not stripped.startswith(("{", "[", '"')):
            return None  # bare error text from a failed call
        if stripped.startswith('{"error"') or re.search(r'"successful"\s*:\s*false', stripped):
            return None
        return {}
    if isinstance(data, dict) and (data.get("error") or data.get("successful") is False):
        return None
    return data


def _sent_action(args: str) -> Optional[Any]:
    """The stored send action a call passed as `action_id`, if any (never raises)."""
    action_id = _field(args, "action_id")
    if not action_id:
        return None
    try:
        from ..approvals import get_action

        return get_action(int(action_id))
    except Exception:  # pragma: no cover - defensive: the description is best effort
        return None


def _describe(tool: str, args: str, response: str, data: Any) -> Optional[str]:
    """One side effect in plain words, or None for read-only tools."""
    to = _field(args, "recipient_email")
    if isinstance(data, dict) and data.get("status") == "approval_required":
        # The send gate held it: nothing went out, the user has to approve it first.
        return f"{tool.removeprefix('gmail_')} held for approval (action {data.get('action_id')})"
    if tool == "gmail_create_draft":
        draft_id = find_key(data, ("draft_id", "id")) or _field(response, "id") or "(id unknown)"
        return f"draft {draft_id} to {to}" if to else f"draft {draft_id}"
    ref = None
    action = _sent_action(args) if tool in {"gmail_execute_draft", "gmail_reply_to_thread", "gmail_forward_email"} else None
    if action is not None:  # sent by action_id: the stored action says where it went
        ref = action.gmail_ref
        to = to or ", ".join(action.recipients.get("to") or []) or None
    if tool == "gmail_execute_draft":
        return f"sent draft {_field(args, 'draft_id') or ref or '(id unknown)'}" + (f" to {to}" if action else "")
    if tool == "gmail_reply_to_thread":
        thread = _field(args, "thread_id") or ref or "(unknown)"
        return f"replied to {to or '(recipient unknown)'} on thread {thread}"
    if tool == "gmail_forward_email":
        message = _field(args, "message_id") or ref or "(unknown)"
        return f"forwarded message {message} to {to or '(recipient unknown)'}"
    if tool == "gmail_delete_draft":
        return f"deleted draft {_field(args, 'draft_id') or '(id unknown)'}"
    if tool == "createTrigger":
        return f"trigger {find_key(data, ('trigger_id', 'id')) or _field(response, 'trigger_id') or '(id unknown)'} set"
    if tool == "updateTrigger":
        trigger_id = _field(args, "trigger_id") or find_key(data, ("trigger_id",)) or "(id unknown)"
        return f"trigger {trigger_id} {'paused' if _field(args, 'status') == 'paused' else 'updated'}"
    return None


def side_effects_since(log_store: ExecutionAgentLogStore, agent_id: str, start: int) -> List[str]:
    """Side effects (drafts, sends, triggers) from the agent's log entries at index `start` onwards."""
    effects: List[str] = []
    pending_args: Dict[str, str] = {}
    for tag, _timestamp, payload in list(log_store.iter_entries(agent_id))[start:]:
        if tag == "agent_action":
            match = _CALL_PATTERN.match(payload)
            if match:
                pending_args[match.group(1)] = match.group(2)
        elif tag == "tool_response":
            tool, _, response = payload.partition(": ")
            args = pending_args.pop(tool, "")
            data = _parse_response(response)
            if data is None:
                continue
            effect = _describe(tool, args, response, data)
            if effect:
                effects.append(effect)
    return effects


def _with_side_effects(result: ExecutionResult, effects: List[str]) -> ExecutionResult:
    """Turn a partial result's response into `<reason>. Already done: <effects>.`"""
    reason = (result.response or "Stopped before finishing").strip().rstrip(".")
    done = "; ".join(effects) if effects else "nothing"
    result.side_effects = list(effects)
    result.response = f"{reason}. Already done: {done}."
    return result


# -- the runner -----------------------------------------------------------------------------------------------


class AgentRunner:
    """Serialize runs per agent, cap concurrent runs, and report partial results."""

    def __init__(
        self,
        max_concurrent: int = 4,
        timeout_seconds: float = 90,
        *,
        log_store: Optional[ExecutionAgentLogStore] = None,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds
        self._log_store = log_store
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._agent_locks: Dict[str, asyncio.Lock] = {}
        self._slots = asyncio.Semaphore(max_concurrent)

    def _logs(self) -> ExecutionAgentLogStore:
        return self._log_store or get_execution_agent_logs()

    def _bind_loop(self) -> None:
        """Locks belong to one event loop; start fresh if the shared runner is used from a new one."""
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._loop = loop
            self._agent_locks = {}
            self._slots = asyncio.Semaphore(self.max_concurrent)

    def _lock_for(self, agent_id: str) -> asyncio.Lock:
        lock = self._agent_locks.get(agent_id)
        if lock is None:
            lock = self._agent_locks[agent_id] = asyncio.Lock()
        return lock

    async def run(
        self,
        agent_id: str,
        instructions: str,
        *,
        origin: str,
        trigger_id: Optional[int] = None,
        agent_name: Optional[str] = None,
    ) -> ExecutionResult:
        """Run the agent, schedule its summary update and compaction and deliver the result.

        `origin` is "interaction" (sent by the interaction agent) or "trigger" (a trigger fire, with `trigger_id`).
        `agent_name` is the display name used in prompts and the result payload; it defaults to the card's name.
        """
        self._bind_loop()
        agent_name = agent_name or _display_name(agent_id)

        async with self._lock_for(agent_id):
            async with self._slots:
                result = await self._execute(agent_id, agent_name, instructions, origin, trigger_id)

        _schedule_summary(agent_id, instructions, format_result(result))
        _schedule_compaction(agent_id)
        deliver_result(result, origin=origin, trigger_id=trigger_id)
        return result

    async def _execute(
        self, agent_id: str, agent_name: str, instructions: str, origin: str, trigger_id: Optional[int]
    ) -> ExecutionResult:
        logs = self._logs()
        start = len(list(logs.iter_entries(agent_id)))  # side effects of this run are the entries after this
        token = current_run.set(RunContext(agent_id=agent_id, origin=origin, trigger_id=trigger_id))
        try:
            logger.info(f"[{agent_name}] Execution started")
            runtime = ExecutionAgentRuntime(agent_id, agent_name)
            result = await asyncio.wait_for(runtime.execute(instructions), timeout=self.timeout_seconds)
            status = "PARTIAL" if result.partial else ("SUCCESS" if result.success else "FAILED")
            logger.info(f"[{agent_name}] Execution finished: {status}")
        except asyncio.TimeoutError:
            logger.error(f"[{agent_name}] Execution timed out after {self.timeout_seconds}s")
            reason = f"Timed out after {self.timeout_seconds:g} seconds before finishing"
            logs.record_agent_response(agent_id, f"Partial: {reason}")
            result = ExecutionResult(
                agent_name=agent_name,
                agent_id=agent_id,
                success=False,
                response=reason,
                error="Timeout",
                partial=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(f"[{agent_name}] Execution failed unexpectedly")
            result = ExecutionResult(
                agent_name=agent_name,
                agent_id=agent_id,
                success=False,
                response=f"Execution failed: {exc}",
                error=str(exc),
            )
        finally:
            current_run.reset(token)

        if result.partial:
            result = _with_side_effects(result, side_effects_since(logs, agent_id, start))
        return result


_runner: Optional[AgentRunner] = None


def get_agent_runner() -> AgentRunner:
    """The shared runner, sized from settings."""
    global _runner
    if _runner is None:
        settings = get_settings()
        _runner = AgentRunner(settings.max_concurrent_agents, settings.agent_timeout_seconds)
    return _runner
