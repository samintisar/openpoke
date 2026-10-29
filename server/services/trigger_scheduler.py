"""Background scheduler that watches trigger definitions and executes them."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Set

from ..logging_config import logger
from . import approvals
from .agents.lifecycle import sweep
from .agents.registry import get_agent_directory
from .execution.runner import get_agent_runner
from .triggers import TriggerRecord, get_trigger_service


UTC = timezone.utc
LIFECYCLE_SWEEP_INTERVAL = timedelta(hours=1)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _agents_with_pending_sends(now: datetime) -> Set[str]:
    """Expire approvals past their time, then the agents still waiting on one; empty if the store fails."""
    try:
        approvals.expire_stale(now)
        return approvals.agent_ids_with_pending(now)
    except Exception as exc:
        logger.exception("Pending send lookup failed", extra={"error": str(exc)})
        return set()


def _isoformat(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class TriggerScheduler:
    """Polls stored triggers and launches execution agents when due."""

    def __init__(self, poll_interval_seconds: float = 10.0) -> None:
        self._poll_interval = poll_interval_seconds
        self._service = get_trigger_service()
        self._task: Optional[asyncio.Task[None]] = None
        self._running = False
        self._in_flight: Set[int] = set()
        self._lock = asyncio.Lock()
        self._last_sweep: Optional[datetime] = None

    async def start(self) -> None:
        async with self._lock:
            if self._task and not self._task.done():
                return
            loop = asyncio.get_running_loop()
            self._running = True
            self._task = loop.create_task(self._run(), name="trigger-scheduler")
            logger.info("Trigger scheduler started", extra={"interval": self._poll_interval})

    async def stop(self) -> None:
        async with self._lock:
            self._running = False
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                self._task = None
                logger.info("Trigger scheduler stopped")

    async def _run(self) -> None:
        try:
            while self._running:
                await self._poll_once()
                self._maybe_sweep(_utc_now())
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Trigger scheduler loop crashed", extra={"error": str(exc)})

    def run_lifecycle_sweep(self, now: Optional[datetime] = None) -> Dict[str, str]:
        """Expire stale send approvals and demote stale agents (open work = a trigger due within 24h or a
        pending send under 7 days old); logs failures, never raises."""
        now = now or _utc_now()
        self._last_sweep = now
        try:
            open_work = self._service.agent_ids_with_triggers_due(now) | _agents_with_pending_sends(now)
            changed = sweep(get_agent_directory(), now, open_work)
        except Exception as exc:
            logger.exception("Agent lifecycle sweep failed", extra={"error": str(exc)})
            return {}
        if changed:
            logger.info("Agent lifecycle sweep", extra={"changed": changed})
        return changed

    def _maybe_sweep(self, now: datetime) -> None:
        if self._last_sweep is None or now - self._last_sweep >= LIFECYCLE_SWEEP_INTERVAL:
            self.run_lifecycle_sweep(now)

    def _revive(self, trigger: TriggerRecord, fired_at: datetime) -> None:
        """A trigger fire is activity: the agent's card goes active and its last_active_at moves to now."""
        try:
            get_agent_directory().touch(trigger.agent_id, trigger.payload, fired_at)
        except KeyError:
            logger.warning(
                "Trigger agent has no registry card",
                extra={"trigger_id": trigger.id, "agent_id": trigger.agent_id},
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to revive trigger agent",
                extra={"trigger_id": trigger.id, "agent_id": trigger.agent_id, "error": str(exc)},
            )

    async def _poll_once(self) -> None:
        now = _utc_now()
        due_triggers = self._service.get_due_triggers(before=now)
        if not due_triggers:
            return

        for trigger in due_triggers:
            if trigger.id in self._in_flight:
                continue
            if not trigger.agent_id:
                logger.warning(
                    "Skipping trigger without agent_id",
                    extra={"trigger_id": trigger.id, "agent": trigger.agent_name},
                )
                continue
            self._in_flight.add(trigger.id)
            asyncio.create_task(self._execute_trigger(trigger), name=f"trigger-{trigger.id}")

    async def _execute_trigger(self, trigger: TriggerRecord) -> None:
        try:
            fired_at = _utc_now()
            instructions = self._format_instructions(trigger, fired_at)
            logger.info(
                "Dispatching trigger",
                extra={
                    "trigger_id": trigger.id,
                    "agent": trigger.agent_name,
                    "scheduled_for": trigger.next_trigger,
                },
            )
            self._revive(trigger, fired_at)
            result = await get_agent_runner().run(
                trigger.agent_id,
                instructions,
                origin="trigger",
                trigger_id=trigger.id,
                agent_name=trigger.agent_name,
            )
            if result.success:
                self._handle_success(trigger, fired_at)
            else:
                error_text = result.error or result.response
                self._handle_failure(trigger, fired_at, error_text)
        except Exception as exc:  # pragma: no cover - defensive
            self._handle_failure(trigger, _utc_now(), str(exc))
            logger.exception(
                "Trigger execution failed unexpectedly",
                extra={"trigger_id": trigger.id, "agent": trigger.agent_name},
            )
        finally:
            self._in_flight.discard(trigger.id)

    def _handle_success(self, trigger: TriggerRecord, fired_at: datetime) -> None:
        logger.info(
            "Trigger completed",
            extra={"trigger_id": trigger.id, "agent": trigger.agent_name},
        )
        self._service.schedule_next_occurrence(trigger, fired_at=fired_at)

    def _handle_failure(self, trigger: TriggerRecord, fired_at: datetime, error: str) -> None:
        logger.warning(
            "Trigger execution failed",
            extra={
                "trigger_id": trigger.id,
                "agent": trigger.agent_name,
                "error": error,
            },
        )
        self._service.record_failure(trigger, error)
        if trigger.recurrence_rule:
            self._service.schedule_next_occurrence(trigger, fired_at=fired_at)
        else:
            self._service.clear_next_fire(trigger.id, agent_id=trigger.agent_id)

    def _format_instructions(self, trigger: TriggerRecord, fired_at: datetime) -> str:
        scheduled_for = trigger.next_trigger or _isoformat(fired_at)
        metadata_lines = [f"Trigger ID: {trigger.id}"]
        if trigger.recurrence_rule:
            metadata_lines.append(f"Recurrence: {trigger.recurrence_rule}")
        if trigger.timezone:
            metadata_lines.append(f"Timezone: {trigger.timezone}")
        if trigger.start_time:
            metadata_lines.append(f"Start Time (UTC): {trigger.start_time}")

        metadata = "\n".join(f"- {line}" for line in metadata_lines)
        return (
            f"Trigger fired at {_isoformat(fired_at)} (UTC).\n"
            f"Scheduled occurrence time: {scheduled_for}.\n\n"
            f"Metadata:\n{metadata}\n\n"
            f"Payload:\n{trigger.payload}"
            f"{self._standing_note(trigger)}"
        )

    def _standing_note(self, trigger: TriggerRecord) -> str:
        """Tell the agent which recipients the user pre-approved for this trigger, so it sends instead of asking."""
        try:
            standing = approvals.get_approval_store().active_standing(trigger.id, trigger.agent_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Standing approval lookup failed for trigger {trigger.id}: {exc}")
            return ""
        recipients = sorted({address for approval in standing for address in approval.recipients})
        if not recipients:
            return ""
        return (
            "\n\nStanding approval: the user pre-approved sends from this trigger to exactly "
            f"{', '.join(recipients)}. Create the draft and send it with gmail_execute_draft in this run; "
            "no further approval is needed for these recipients. Any other recipient still needs approval."
        )


_scheduler_instance: Optional[TriggerScheduler] = None


def get_trigger_scheduler() -> TriggerScheduler:
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = TriggerScheduler()
    return _scheduler_instance


__all__ = ["LIFECYCLE_SWEEP_INTERVAL", "TriggerScheduler", "get_trigger_scheduler"]
