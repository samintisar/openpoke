from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from zoneinfo import ZoneInfo

from ...logging_config import logger
from ..approvals import gate as approval_gate
from .models import TriggerRecord
from .store import TriggerStore
from .utils import (
    build_recurrence,
    coerce_start_datetime,
    load_rrule,
    normalize_status,
    parse_iso,
    resolve_timezone,
    to_storage_timestamp,
    utc_now,
)


MISSED_TRIGGER_GRACE_PERIOD = timedelta(minutes=5)
# An agent whose trigger fires within this window has open work and is never demoted by the lifecycle.
OPEN_WORK_WINDOW = timedelta(hours=24)


class TriggerService:
    """High-level trigger management with recurrence awareness."""

    def __init__(self, store: TriggerStore):
        self._store = store

    def create_trigger(
        self,
        *,
        agent_id: str,
        agent_name: str,
        payload: str,
        recurrence_rule: Optional[str] = None,
        start_time: Optional[str] = None,
        timezone_name: Optional[str] = None,
        status: Optional[str] = None,
    ) -> TriggerRecord:
        tz = resolve_timezone(timezone_name)
        now = utc_now()
        start_dt_local = coerce_start_datetime(start_time, tz, now)
        stored_recurrence = build_recurrence(recurrence_rule, start_dt_local, tz)
        next_fire = self._compute_next_fire(
            stored_recurrence=stored_recurrence,
            start_dt_local=start_dt_local,
            tz=tz,
            now=now,
        )
        timestamp = to_storage_timestamp(now)
        record: Dict[str, Any] = {
            "agent_id": agent_id,
            "agent_name": agent_name,
            "payload": payload,
            "start_time": to_storage_timestamp(start_dt_local),
            "next_trigger": to_storage_timestamp(next_fire) if next_fire else None,
            "recurrence_rule": stored_recurrence,
            "timezone": getattr(tz, "key", "UTC"),
            "status": normalize_status(status),
            "last_error": None,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        trigger_id = self._store.insert(record)
        created = self._store.fetch_one(trigger_id, agent_id)
        if not created:  # pragma: no cover - defensive
            raise RuntimeError("Failed to load trigger after insert")
        return created

    def update_trigger(
        self,
        trigger_id: int,
        *,
        agent_id: str,
        payload: Optional[str] = None,
        recurrence_rule: Optional[str] = None,
        start_time: Optional[str] = None,
        timezone_name: Optional[str] = None,
        status: Optional[str] = None,
        last_error: Optional[str] = None,
        clear_error: bool = False,
    ) -> Optional[TriggerRecord]:
        existing = self._store.fetch_one(trigger_id, agent_id)
        if existing is None:
            return None

        tz = resolve_timezone(timezone_name or existing.timezone)
        start_reference = (
            parse_iso(existing.start_time)
            if existing.start_time
            else utc_now()
        )
        start_dt_local = coerce_start_datetime(start_time, tz, start_reference)

        fields: Dict[str, Any] = {}
        if payload is not None:
            fields["payload"] = payload

        normalized_status = None
        status_changed_to_active = False
        if status is not None:
            normalized_status = normalize_status(status)
            fields["status"] = normalized_status
            status_changed_to_active = (
                normalized_status == "active" and existing.status != "active"
            )
            if normalized_status != "active":
                # A paused or completed trigger loses its standing send approvals; resuming does not restore them.
                approval_gate.revoke_standing(trigger_id)
        else:
            normalized_status = existing.status

        if start_time is not None:
            fields["start_time"] = to_storage_timestamp(start_dt_local.astimezone(tz))
        if timezone_name is not None:
            fields["timezone"] = getattr(tz, "key", "UTC")

        schedule_inputs_changed = any(
            value is not None for value in (recurrence_rule, start_time, timezone_name)
        )

        recurrence_source = (
            recurrence_rule if recurrence_rule is not None else existing.recurrence_rule
        )
        if schedule_inputs_changed:
            stored_recurrence = (
                build_recurrence(recurrence_source, start_dt_local, tz)
                if recurrence_source
                else None
            )
        else:
            stored_recurrence = recurrence_source

        next_trigger_dt = (
            parse_iso(existing.next_trigger)
            if existing.next_trigger
            else None
        )
        now = utc_now()
        should_recompute_schedule = schedule_inputs_changed

        if status_changed_to_active:
            if next_trigger_dt is None:
                should_recompute_schedule = True
            else:
                missed_duration = now - next_trigger_dt
                if missed_duration > MISSED_TRIGGER_GRACE_PERIOD:
                    should_recompute_schedule = True

        if should_recompute_schedule:
            next_fire = self._compute_next_fire(
                stored_recurrence=stored_recurrence,
                start_dt_local=start_dt_local,
                tz=tz,
                now=now,
            )
            if (
                stored_recurrence is None
                and recurrence_rule is None
                and start_time is None
                and status_changed_to_active
                and next_fire is not None
                and next_fire <= now
            ):
                next_fire = now
            fields["next_trigger"] = (
                to_storage_timestamp(next_fire) if next_fire else None
            )
            if schedule_inputs_changed:
                fields["recurrence_rule"] = stored_recurrence
        elif schedule_inputs_changed:
            fields["recurrence_rule"] = stored_recurrence

        if clear_error:
            fields["last_error"] = None
        elif last_error is not None:
            fields["last_error"] = last_error

        if not fields:
            return existing

        updated = self._store.update(trigger_id, agent_id, fields)
        return self._store.fetch_one(trigger_id, agent_id) if updated else existing

    def get_trigger(self, trigger_id: int) -> Optional[TriggerRecord]:
        """Any agent's trigger by id (the interaction agent approves recurring sends by trigger id)."""
        return self._store.fetch_by_id(trigger_id)

    def list_triggers(self, *, agent_id: str) -> List[TriggerRecord]:
        return self._store.list_for_agent(agent_id)

    def get_due_triggers(
        self, *, before: datetime, agent_id: Optional[str] = None
    ) -> List[TriggerRecord]:
        iso_cutoff = to_storage_timestamp(before)
        return self._store.fetch_due(agent_id, iso_cutoff)

    def agent_ids_with_triggers_due(
        self, now: datetime, within: timedelta = OPEN_WORK_WINDOW
    ) -> set[str]:
        """Agents with an active trigger due by `now + within` (overdue included): open work for the lifecycle."""
        return {
            trigger.agent_id
            for trigger in self.get_due_triggers(before=now + within)
            if trigger.agent_id
        }

    def backfill_agent_ids(self, resolve: Callable[[str], Optional[str]]) -> int:
        """Give rows written before agent ids existed an `agent_id`, resolved from their `agent_name`."""
        return self._store.backfill_agent_ids(resolve)

    def delete_trigger(self, trigger_id: int, *, agent_id: str) -> bool:
        """Delete one of `agent_id`'s triggers, revoking its standing send approvals; False if not found."""
        if self._store.fetch_one(trigger_id, agent_id) is None:
            return False
        approval_gate.revoke_standing(trigger_id)
        return self._store.delete(trigger_id, agent_id)

    def mark_as_completed(self, trigger_id: int, *, agent_id: str) -> None:
        approval_gate.revoke_standing(trigger_id)
        self._store.update(
            trigger_id,
            agent_id,
            {
                "status": "completed",
                "next_trigger": None,
                "last_error": None,
            },
        )

    def schedule_next_occurrence(
        self,
        trigger: TriggerRecord,
        *,
        fired_at: datetime,
    ) -> Optional[TriggerRecord]:
        if not trigger.recurrence_rule:
            self.mark_as_completed(trigger.id, agent_id=trigger.agent_id)
            return self._store.fetch_one(trigger.id, trigger.agent_id)

        tz = resolve_timezone(trigger.timezone)
        next_fire = self._compute_next_after(trigger.recurrence_rule, fired_at, tz)
        fields: Dict[str, Any] = {
            "next_trigger": to_storage_timestamp(next_fire) if next_fire else None,
            "last_error": None,
        }
        if next_fire is None:
            fields["status"] = "completed"
            approval_gate.revoke_standing(trigger.id)
        self._store.update(trigger.id, trigger.agent_id, fields)
        return self._store.fetch_one(trigger.id, trigger.agent_id)

    def record_failure(self, trigger: TriggerRecord, error: str) -> None:
        self._store.update(
            trigger.id,
            trigger.agent_id,
            {
                "last_error": error,
            },
        )

    def clear_next_fire(self, trigger_id: int, *, agent_id: str) -> Optional[TriggerRecord]:
        self._store.update(
            trigger_id,
            agent_id,
            {
                "next_trigger": None,
            },
        )
        return self._store.fetch_one(trigger_id, agent_id)

    def clear_all(self) -> None:
        approval_gate.revoke_standing()  # every trigger is deleted, so every standing approval goes
        self._store.clear_all()

    def _compute_next_fire(
        self,
        *,
        stored_recurrence: Optional[str],
        start_dt_local: datetime,
        tz: ZoneInfo,
        now: datetime,
    ) -> Optional[datetime]:
        if stored_recurrence:
            rule = load_rrule(stored_recurrence)
            next_occurrence = rule.after(now.astimezone(tz), inc=True)
            if next_occurrence is None:
                return None
            if next_occurrence.tzinfo is None:
                next_occurrence = next_occurrence.replace(tzinfo=tz)
            return next_occurrence.astimezone(tz)

        if start_dt_local < now.astimezone(tz):
            logger.warning(
                "start_time in the past; trigger will fire immediately",
                extra={"start_time": start_dt_local.isoformat()},
            )
        return start_dt_local

    def _compute_next_after(
        self,
        stored_recurrence: str,
        fired_at: datetime,
        tz: ZoneInfo,
    ) -> Optional[datetime]:
        rule = load_rrule(stored_recurrence)
        next_occurrence = rule.after(fired_at.astimezone(tz), inc=False)
        if next_occurrence is None:
            return None
        if next_occurrence.tzinfo is None:
            next_occurrence = next_occurrence.replace(tzinfo=tz)
        return next_occurrence.astimezone(tz)


__all__ = ["TriggerService", "MISSED_TRIGGER_GRACE_PERIOD", "OPEN_WORK_WINDOW"]
