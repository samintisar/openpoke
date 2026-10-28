"""Agent lifecycle: status boundaries, the sweep, revival, and where the server runs it."""

from datetime import datetime, timedelta, timezone

import pytest

import server.services.trigger_scheduler as scheduler_module
from server.agents.execution_agent.runtime import ExecutionResult
from server.services.agents.cards import AgentCard
from server.services.agents.directory import InMemoryDirectory
from server.services.agents.lifecycle import LifecycleConfig, next_status, sweep
from server.services.agents.registry import SqliteAgentDirectory
from server.services.trigger_scheduler import LIFECYCLE_SWEEP_INTERVAL, TriggerScheduler
from server.services.triggers.models import TriggerRecord
from server.services.triggers.service import TriggerService
from server.services.triggers.store import TriggerStore
from server.services.triggers.utils import to_storage_timestamp, utc_now

NOW = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)
DAY = timedelta(days=1)
SECOND = timedelta(seconds=1)


def card(agent_id, idle, status="active"):
    return AgentCard(id=agent_id, name=agent_id, purpose=agent_id, status=status, created_at=NOW - 60 * DAY,
                     last_active_at=NOW - idle)


@pytest.mark.parametrize(
    ("idle", "expected"),
    [
        (0 * DAY, "active"),
        (7 * DAY - SECOND, "active"),
        (7 * DAY, "dormant"),
        (30 * DAY - SECOND, "dormant"),
        (30 * DAY, "archived"),
        (400 * DAY, "archived"),
    ],
)
def test_status_boundaries(idle, expected):
    assert next_status(card("a", idle), NOW, has_open_work=False) == expected


def test_open_work_is_never_demoted():
    for idle in (0 * DAY, 7 * DAY, 30 * DAY, 400 * DAY):
        assert next_status(card("a", idle, status="archived"), NOW, has_open_work=True) == "active"


def test_config_and_naive_timestamps():
    cfg = LifecycleConfig(dormant_after_days=1, archive_after_days=2)
    assert next_status(card("a", 1 * DAY), NOW, False, cfg) == "dormant"
    assert next_status(card("a", 2 * DAY), NOW, False, cfg) == "archived"
    naive = AgentCard(id="a", name="a", purpose="a", last_active_at=(NOW - 8 * DAY).replace(tzinfo=None))
    assert next_status(naive, NOW, False) == "dormant"


class CountingDirectory(InMemoryDirectory):
    def __init__(self, cards):
        super().__init__(cards)
        self.updates = []

    def update_card(self, agent_id, **fields):
        self.updates.append((agent_id, fields))
        super().update_card(agent_id, **fields)


def test_sweep_changes_only_what_needs_changing():
    directory = CountingDirectory([
        card("fresh", 1 * DAY),
        card("stale", 8 * DAY),
        card("old", 40 * DAY, status="dormant"),
        card("still-dormant", 10 * DAY, status="dormant"),
        card("still-archived", 90 * DAY, status="archived"),
        card("due-soon", 90 * DAY, status="archived"),
    ])
    changed = sweep(directory, NOW, open_work_ids={"due-soon", "unknown-id"})
    assert changed == {"stale": "dormant", "old": "archived", "due-soon": "active"}
    assert directory.updates == [(agent_id, {"status": status}) for agent_id, status in changed.items()]
    assert {c.id: c.status for c in directory.list_cards()} == {
        "fresh": "active", "stale": "dormant", "old": "archived", "still-dormant": "dormant",
        "still-archived": "archived", "due-soon": "active",
    }
    assert sweep(directory, NOW, open_work_ids={"due-soon"}) == {}


def test_sweep_over_sqlite_registry(tmp_path):
    directory = SqliteAgentDirectory(tmp_path / "agents.db")
    directory.add_cards([card("fresh", 1 * DAY), card("stale", 8 * DAY), card("old", 31 * DAY)])
    assert sweep(directory, NOW) == {"stale": "dormant", "old": "archived"}
    assert {c.id: c.status for c in directory.list_cards()} == {"fresh": "active", "stale": "dormant", "old": "archived"}
    # Revival: routing to an archived agent makes it active and fresh, so the next sweep leaves it alone.
    directory.touch("old", "follow up", NOW)
    assert directory.get("old").status == "active" and directory.get("old").last_active_at == NOW
    assert sweep(directory, NOW) == {}


def test_touch_revives_in_memory():
    directory = InMemoryDirectory([card("old", 40 * DAY, status="archived")])
    directory.touch("old", "any news?", NOW)
    assert directory.get("old").status == "active"
    assert sweep(directory, NOW + 6 * DAY) == {}
    assert sweep(directory, NOW + 7 * DAY) == {"old": "dormant"}


# -- Open work: triggers due soon -------------------------------------------------------------------


def test_agent_ids_with_triggers_due(tmp_path):
    svc = TriggerService(TriggerStore(tmp_path / "triggers.db"))
    now = utc_now()

    def at(delta):
        return to_storage_timestamp(now + delta)

    svc.create_trigger(agent_id="soon", agent_name="Soon", payload="p", start_time=at(timedelta(hours=2)))
    svc.create_trigger(agent_id="overdue", agent_name="Overdue", payload="p", start_time=at(-timedelta(hours=1)))
    svc.create_trigger(agent_id="later", agent_name="Later", payload="p", start_time=at(timedelta(hours=48)))
    svc.create_trigger(agent_id="paused", agent_name="Paused", payload="p", start_time=at(timedelta(hours=1)),
                       status="paused")
    assert svc.agent_ids_with_triggers_due(now) == {"soon", "overdue"}
    assert svc.agent_ids_with_triggers_due(now, within=timedelta(hours=72)) == {"soon", "overdue", "later"}


# -- Scheduler hook ---------------------------------------------------------------------------------


class FakeTriggerService:
    def __init__(self, due=()):
        self.due = set(due)
        self.scheduled = []

    def agent_ids_with_triggers_due(self, now, within=timedelta(hours=24)):
        return set(self.due)

    def schedule_next_occurrence(self, trigger, *, fired_at):
        self.scheduled.append(trigger.id)

    def record_failure(self, trigger, error):
        pass

    def clear_next_fire(self, trigger_id, *, agent_id):
        pass


def scheduler(service, directory, monkeypatch):
    monkeypatch.setattr(scheduler_module, "get_agent_directory", lambda: directory)
    instance = TriggerScheduler(poll_interval_seconds=0)
    instance._service = service
    return instance


def test_scheduler_sweep_uses_open_work(monkeypatch):
    directory = InMemoryDirectory([card("stale", 8 * DAY), card("due", 40 * DAY, status="archived")])
    instance = scheduler(FakeTriggerService(due={"due"}), directory, monkeypatch)
    assert instance.run_lifecycle_sweep(NOW) == {"stale": "dormant", "due": "active"}


def test_scheduler_sweep_failure_is_logged_not_raised(monkeypatch):
    def broken():
        raise RuntimeError("registry unavailable")

    instance = scheduler(FakeTriggerService(), None, monkeypatch)
    monkeypatch.setattr(scheduler_module, "get_agent_directory", broken)
    assert instance.run_lifecycle_sweep(NOW) == {}


def test_scheduler_sweeps_hourly(monkeypatch):
    calls = []
    monkeypatch.setattr(scheduler_module, "sweep", lambda directory, now, open_work: calls.append(now) or {})
    instance = scheduler(FakeTriggerService(), InMemoryDirectory(), monkeypatch)
    instance._maybe_sweep(NOW)
    instance._maybe_sweep(NOW + timedelta(minutes=30))
    instance._maybe_sweep(NOW + LIFECYCLE_SWEEP_INTERVAL)
    assert calls == [NOW, NOW + LIFECYCLE_SWEEP_INTERVAL]


async def test_scheduler_loop_runs_the_sweep_and_survives_its_failure(monkeypatch):
    def broken_sweep(directory, now, open_work):
        raise RuntimeError("boom")

    monkeypatch.setattr(scheduler_module, "sweep", broken_sweep)
    instance = scheduler(FakeTriggerService(), InMemoryDirectory(), monkeypatch)
    polls = []

    async def poll_once():
        polls.append(1)
        if len(polls) == 2:
            instance._running = False

    instance._poll_once = poll_once
    instance._running = True
    await instance._run()
    assert len(polls) == 2 and instance._last_sweep is not None


class StubRunner:
    calls = []

    async def run(self, agent_id, instructions, *, origin, trigger_id=None, agent_name=None):
        StubRunner.calls.append((agent_id, origin, trigger_id))
        return ExecutionResult(agent_name=agent_name or agent_id, agent_id=agent_id, success=True, response="done")


def trigger(agent_id):
    return TriggerRecord(id=7, agent_id=agent_id, agent_name="Dentist", payload="Check the dentist's reply",
                         status="active", created_at="t", updated_at="t")


async def test_trigger_fire_revives_its_agent(monkeypatch):
    directory = InMemoryDirectory([card("dentist", 40 * DAY, status="archived")])
    service = FakeTriggerService()
    instance = scheduler(service, directory, monkeypatch)
    monkeypatch.setattr(scheduler_module, "get_agent_runner", StubRunner)
    before = utc_now()
    await instance._execute_trigger(trigger("dentist"))
    revived = directory.get("dentist")
    assert revived.status == "active" and revived.last_active_at >= before
    assert StubRunner.calls[-1] == ("dentist", "trigger", 7) and service.scheduled == [7]
    assert sweep(directory, utc_now()) == {}


async def test_trigger_fire_for_unknown_agent_still_runs(monkeypatch):
    service = FakeTriggerService()
    instance = scheduler(service, InMemoryDirectory(), monkeypatch)
    monkeypatch.setattr(scheduler_module, "get_agent_runner", StubRunner)
    await instance._execute_trigger(trigger("legacy-slug"))
    assert StubRunner.calls[-1] == ("legacy-slug", "trigger", 7) and service.scheduled == [7]
