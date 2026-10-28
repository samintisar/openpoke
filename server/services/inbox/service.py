"""The interaction inbox: the one queue through which anything reaches the interaction agent.

User messages, execution-agent results, trigger results and watcher alerts are posted here and persisted.
A single consumer task runs one interaction-agent turn at a time:

1. It waits for work.
2. If the oldest pending item is not a user message, it waits `debounce_seconds` so near-simultaneous results
   coalesce. User messages skip the wait.
3. It takes every pending item and marks it `processing`.
4. User messages (joined in order) run as one user turn; the other items, joined into one agent message, run
   as a second turn right after.
5. It marks the items `done`, also when the turn failed (the error is logged; items are never retried).

Items that arrive during a turn wait for the next one. On start, items a previous process left `processing`
are reset to `pending` and processed again; in-flight execution-agent runs are not resumed.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional, Sequence

from ...logging_config import logger
from .store import KINDS, InboxItem, InboxStore


def _default_runtime() -> Any:
    from ...agents.interaction_agent.runtime import InteractionAgentRuntime

    return InteractionAgentRuntime()


def _default_conversation_log() -> Any:
    from ..conversation import get_conversation_log

    return get_conversation_log()


class InteractionInbox:
    """Durable queue plus the single consumer that turns its items into interaction-agent turns."""

    def __init__(
        self,
        store: InboxStore,
        *,
        debounce_seconds: float = 2.0,
        runtime_factory: Callable[[], Any] = _default_runtime,
        conversation_log: Optional[Any] = None,
    ) -> None:
        self.store = store
        self.debounce_seconds = debounce_seconds
        self._runtime_factory = runtime_factory
        self._conversation_log = conversation_log
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._wake: Optional[asyncio.Event] = None
        self._task: Optional[asyncio.Task[None]] = None

    # -- producers ---------------------------------------------------------------------------------------------

    def post(self, kind: str, payload: str, agent_id: Optional[str] = None) -> InboxItem:
        """Persist an item and wake the consumer.

        A user message is written to the conversation log first, so the chat shows it at once and its log entry
        is never later than the item's `created_at` (the time approvals compare against).
        """
        if kind not in KINDS:
            raise ValueError(f"Unknown inbox kind: {kind!r}")
        if kind == "user":
            (self._conversation_log or _default_conversation_log()).record_user_message(payload)
        item = self.store.insert(kind, payload, agent_id, created_at=datetime.now(timezone.utc))
        self._notify()
        return item

    def _notify(self) -> None:
        loop, wake = self._loop, self._wake
        if loop is None or wake is None or loop.is_closed():
            return  # not started: the item waits in the store for `start`
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            wake.set()
        else:
            loop.call_soon_threadsafe(wake.set)

    # -- lifecycle ---------------------------------------------------------------------------------------------

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        recovered = self.store.reset_processing()
        if recovered:
            logger.info(f"Inbox: {recovered} item(s) left mid-turn are pending again")
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        if self.store.pending():
            self._wake.set()
        self._task = self._loop.create_task(self._run(), name="interaction-inbox")
        logger.info("Interaction inbox started")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self._loop = None
        self._wake = None
        logger.info("Interaction inbox stopped")

    # -- consumer ----------------------------------------------------------------------------------------------

    async def _run(self) -> None:
        assert self._wake is not None
        while True:
            await self._wake.wait()
            self._wake.clear()
            try:
                while await self._process_next_batch():
                    pass
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive (store failure); retry on the next wake
                logger.exception("Inbox consumer failed; retrying shortly")
                await asyncio.sleep(1.0)
                self._wake.set()

    async def _process_next_batch(self) -> bool:
        """Run the turns for everything pending now; False when there was nothing."""
        pending = self.store.pending()
        if not pending:
            return False
        if pending[0].kind != "user" and self.debounce_seconds > 0:
            await asyncio.sleep(self.debounce_seconds)
            pending = self.store.pending()
        self.store.mark([item.id for item in pending], "processing")

        users = [item for item in pending if item.kind == "user"]
        others = [item for item in pending if item.kind != "user"]
        if users:
            await self._run_turn(users, self._user_turn)
        if others:
            await self._run_turn(others, self._agent_turn)
        return True

    async def _run_turn(self, items: List[InboxItem], turn: Callable[[Any, List[InboxItem]], Any]) -> None:
        ids = [item.id for item in items]
        try:
            runtime = self._runtime_factory()
            result = await turn(runtime, items)
            if result is not None and not getattr(result, "success", True):
                logger.error(f"Inbox: interaction turn for items {ids} failed: {getattr(result, 'error', None)}")
        except asyncio.CancelledError:
            raise  # stopped mid-turn: the items stay `processing` and are reset on the next start
        except Exception as exc:
            logger.exception(f"Inbox: interaction turn for items {ids} raised: {exc}")
        self.store.mark(ids, "done")

    async def _user_turn(self, runtime: Any, items: Sequence[InboxItem]) -> Any:
        texts = [item.payload for item in items]
        user_message_at = max(item.created_at for item in items)
        return await runtime.execute("\n\n".join(texts), user_message_at=user_message_at, recorded=texts)

    async def _agent_turn(self, runtime: Any, items: Sequence[InboxItem]) -> Any:
        kinds = {item.kind for item in items}
        origin = kinds.pop() if len(kinds) == 1 else "agent"
        return await runtime.handle_agent_message("\n".join(item.payload for item in items), origin=origin)


__all__ = ["InteractionInbox"]
