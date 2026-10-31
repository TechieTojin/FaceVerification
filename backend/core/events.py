"""In-process publish/subscribe bus feeding Server-Sent Events.

Publishers are usually sync request handlers running in the threadpool, so
`publish` hands items to each subscriber's asyncio queue via
`call_soon_threadsafe`. Each subscriber queue is bounded; a slow client drops
its oldest events rather than growing memory (it will refetch on reconnect).

Tenant isolation: every event carries the organisation it belongs to, and
optionally a recipient user. A subscriber only receives events of its own
organisation (and, for user-addressed events, only its own). Events without an
organisation are never delivered to browsers.

Limitation: the bus is per-process. With several API worker processes each
worker only fans out its own events; the dashboard also refetches on a timer.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)

_QUEUE_SIZE = 100
# Channels whose events must name a recipient; they are never broadcast to a whole organisation.
USER_ADDRESSED_CHANNELS = frozenset({"notifications", "pairing"})


@dataclass(eq=False)  # identity-hashable so subscribers can live in a set
class _Subscriber:
    queue: asyncio.Queue
    loop: asyncio.AbstractEventLoop
    channels: frozenset[str]
    org_id: str
    user_id: str | None = None
    dropped: int = field(default=0)


class EventBus:
    def __init__(self) -> None:
        self._subs: set[_Subscriber] = set()
        self._lock = threading.Lock()

    def subscribe(self, channels: set[str], org_id: str, user_id: str | None = None) -> _Subscriber:
        sub = _Subscriber(asyncio.Queue(maxsize=_QUEUE_SIZE), asyncio.get_running_loop(), frozenset(channels),
                          org_id, user_id)
        with self._lock:
            self._subs.add(sub)
        return sub

    def unsubscribe(self, sub: _Subscriber) -> None:
        with self._lock:
            self._subs.discard(sub)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)

    def publish(self, channel: str, event: str, data: dict[str, Any], org_id: str | None = None,
                user_id: str | None = None) -> None:
        if org_id is None or (channel in USER_ADDRESSED_CHANNELS and user_id is None):
            return
        payload = json.dumps({"event": event, "channel": channel, "at": datetime.utcnow().isoformat() + "Z", "data": data}, default=str)
        with self._lock:
            targets = [s for s in self._subs
                       if channel in s.channels and s.org_id == org_id and (user_id is None or s.user_id == user_id)]
        for sub in targets:
            try:
                sub.loop.call_soon_threadsafe(self._offer, sub, payload)
            except RuntimeError:
                self.unsubscribe(sub)

    @staticmethod
    def _offer(sub: _Subscriber, payload: str) -> None:
        if sub.queue.full():
            try:
                sub.queue.get_nowait()
                sub.dropped += 1
            except asyncio.QueueEmpty:
                pass
        sub.queue.put_nowait(payload)


bus = EventBus()
