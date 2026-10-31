"""Bounded execution of CPU-heavy model inference.

Request handlers run in FastAPI's threadpool and call `runtime.run(...)`.
Work executes on a dedicated pool of `INFERENCE_WORKERS` threads; at most
`workers + queue_limit` jobs may be admitted at once. When full, `run` raises
`InferenceBusy` immediately (→ HTTP 503 + Retry-After) instead of queueing
without bound. The asyncio event loop never executes model code.
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, TypeVar

from backend.config import settings

log = logging.getLogger(__name__)
T = TypeVar("T")


class InferenceBusy(Exception):
    pass


class InferenceUnavailable(Exception):
    pass


@dataclass
class _Stats:
    completed: int = 0
    rejected: int = 0
    failed: int = 0
    timeouts: int = 0
    latencies: collections.deque = field(default_factory=lambda: collections.deque(maxlen=500))


class InferenceRuntime:
    def __init__(self, workers: int, queue_limit: int) -> None:
        self.workers = max(1, workers)
        self.capacity = self.workers + max(0, queue_limit)
        self._executor: Optional[ThreadPoolExecutor] = None
        self._slots = threading.BoundedSemaphore(self.capacity)
        self._inflight = 0
        self._lock = threading.Lock()
        self.stats = _Stats()
        self.provider = None
        self.models: list[dict[str, Any]] = []
        self.warmup_state = "NOT_STARTED"   # NOT_STARTED | WARMING | READY | DEGRADED | FAILED
        self.warmup_started_at: Optional[datetime] = None
        self.warmup_finished_at: Optional[datetime] = None
        self.started_at = datetime.utcnow()

    # ── lifecycle ──────────────────────────────────────────────────────────
    def configure(self, provider) -> None:
        self.provider = provider
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="inference")

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def warmup(self, on_done: Optional[Callable[[list[dict[str, Any]]], None]] = None) -> None:
        if self.provider is None:
            raise InferenceUnavailable("No inference provider configured")
        self.warmup_state = "WARMING"
        self.warmup_started_at = datetime.utcnow()
        try:
            results = self.provider.warmup()
        except Exception as exc:  # noqa: BLE001
            log.exception("Warm-up crashed")
            results = [{"name": "provider", "task": "all", "status": "FAILED", "error": str(exc)[:300]}]
        self.models = results
        failed = [m for m in results if m.get("status") != "READY"]
        self.warmup_state = "READY" if not failed else ("FAILED" if len(failed) == len(results) else "DEGRADED")
        self.warmup_finished_at = datetime.utcnow()
        log.info("Model warm-up finished: %s", self.warmup_state)
        if on_done:
            on_done(results)

    def warmup_async(self, on_done=None) -> threading.Thread:
        t = threading.Thread(target=self.warmup, args=(on_done,), daemon=True, name="model-warmup")
        t.start()
        return t

    @property
    def ready(self) -> bool:
        return self.warmup_state == "READY"

    # ── execution ──────────────────────────────────────────────────────────
    def run(self, fn: Callable[..., T], *args: Any, timeout: float = 30.0) -> T:
        if self._executor is None or self.provider is None:
            raise InferenceUnavailable("Inference runtime is not running")
        if not self._slots.acquire(blocking=False):
            self.stats.rejected += 1
            raise InferenceBusy("Inference capacity exhausted")
        with self._lock:
            self._inflight += 1
        t0 = time.perf_counter()
        try:
            future = self._executor.submit(fn, *args)
            try:
                result = future.result(timeout=timeout)
            except FutureTimeout:
                self.stats.timeouts += 1
                raise
            self.stats.completed += 1
            self.stats.latencies.append((time.perf_counter() - t0) * 1000)
            return result
        except FutureTimeout:
            raise
        except Exception:
            self.stats.failed += 1
            raise
        finally:
            with self._lock:
                self._inflight -= 1
            self._slots.release()

    def snapshot(self) -> dict[str, Any]:
        lat = sorted(self.stats.latencies)

        def pct(p: float) -> Optional[float]:
            if not lat:
                return None
            k = min(len(lat) - 1, max(0, int(round(p / 100.0 * (len(lat) - 1)))))
            return round(lat[k], 1)

        return {
            "provider": getattr(self.provider, "name", None),
            "warmup_state": self.warmup_state,
            "ready": self.ready,
            "workers": self.workers,
            "capacity": self.capacity,
            "in_flight": self._inflight,
            "completed": self.stats.completed,
            "rejected": self.stats.rejected,
            "failed": self.stats.failed,
            "timeouts": self.stats.timeouts,
            "latency_ms": {"p50": pct(50), "p95": pct(95), "p99": pct(99), "samples": len(lat)},
            "models": self.models,
        }


runtime = InferenceRuntime(settings.inference_workers, settings.inference_queue_limit)


def build_provider():
    if settings.inference_provider == "fake":
        from backend.services.biometric.providers import FakeProvider
        log.warning("Using FakeProvider — APP_ENV=test only; results are synthetic.")
        return FakeProvider()
    from backend.services.biometric.providers import DeepFaceProvider
    return DeepFaceProvider(settings.landmarker_model_path)
