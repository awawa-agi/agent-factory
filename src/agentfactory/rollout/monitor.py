from __future__ import annotations

"""Light‑weight async ExecutionMonitor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   Keeps track of active / completed tasks
*   Maintains per‑URL load statistics ("how many rollouts are hitting each server")
*   Renders a live tqdm progress‑bar that shows active tasks & the busiest URLs

Designed for AnyIO (works on asyncio / Trio back‑ends).

Usage example is included at the bottom – run the file directly to see the bar.
"""

from contextlib import asynccontextmanager
from typing import List, Optional, Set, Dict
from collections import Counter

import anyio
import tqdm.asyncio as tqdm

__all__ = ["ExecutionMonitor"]


class ExecutionMonitor:
    """A minimal concurrent monitor with URL load balancing helper."""

    # ------------------------------------------------------------------
    # construction & context‑manager lifecycle
    # ------------------------------------------------------------------

    def __init__(
        self,
        total_tasks: int,
        description: str = "Processing rollouts",
        max_display_urls: int = 5,
    ) -> None:
        self.total_tasks = total_tasks
        self.description = description
        self.max_display_urls = max_display_urls

        # internal state
        self._lock = anyio.Lock()
        self.active_ids: Set[int | str] = set()
        self.done = 0
        self.url_load: Counter[str] = Counter()
        self.pbar: tqdm.tqdm | None = None

    async def __aenter__(self) -> "ExecutionMonitor":
        self.pbar = tqdm.tqdm(
            total=self.total_tasks,
            desc=self.description,
            unit="task",
            dynamic_ncols=True,
            leave=True,
            smoothing=0.03,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: D401
        if self.pbar:
            self.pbar.close()
            self.pbar = None
        return False  # don't suppress exceptions

    # ------------------------------------------------------------------
    # URL allocation helper (atomic under the same lock)
    # ------------------------------------------------------------------

    async def allocate_url(self, base_urls: List[str]) -> str:
        """Pick the least‑loaded URL **and reserve a slot** (+=1)."""
        async with self._lock:
            for u in base_urls:
                self.url_load.setdefault(u, 0)
            url = min(base_urls, key=lambda u: self.url_load[u])
            self.url_load[url] += 1  # reserve immediately – avoids race window
            return url

    # ------------------------------------------------------------------
    # Task tracking – expose as async context‑manager
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def track(self, rollout_id: int | str, url: Optional[str] = None):
        """Register a task while it runs – use with *async with* syntax."""
        # enter
        async with self._lock:
            self.active_ids.add(rollout_id)
            self._update_postfix_locked()
        try:
            yield  # --------‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑  roll the task  ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
        finally:
            # exit – update counts & progress bar
            async with self._lock:
                self.active_ids.discard(rollout_id)
                self.done += 1
                if url:
                    self.url_load[url] = max(0, self.url_load[url] - 1)
                if self.pbar:
                    self.pbar.update(1)
                self._update_postfix_locked()

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _update_postfix_locked(self) -> None:
        """Assumes caller already holds *self._lock*."""
        if not self.pbar:
            return

        postfix: Dict[str, str | int] = {
            "active": len(self.active_ids),
        }

        if self.url_load:
            top = sorted(
                self.url_load.items(), key=lambda x: x[1], reverse=True
            )[: self.max_display_urls]
            info = [f"{u.split('//')[-1].split('/')[0]}:{n}" for u, n in top if n > 0]
            if info:
                postfix["urls"] = ",".join(info)

        self.pbar.set_postfix(postfix)


# ----------------------------------------------------------------------
# example usage – run with «python execution_monitor.py»
# ----------------------------------------------------------------------

async def _fake_job(rid: int, monitor: ExecutionMonitor, base_urls):
    """Dummy coroutine that sleeps for a bit to simulate work."""
    url = await monitor.allocate_url(base_urls)
    async with monitor.track(rid, url):
        # pretend workload
        await anyio.sleep(1.0 + 0.1 * (rid % 5) + rid * 0.5)


async def _demo():
    n = 30
    base_urls = [
        "http://server1:8000",
        "http://server2:8000",
        "http://server3:8000",
    ]
    async with ExecutionMonitor(total_tasks=n, description="Rollouts demo") as mon:
        async with anyio.create_task_group() as tg:
            for rid in range(n):
                tg.start_soon(_fake_job, rid, mon, base_urls)


if __name__ == "__main__":
    anyio.run(_demo)
