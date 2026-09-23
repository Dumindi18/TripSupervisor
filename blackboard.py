"""
Blackboard: shared state that multiple agents read/write concurrently.

mode="unsafe" (default) -> NO locking. Concurrent read-modify-write sequences
    from different agents can interleave arbitrarily, producing genuine
    lost updates, stale reads, and dirty reads -- real race conditions,
    not simulated ones.
mode="safe"   -> every op is serialized with a per-key asyncio.Lock, so you
    can run the exact same workflow with races eliminated, as a control.

Every read/write is recorded as a BlackboardEvent (who, when, what key,
what value, what version they saw / produced). This history is the raw
material for attributing a downstream failure back to a specific race.
"""
import asyncio
import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional


@dataclass
class BlackboardEvent:
    ts: float
    agent_id: str
    op: str  # "read" | "write"
    key: str
    value: Any
    version_seen: Optional[int] = None   # for reads: version of the value read
    version_after: Optional[int] = None  # for writes: version after this write


class Blackboard:
    def __init__(self, mode: str = "unsafe"):
        assert mode in ("safe", "unsafe")
        self.mode = mode
        self._data: Dict[str, Any] = {}
        self._versions: Dict[str, int] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self.history: List[BlackboardEvent] = []

    def _lock_for(self, key: str) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    def _read_unlocked(self, agent_id: str, key: str, default=None):
        value = self._data.get(key, default)
        version = self._versions.get(key, 0)
        self.history.append(BlackboardEvent(time.time(), agent_id, "read", key, value, version_seen=version))
        return value

    def _write_unlocked(self, agent_id: str, key: str, value: Any):
        self._versions[key] = self._versions.get(key, 0) + 1
        self._data[key] = value
        self.history.append(BlackboardEvent(time.time(), agent_id, "write", key, value, version_after=self._versions[key]))

    async def read(self, agent_id: str, key: str, default=None) -> Any:
        if self.mode == "safe":
            async with self._lock_for(key):
                return self._read_unlocked(agent_id, key, default)
        return self._read_unlocked(agent_id, key, default)

    async def write(self, agent_id: str, key: str, value: Any):
        if self.mode == "safe":
            async with self._lock_for(key):
                self._write_unlocked(agent_id, key, value)
                return
        self._write_unlocked(agent_id, key, value)

    async def read_modify_write(
        self,
        agent_id: str,
        key: str,
        modify_fn: Callable[[Any], Any],
        default: Any = None,
        delay: float = 0.0,
    ) -> Any:
        """
        The canonical race shape: read current value -> do work (delay
        simulates LLM latency) -> write back a value derived from what was
        read. In unsafe mode, if agent B reads while agent A's write is
        still pending, B's write can clobber A's -- a lost update.
        modify_fn may be sync or async.
        """
        if self.mode == "safe":
            async with self._lock_for(key):
                current = self._read_unlocked(agent_id, key, default)
                if delay:
                    await asyncio.sleep(delay)
                new_value = modify_fn(current)
                if asyncio.iscoroutine(new_value):
                    new_value = await new_value
                self._write_unlocked(agent_id, key, new_value)
                return new_value

        current = self._read_unlocked(agent_id, key, default)
        if delay:
            await asyncio.sleep(delay)
        new_value = modify_fn(current)
        if asyncio.iscoroutine(new_value):
            new_value = await new_value
        self._write_unlocked(agent_id, key, new_value)
        return new_value

    def snapshot(self) -> Dict[str, Any]:
        return dict(self._data)

    def dump_history(self, path: str):
        with open(path, "w") as f:
            for ev in self.history:
                f.write(json.dumps(asdict(ev)) + "\n")

    def verify_accumulator(self, key: str, expected_value: float, tolerance: float = 0.01) -> Dict[str, Any]:
        """
        For keys that multiple agents concurrently subtract/add to (e.g. a
        shared running budget), detect_lost_updates() is the wrong tool: it
        flags every different-agent overwrite as "lost", but on an
        accumulator every write is SUPPOSED to overwrite the previous value
        -- that's only a bug if the write was computed from a stale read.

        This checks the thing that actually matters: does the final stored
        value match what it should be if every update had correctly built
        on the one before it? A mismatch means at least one agent's
        read-modify-write used a stale base value -- a genuine lost update,
        even though the write history alone can't distinguish it from a
        legitimate sequential update.
        """
        actual = self._data.get(key)
        race_detected = actual is None or abs(actual - expected_value) > tolerance
        return {
            "key": key,
            "expected": expected_value,
            "actual": actual,
            "race_detected": race_detected,
        }

    def detect_lost_updates(self, key: str) -> List[dict]:
        """
        Post-hoc analysis: find writes to `key` whose version was
        immediately superseded by another agent's write before anyone
        could have read it -- i.e. writes that were silently discarded.

        NOTE: this is the right tool for a key where each write fully
        REPLACES the value (e.g. a shared text draft, where the last
        writer's version genuinely discards the others' edits). It is
        the WRONG tool for an accumulator key that multiple agents
        legitimately subtract/add to in sequence -- there, every write is
        supposed to overwrite the previous one, so this will over-count.
        Use verify_accumulator() for that case instead.
        """
        writes = [e for e in self.history if e.op == "write" and e.key == key]
        writes.sort(key=lambda e: e.version_after)
        lost = []
        for i in range(len(writes) - 1):
            current, nxt = writes[i], writes[i + 1]
            if nxt.agent_id != current.agent_id and (nxt.ts - current.ts) < 5.0:
                lost.append({
                    "key": key,
                    "overwritten_agent": current.agent_id,
                    "overwritten_version": current.version_after,
                    "overwritten_value_preview": str(current.value)[:100],
                    "overwriting_agent": nxt.agent_id,
                    "overwriting_version": nxt.version_after,
                    "time_gap_s": nxt.ts - current.ts,
                })
        return lost
