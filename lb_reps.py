from collections import deque
from typing import Dict, List
import random


class REPSLB:
    def __init__(self,
                 mode: str = "reps_plus",
                 ev_space_size: int = 4096,
                 buffer_size: int = 8,
                 num_tiers: int = 8,
                 rtt_thresholds: List[float] = None):
        assert mode in ("reps", "reps_plus")
        self.mode = mode
        self.ev_space_size = ev_space_size
        self.buffer_size = buffer_size
        self.num_tiers = num_tiers
        if rtt_thresholds is None:
            self.rtt_thresholds = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 10.0]
        else:
            self.rtt_thresholds = rtt_thresholds
        self.state: Dict[int, dict] = {}

    def _get_state(self, flow_id: int) -> dict:
        if flow_id not in self.state:
            if self.mode == "reps":
                self.state[flow_id] = {
                    "buffer": deque(maxlen=self.buffer_size),
                    "idx": 0,
                }
            else:
                self.state[flow_id] = {
                    "buckets": [deque(maxlen=self.buffer_size)
                                for _ in range(self.num_tiers)]
                }
        return self.state[flow_id]

    def _hash(self, flow_id: int, ev: int, n: int) -> int:
        return (hash((flow_id, ev)) & 0x7fffffff) % n

    def _pick_ev_reps(self, state: dict) -> int:
        buf: deque = state["buffer"]
        if buf:
            if state["idx"] >= len(buf):
                state["idx"] = 0
            ev = buf[state["idx"]]
            state["idx"] = (state["idx"] + 1) % len(buf)
            return ev
        return random.randint(0, self.ev_space_size - 1)

    def _tier_for_ratio(self, ratio: float) -> int:
        for i, thr in enumerate(self.rtt_thresholds):
            if ratio <= thr:
                return min(i, self.num_tiers - 1)
        return self.num_tiers - 1

    def _pick_ev_reps_plus(self, state: dict) -> int:
        buckets: List[deque] = state["buckets"]
        for b in buckets:
            if b:
                ev = b[0]
                b.rotate(-1)
                return ev
        return random.randint(0, self.ev_space_size - 1)

    def choose_path(self, flow, paths: List[List[object]], now: float):
        if not paths:
            return None, 0
        state = self._get_state(flow.flow_id)
        if self.mode == "reps":
            ev = self._pick_ev_reps(state)
        else:
            ev = self._pick_ev_reps_plus(state)
        idx = self._hash(flow.flow_id, ev, len(paths))
        return paths[idx], ev

    def on_ack(self, flow, ev: int, rtt_sample: float, ecn_marked: bool):
        state = self._get_state(flow.flow_id)
        if self.mode == "reps":
            if not ecn_marked:
                buf: deque = state["buffer"]
                if ev not in buf:
                    buf.append(ev)
        else:
            if flow.base_rtt is None:
                return
            if ecn_marked:
                return
            ratio = rtt_sample / flow.base_rtt if flow.base_rtt > 0 else 1.0
            tier = self._tier_for_ratio(ratio)
            buckets: List[deque] = state["buckets"]
            b = buckets[tier]
            if ev not in b:
                b.append(ev)
