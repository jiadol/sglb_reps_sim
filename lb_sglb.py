from typing import List
import random


class SGLBLB:
    def __init__(self,
                 w_max_queue: float = 1.0,
                 w_avg_queue: float = 1.0,
                 w_latency: float = 1e6):
        self.w_max = w_max_queue
        self.w_avg = w_avg_queue
        self.w_lat = w_latency

    def choose_path(self, flow, paths: List[List[object]], now: float):
        if not paths:
            return None, 0
        scores = []
        for path in paths:
            if not path:
                scores.append(float("inf"))
                continue
            qs = [link.queue_len_bytes for link in path]
            max_q = max(qs)
            avg_q = sum(qs) / len(qs)
            lat = sum(link.prop_delay for link in path)
            score = self.w_max * max_q + self.w_avg * avg_q + self.w_lat * lat
            scores.append(score)
        min_score = min(scores)
        best_indices = [i for i, s in enumerate(scores) if s == min_score]
        idx = random.choice(best_indices)
        return paths[idx], 0

    def on_ack(self, flow, ev: int, rtt_sample: float, ecn_marked: bool):
        return
