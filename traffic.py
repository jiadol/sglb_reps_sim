from dataclasses import dataclass
from typing import List
import random


@dataclass
class FlowSpec:
    flow_id: int
    src: str
    dst: str
    size_bytes: int
    start_time: float
    is_mice: bool = True


def generate_random_flows(hosts: List[str],
                          num_flows: int = 20,
                          mice_fraction: float = 0.8,
                          mice_size_bytes: int = 128 * 1024,
                          elephant_size_bytes: int = 32 * 1024 * 1024,
                          max_start_time: float = 0.0001) -> List[FlowSpec]:
    flows: List[FlowSpec] = []
    for fid in range(num_flows):
        src, dst = random.sample(hosts, 2)
        is_mice = random.random() < mice_fraction
        size = mice_size_bytes if is_mice else elephant_size_bytes
        start = random.random() * max_start_time
        flows.append(FlowSpec(fid, src, dst, size, start, is_mice))
    return flows


def generate_incast_flows(hosts: List[str],
                          receiver: str,
                          num_senders: int,
                          size_bytes: int,
                          start_time: float = 0.0,
                          first_flow_id: int = 0) -> List[FlowSpec]:
    senders = [h for h in hosts if h != receiver][:num_senders]
    flows: List[FlowSpec] = []
    for i, s in enumerate(senders):
        flows.append(
            FlowSpec(flow_id=first_flow_id + i,
                     src=s,
                     dst=receiver,
                     size_bytes=size_bytes,
                     start_time=start_time,
                     is_mice=False)
        )
    return flows
