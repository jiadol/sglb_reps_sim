import simpy
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from traffic import FlowSpec


@dataclass
class Packet:
    flow_id: int
    seq: int
    size_bytes: int
    src: str
    dst: str
    path: List[object] = field(default_factory=list)
    hop_index: int = 0
    send_time: float = 0.0
    ev: int = 0
    ecn_marked: bool = False


class Flow:
    def __init__(self,
                 env: simpy.Environment,
                 spec: FlowSpec,
                 topology,
                 lb,
                 mtu_bytes: int = 1500):
        self.env = env
        self.spec = spec
        self.topology = topology
        self.lb = lb

        self.flow_id = spec.flow_id
        self.mtu_bytes = mtu_bytes

        self.bytes_sent = 0
        self.bytes_acked = 0
        self.bytes_in_flight = 0

        self.next_seq = 0
        self.inflight: Dict[int, Packet] = {}

        self.base_rtt: Optional[float] = None
        self.target_rtt: Optional[float] = None
        self.rtt_target_factor = 1.5

        self.cwnd_bytes = 10 * mtu_bytes
        self.cwnd_max_bytes = 1e9
        self.ema_ecn = 0.0
        self.ema_alpha = 0.1
        self.alpha_dec = 0.1
        self.fair_dec_bytes = mtu_bytes
        self.fair_inc_bytes = mtu_bytes * 0.1

        self.ack_store = simpy.Store(env)

        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.completed = False

    def _send_one_packet(self):
        remaining = self.spec.size_bytes - self.bytes_sent
        if remaining <= 0:
            return
        size = min(self.mtu_bytes, remaining)
        seq = self.next_seq
        self.next_seq += 1

        pkt = Packet(flow_id=self.flow_id,
                     seq=seq,
                     size_bytes=size,
                     src=self.spec.src,
                     dst=self.spec.dst)

        paths = self.topology.get_paths(self.spec.src, self.spec.dst)
        path, ev = self.lb.choose_path(self, paths, self.env.now)
        if path is None:
            return

        pkt.path = path
        pkt.hop_index = 0
        pkt.ev = ev
        pkt.send_time = self.env.now

        self.inflight[seq] = pkt
        self.bytes_sent += size
        self.bytes_in_flight += size

        first_link = pkt.path[0]
        first_link.put(pkt)

        if self.start_time is None:
            self.start_time = self.env.now

    def handle_ack(self, packet: Packet, rtt_sample: float, ecn_marked: bool):
        self.ack_store.put((packet.seq, rtt_sample, ecn_marked, packet.ev))

    def _update_cwnd(self, rtt_sample: float, ecn_marked: bool):
        if self.base_rtt is None:
            return
        if self.target_rtt is None:
            self.target_rtt = self.base_rtt * self.rtt_target_factor

        self.ema_ecn = (1.0 - self.ema_alpha) * self.ema_ecn + \
                       self.ema_alpha * (1.0 if ecn_marked else 0.0)
        high_rtt = rtt_sample > self.target_rtt

        if ecn_marked and high_rtt and self.ema_ecn > 0.25:
            self.cwnd_bytes *= (1.0 - self.alpha_dec)
        elif ecn_marked and not high_rtt:
            self.cwnd_bytes -= self.fair_dec_bytes
        elif (not ecn_marked) and high_rtt:
            self.cwnd_bytes += self.fair_inc_bytes
        else:
            self.cwnd_bytes += self.mtu_bytes

        if self.cwnd_bytes < self.mtu_bytes:
            self.cwnd_bytes = self.mtu_bytes
        if self.cwnd_bytes > self.cwnd_max_bytes:
            self.cwnd_bytes = self.cwnd_max_bytes

    def _on_ack(self, seq: int, rtt_sample: float, ecn_marked: bool, ev: int):
        pkt = self.inflight.pop(seq, None)
        if pkt:
            self.bytes_acked += pkt.size_bytes
            self.bytes_in_flight -= pkt.size_bytes
        if self.base_rtt is None or rtt_sample < self.base_rtt:
            self.base_rtt = rtt_sample
            self.target_rtt = self.base_rtt * self.rtt_target_factor

        self._update_cwnd(rtt_sample, ecn_marked)
        self.lb.on_ack(self, ev, rtt_sample, ecn_marked)

    def run(self):
        yield self.env.timeout(self.spec.start_time)
        while self.bytes_acked < self.spec.size_bytes:
            while self.bytes_in_flight < self.cwnd_bytes and self.bytes_sent < self.spec.size_bytes:
                self._send_one_packet()
                if self.bytes_sent >= self.spec.size_bytes:
                    break
            seq, rtt, ecn, ev = (yield self.ack_store.get())
            self._on_ack(seq, rtt, ecn, ev)

        self.end_time = self.env.now
        self.completed = True

    def fct(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
