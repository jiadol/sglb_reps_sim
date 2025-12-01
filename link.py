import simpy
import random


class Link:
    def __init__(self, env: simpy.Environment, name: str,
                 capacity_bps: float, prop_delay: float,
                 ecn_qmin_bytes: int = None, ecn_qmax_bytes: int = None):
        self.env = env
        self.name = name
        self.capacity_bps = capacity_bps
        self.prop_delay = prop_delay
        self.ecn_qmin = ecn_qmin_bytes
        self.ecn_qmax = ecn_qmax_bytes

        self.store = simpy.Store(env)
        self.queue_len_bytes = 0
        self.queue_max_bytes = 0
        self.queue_area = 0.0
        self.last_q_update = env.now
        self.busy_time = 0.0

        self.forward = lambda pkt: None
        self.env.process(self._run())

    def set_forward(self, cb):
        self.forward = cb

    def _update_queue_area(self):
        now = self.env.now
        dt = now - self.last_q_update
        if dt > 0:
            self.queue_area += self.queue_len_bytes * dt
            self.last_q_update = now

    def put(self, packet):
        self._update_queue_area()
        self.queue_len_bytes += packet.size_bytes
        if self.queue_len_bytes > self.queue_max_bytes:
            self.queue_max_bytes = self.queue_len_bytes

        if self.ecn_qmin is not None and self.ecn_qmax is not None:
            q = self.queue_len_bytes
            if q >= self.ecn_qmax:
                packet.ecn_marked = True
            elif q >= self.ecn_qmin:
                p = (q - self.ecn_qmin) / float(self.ecn_qmax - self.ecn_qmin)
                p = max(0.0, min(1.0, p))
                if random.random() < p:
                    packet.ecn_marked = True

        return self.store.put(packet)

    def _run(self):
        while True:
            pkt = (yield self.store.get())
            size_bits = pkt.size_bytes * 8.0
            service_time = size_bits / self.capacity_bps if self.capacity_bps > 0 else 0.0
            self.busy_time += service_time
            yield self.env.timeout(service_time + self.prop_delay)
            self._update_queue_area()
            self.queue_len_bytes -= pkt.size_bytes
            if self.queue_len_bytes < 0:
                self.queue_len_bytes = 0
            self.forward(pkt)

    def finalize(self, sim_time: float):
        dt = sim_time - self.last_q_update
        if dt > 0:
            self.queue_area += self.queue_len_bytes * dt
            self.last_q_update = sim_time

    def stats(self, sim_time: float):
        self.finalize(sim_time)
        avg_q = self.queue_area / sim_time if sim_time > 0 else 0.0
        util = self.busy_time / sim_time if sim_time > 0 else 0.0
        return {
            "name": self.name,
            "avg_queue_bytes": avg_q,
            "max_queue_bytes": self.queue_max_bytes,
            "utilization": util,
        }
