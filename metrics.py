from collections import defaultdict


class MetricsCollector:
    def __init__(self):
        self.max_seq = defaultdict(lambda: -1)
        self.reordered = defaultdict(int)
        self.total = defaultdict(int)

    def register_flow(self, flow):
        self.max_seq[flow.flow_id] = -1

    def on_packet_delivered(self, packet, now: float):
        fid = packet.flow_id
        seq = packet.seq
        self.total[fid] += 1

        if seq < self.max_seq[fid]:
            self.reordered[fid] += 1
        else:
            self.max_seq[fid] = seq

    def summary(self, sim_time: float, topology, flows):
        stats = {
            "flows": {},
            "links": [],
        }
        for f in flows:
            if not f.completed:
                continue
            fct = f.fct()
            thr = f.spec.size_bytes * 8.0 / fct if fct > 0 else 0.0
            fid = f.flow_id
            stats["flows"][fid] = {
                "src": f.spec.src,
                "dst": f.spec.dst,
                "size_bytes": f.spec.size_bytes,
                "fct": fct,
                "throughput_bps": thr,
                "reordered_pkts": self.reordered.get(fid, 0),
                "total_pkts": self.total.get(fid, 0),
            }
        for link in topology.links:
            stats["links"].append(link.stats(sim_time))
        return stats
