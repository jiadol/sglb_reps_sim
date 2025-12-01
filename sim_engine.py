import simpy
from topology import Topology
from transport import Flow
from metrics import MetricsCollector


class Simulator:
    def __init__(self, lb,
                 num_leaves: int = 4,
                 num_spines: int = 4,
                 hosts_per_leaf: int = 4,
                 link_capacity_gbps: float = 100.0,
                 prop_delay_us: float = 5.0,
                 ecn_qmin_kb: int = 64,
                 ecn_qmax_kb: int = 128):
        self.env = simpy.Environment()
        self.topology = Topology(self.env,
                                  num_leaves=num_leaves,
                                  num_spines=num_spines,
                                  hosts_per_leaf=hosts_per_leaf,
                                  link_capacity_gbps=link_capacity_gbps,
                                  prop_delay_us=prop_delay_us,
                                  ecn_qmin_kb=ecn_qmin_kb,
                                  ecn_qmax_kb=ecn_qmax_kb)
        self.lb = lb
        self.metrics = MetricsCollector()
        self.flows = {}

        self.topology.on_packet_delivered = self._on_packet_delivered

    def add_flows(self, flow_specs):
        for spec in flow_specs:
            flow = Flow(self.env, spec, self.topology, self.lb)
            self.flows[spec.flow_id] = flow
            self.metrics.register_flow(flow)
            self.env.process(flow.run())

    def _on_packet_delivered(self, packet):
        self.metrics.on_packet_delivered(packet, self.env.now)
        flow = self.flows.get(packet.flow_id)
        if flow is None:
            return
        rtt_sample = self.env.now - packet.send_time
        flow.handle_ack(packet, rtt_sample, packet.ecn_marked)

    def run(self, until=None):
        if until is not None:
            self.env.run(until=until)
        else:
            self.env.run()
        return self.metrics.summary(self.env.now, self.topology, list(self.flows.values()))
