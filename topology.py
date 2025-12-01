from typing import Dict, List, Tuple
from link import Link


class Topology:
    def __init__(self, env,
                 num_leaves: int = 4,
                 num_spines: int = 4,
                 hosts_per_leaf: int = 4,
                 link_capacity_gbps: float = 100.0,
                 prop_delay_us: float = 5.0,
                 ecn_qmin_kb: int = 64,
                 ecn_qmax_kb: int = 128):
        self.env = env
        self.num_leaves = num_leaves
        self.num_spines = num_spines
        self.hosts_per_leaf = hosts_per_leaf

        self.leaves = [f"L{i}" for i in range(num_leaves)]
        self.spines = [f"S{j}" for j in range(num_spines)]
        self.hosts: List[str] = []

        self.links: List[Link] = []
        self.host_leaf: Dict[str, str] = {}
        self.host_uplink: Dict[str, Link] = {}
        self.host_downlink: Dict[Tuple[str, str], Link] = {}
        self.leaf_to_spine: Dict[Tuple[str, str], Link] = {}
        self.spine_to_leaf: Dict[Tuple[str, str], Link] = {}
        self.paths: Dict[Tuple[str, str], List[List[Link]]] = {}

        self.capacity_bps = link_capacity_gbps * 1e9
        self.prop_delay = prop_delay_us * 1e-6
        self.ecn_qmin = ecn_qmin_kb * 1024 if ecn_qmin_kb is not None else None
        self.ecn_qmax = ecn_qmax_kb * 1024 if ecn_qmax_kb is not None else None

        self.on_packet_delivered = None

        self._build_links()
        self._build_paths()

    def _build_links(self):
        for li, leaf in enumerate(self.leaves):
            for hi in range(self.hosts_per_leaf):
                host = f"H{li}_{hi}"
                self.hosts.append(host)
                self.host_leaf[host] = leaf

                up = Link(self.env, f"{host}->{leaf}",
                          self.capacity_bps, self.prop_delay,
                          self.ecn_qmin, self.ecn_qmax)
                down = Link(self.env, f"{leaf}->{host}",
                            self.capacity_bps, self.prop_delay,
                            self.ecn_qmin, self.ecn_qmax)
                up.set_forward(self._forward_packet)
                down.set_forward(self._forward_packet)

                self.host_uplink[host] = up
                self.host_downlink[(leaf, host)] = down
                self.links.extend([up, down])

        for leaf in self.leaves:
            for spine in self.spines:
                up = Link(self.env, f"{leaf}->{spine}",
                          self.capacity_bps, self.prop_delay,
                          self.ecn_qmin, self.ecn_qmax)
                down = Link(self.env, f"{spine}->{leaf}",
                            self.capacity_bps, self.prop_delay,
                            self.ecn_qmin, self.ecn_qmax)
                up.set_forward(self._forward_packet)
                down.set_forward(self._forward_packet)
                self.leaf_to_spine[(leaf, spine)] = up
                self.spine_to_leaf[(spine, leaf)] = down
                self.links.extend([up, down])

    def _build_paths(self):
        for src in self.hosts:
            for dst in self.hosts:
                if src == dst:
                    continue
                leaf_src = self.host_leaf[src]
                leaf_dst = self.host_leaf[dst]
                paths: List[List[Link]] = []
                for spine in self.spines:
                    path = [
                        self.host_uplink[src],
                        self.leaf_to_spine[(leaf_src, spine)],
                        self.spine_to_leaf[(spine, leaf_dst)],
                        self.host_downlink[(leaf_dst, dst)],
                    ]
                    paths.append(path)
                self.paths[(src, dst)] = paths

    def get_paths(self, src: str, dst: str) -> List[List[Link]]:
        return self.paths.get((src, dst), [])

    def _forward_packet(self, packet):
        packet.hop_index += 1
        if packet.hop_index >= len(packet.path):
            if self.on_packet_delivered is not None:
                self.on_packet_delivered(packet)
        else:
            next_link = packet.path[packet.hop_index]
            next_link.put(packet)
