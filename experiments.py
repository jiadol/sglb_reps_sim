# file: experiments.py
import os
import random
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from sim_engine import Simulator
from traffic import (
    generate_random_flows,
    generate_incast_flows,
    FlowSpec,
)
from lb_reps import REPSLB
from lb_sglb import SGLBLB


PIC_DIR = "pic"
LOG_DIR = "log"
RAW_PATH = os.path.join(LOG_DIR, "raw.txt")
REPORT_PATH = os.path.join(LOG_DIR, "report.txt")


def init_io():
    os.makedirs(PIC_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    open(RAW_PATH, "w").close()
    open(REPORT_PATH, "w").close()


def append_raw(line: str):
    with open(RAW_PATH, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def append_report_section(title: str,
                          params: Dict[str, str],
                          headers: List[str],
                          rows: List[List[str]]):
    sep = "=" * 70
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(f"{sep}\n")
        f.write(f"{title}\n")
        f.write(f"{sep}\n\n")

        f.write("Parameters:\n")
        for k, v in params.items():
            f.write(f"  {k:<18}: {v}\n")

        f.write("\nResults:\n")

        col_width = 14
        header_line = "  " + "  ".join(f"{h:<{col_width}}" for h in headers)
        underline = "  " + "  ".join("-" * col_width for _ in headers)
        f.write(header_line + "\n")
        f.write(underline + "\n")
        for row in rows:
            row_line = "  " + "  ".join(
                f"{str(c):<{col_width}}" for c in row
            )
            f.write(row_line + "\n")
        f.write("\n\n")


def print_section(title: str):
    line = "=" * 70
    print(f"\n{line}")
    print(title)
    print(line)


def make_lb(name: str, **kwargs):
    if name == "sglb":
        return SGLBLB(**kwargs)
    elif name == "reps":
        return REPSLB(mode="reps", **kwargs)
    elif name == "reps_plus":
        return REPSLB(mode="reps_plus", **kwargs)
    else:
        raise ValueError(f"Unknown LB: {name}")


def run_scenario(lb_name: str, flow_specs: List[FlowSpec], seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    lb = make_lb(lb_name)
    sim = Simulator(lb=lb,
                    num_leaves=4,
                    num_spines=4,
                    hosts_per_leaf=4,
                    link_capacity_gbps=100.0)
    sim.add_flows(flow_specs)
    stats = sim.run()
    return stats


def extract_fct_array(stats: Dict) -> np.ndarray:
    fcts = [fs["fct"] for fs in stats["flows"].values() if fs["fct"] > 0]
    return np.array(fcts) if fcts else np.array([])


def extract_reorder_ratio(stats: Dict) -> float:
    total_reordered = 0
    total_pkts = 0
    for fs in stats["flows"].values():
        total_reordered += fs.get("reordered_pkts", 0)
        total_pkts += fs.get("total_pkts", 0)
    if total_pkts == 0:
        return 0.0
    return total_reordered / float(total_pkts)


def generate_outcast_flows(hosts: List[str],
                           sender: str,
                           num_receivers: int,
                           size_bytes: int,
                           start_time: float = 0.0,
                           first_flow_id: int = 0) -> List[FlowSpec]:
    receivers = [h for h in hosts if h != sender][:num_receivers]
    flows: List[FlowSpec] = []
    for i, r in enumerate(receivers):
        flows.append(
            FlowSpec(flow_id=first_flow_id + i,
                     src=sender,
                     dst=r,
                     size_bytes=size_bytes,
                     start_time=start_time,
                     is_mice=False)
        )
    return flows


def generate_all_to_all_flows(hosts: List[str],
                              size_bytes: int,
                              start_time: float = 0.0,
                              first_flow_id: int = 0) -> List[FlowSpec]:
    flows: List[FlowSpec] = []
    fid = first_flow_id
    for src in hosts:
        for dst in hosts:
            if src == dst:
                continue
            flows.append(
                FlowSpec(flow_id=fid,
                         src=src,
                         dst=dst,
                         size_bytes=size_bytes,
                         start_time=start_time,
                         is_mice=False)
            )
            fid += 1
    return flows


# ------------ Experiment A: Random traffic, load curve ------------

def experiment_a_random_load_curve():
    print_section("Experiment A: Random traffic, P99 FCT vs load")

    algo_names = ["sglb", "reps", "reps_plus"]
    load_levels = [0.3, 0.6, 0.9]
    base_flows = 40
    runs_per_point = 3

    results = {algo: {"load": [], "mean_p99": [], "std_p99": [], "flows": []}
               for algo in algo_names}

    print("\n[ExpA] P99 FCT per algorithm/load (averaged over runs):")
    for algo in algo_names:
        print(f"\n  Algorithm: {algo.upper()}")
        print("    load   flows   runs   mean_P99(ms)   std_P99(ms)")
        print("    ----   -----   ----   -----------   ----------")
        for rho in load_levels:
            num_flows = int(base_flows * (rho / 0.3))
            p99_list = []

            tmp_sim = Simulator(lb=make_lb(algo))
            hosts = tmp_sim.topology.hosts

            for k in range(runs_per_point):
                seed = 1000 + int(rho * 100) * 10 + k
                flows = generate_random_flows(
                    hosts=hosts,
                    num_flows=num_flows,
                    mice_fraction=0.8,
                    mice_size_bytes=128 * 1024,
                    elephant_size_bytes=32 * 1024 * 1024,
                    max_start_time=0.00005,
                )
                stats = run_scenario(algo, flows, seed=seed)
                fcts = extract_fct_array(stats)
                p99 = np.percentile(fcts, 99) if fcts.size > 0 else 0.0
                p99_list.append(p99)

                append_raw(
                    f"ExpA algo={algo} load={rho} flows={num_flows} "
                    f"run={k} seed={seed} p99_s={p99:.9f}"
                )

            mean_p99 = float(np.mean(p99_list))
            std_p99 = float(np.std(p99_list))
            results[algo]["load"].append(rho)
            results[algo]["mean_p99"].append(mean_p99)
            results[algo]["std_p99"].append(std_p99)
            results[algo]["flows"].append(num_flows)

            print(f"    {rho:>4.1f}   {num_flows:>5d}   {runs_per_point:>4d}   "
                  f"{mean_p99*1e3:>11.3f}   {std_p99*1e3:>10.3f}")

    plt.figure(figsize=(6, 4))
    for algo in algo_names:
        x = results[algo]["load"]
        y = np.array(results[algo]["mean_p99"]) * 1e3
        plt.plot(x, y, marker="o", label=algo)
    plt.xlabel("Offered load (relative)")
    plt.ylabel("P99 FCT (ms)")
    plt.title("ExpA: P99 FCT vs load (random traffic)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig_path = os.path.join(PIC_DIR, "exp1_fct_p99_vs_load.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n[ExpA] Figure saved as {fig_path}")

    params = {
        "Algorithms": ", ".join(a.upper() for a in algo_names),
        "Loads": ", ".join(f"{x:.1f}" for x in load_levels),
        "Base flows": str(base_flows),
        "Runs per point": str(runs_per_point),
        "Topology": "4 leaves, 4 spines, 4 hosts/leaf, 100 Gbps",
    }
    headers = ["Algo", "Load", "Flows", "Runs", "Mean_P99(ms)", "Std_P99(ms)"]
    rows: List[List[str]] = []
    for algo in algo_names:
        for i, rho in enumerate(results[algo]["load"]):
            rows.append([
                algo.upper(),
                f"{rho:.1f}",
                str(results[algo]["flows"][i]),
                str(runs_per_point),
                f"{results[algo]['mean_p99'][i]*1e3:.3f}",
                f"{results[algo]['std_p99'][i]*1e3:.3f}",
            ])
    append_report_section(
        "Experiment A: Random traffic, P99 FCT vs load",
        params,
        headers,
        rows,
    )


# ------------ Experiment B: Incast / Outcast ------------

def experiment_b_incast_outcast():
    print_section("Experiment B: Incast/Outcast scenarios")

    algo_names = ["sglb", "reps", "reps_plus"]
    num_senders = 8
    size_bytes = 32 * 1024 * 1024

    incast_p99 = {}
    incast_reorder = {}
    outcast_p99 = {}
    outcast_reorder = {}

    print("\n[ExpB] Incast/Outcast per algorithm:")
    for algo in algo_names:
        tmp_sim = Simulator(lb=make_lb(algo))
        hosts = tmp_sim.topology.hosts

        print(f"\n  Algorithm: {algo.upper()}")

        receiver = hosts[0]
        flows_incast = generate_incast_flows(
            hosts=hosts,
            receiver=receiver,
            num_senders=num_senders,
            size_bytes=size_bytes,
            start_time=0.0,
            first_flow_id=0,
        )
        stats_incast = run_scenario(algo, flows_incast, seed=200)
        fcts_in = extract_fct_array(stats_incast)
        p99_in = np.percentile(fcts_in, 99) if fcts_in.size > 0 else 0.0
        rratio_in = extract_reorder_ratio(stats_incast)
        incast_p99[algo] = p99_in
        incast_reorder[algo] = rratio_in

        append_raw(
            f"ExpB scenario=incast algo={algo} senders={num_senders} "
            f"size_bytes={size_bytes} p99_s={p99_in:.9f} reorder={rratio_in:.9f}"
        )

        print(f"    Incast : P99 FCT = {p99_in*1e3:8.3f} ms, "
              f"reorder = {rratio_in:8.4f}")

        sender = hosts[1]
        flows_outcast = generate_outcast_flows(
            hosts=hosts,
            sender=sender,
            num_receivers=num_senders,
            size_bytes=size_bytes,
            start_time=0.0,
            first_flow_id=0,
        )
        stats_outcast = run_scenario(algo, flows_outcast, seed=201)
        fcts_out = extract_fct_array(stats_outcast)
        p99_out = np.percentile(fcts_out, 99) if fcts_out.size > 0 else 0.0
        rratio_out = extract_reorder_ratio(stats_outcast)
        outcast_p99[algo] = p99_out
        outcast_reorder[algo] = rratio_out

        append_raw(
            f"ExpB scenario=outcast algo={algo} receivers={num_senders} "
            f"size_bytes={size_bytes} p99_s={p99_out:.9f} reorder={rratio_out:.9f}"
        )

        print(f"    Outcast: P99 FCT = {p99_out*1e3:8.3f} ms, "
              f"reorder = {rratio_out:8.4f}")

    x = np.arange(len(algo_names))

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    p99_vals = [incast_p99[a] * 1e3 for a in algo_names]
    plt.bar(x, p99_vals)
    plt.xticks(x, algo_names)
    plt.ylabel("P99 FCT (ms)")
    plt.title("ExpB: Incast P99 FCT")

    plt.subplot(1, 2, 2)
    rr_vals = [incast_reorder[a] for a in algo_names]
    plt.bar(x, rr_vals)
    plt.xticks(x, algo_names)
    plt.ylabel("Reorder ratio")
    plt.title("ExpB: Incast reorder ratio")

    plt.tight_layout()
    fig_path = os.path.join(PIC_DIR, "exp2_incast_fct_reorder.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n[ExpB] Figure saved as {fig_path}")

    print("\n[ExpB] Outcast summary (P99 FCT ms, reorder):")
    for algo in algo_names:
        print(f"  {algo.upper():8s}  P99={outcast_p99[algo]*1e3:8.3f} ms, "
              f"reorder={outcast_reorder[algo]:8.4f}")

    params = {
        "Algorithms": ", ".join(a.upper() for a in algo_names),
        "Num senders": str(num_senders),
        "Flow size (bytes)": str(size_bytes),
        "Topology": "4 leaves, 4 spines, 4 hosts/leaf, 100 Gbps",
    }
    headers = ["Scenario", "Algo", "P99_FCT(ms)", "Reorder"]
    rows: List[List[str]] = []
    for algo in algo_names:
        rows.append([
            "Incast",
            algo.upper(),
            f"{incast_p99[algo]*1e3:.3f}",
            f"{incast_reorder[algo]:.4f}",
        ])
    for algo in algo_names:
        rows.append([
            "Outcast",
            algo.upper(),
            f"{outcast_p99[algo]*1e3:.3f}",
            f"{outcast_reorder[algo]:.4f}",
        ])

    append_report_section(
        "Experiment B: Incast/Outcast scenarios",
        params,
        headers,
        rows,
    )


# ------------ Experiment B2: Shuffle (many-to-many) ------------

def experiment_b2_shuffle():
    print_section("Experiment B2: Shuffle (all-to-all) traffic")

    algo_names = ["sglb", "reps", "reps_plus"]
    size_bytes = 8 * 1024 * 1024

    p99_by_algo = {}
    reorder_by_algo = {}

    print("\n[ExpB2] Shuffle (all-to-all) per algorithm:")
    for algo in algo_names:
        lb = make_lb(algo)
        sim = Simulator(lb=lb,
                        num_leaves=4,
                        num_spines=4,
                        hosts_per_leaf=4,
                        link_capacity_gbps=100.0)
        hosts = sim.topology.hosts

        flows = generate_all_to_all_flows(
            hosts=hosts,
            size_bytes=size_bytes,
            start_time=0.0,
            first_flow_id=0,
        )
        print(f"\n  Algorithm: {algo.upper()}, all-to-all flows={len(flows)}")

        sim.add_flows(flows)
        stats = sim.run()
        fcts = extract_fct_array(stats)
        p99 = np.percentile(fcts, 99) if fcts.size > 0 else 0.0
        rratio = extract_reorder_ratio(stats)
        p99_by_algo[algo] = p99
        reorder_by_algo[algo] = rratio

        append_raw(
            f"ExpB2 scenario=shuffle algo={algo} flows={len(flows)} "
            f"size_bytes={size_bytes} p99_s={p99:.9f} reorder={rratio:.9f}"
        )

        print(f"    Shuffle: P99 FCT = {p99*1e3:8.3f} ms, "
              f"reorder = {rratio:8.4f}")

    x = np.arange(len(algo_names))

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    p99_vals = [p99_by_algo[a] * 1e3 for a in algo_names]
    plt.bar(x, p99_vals)
    plt.xticks(x, algo_names)
    plt.ylabel("P99 FCT (ms)")
    plt.title("ExpB2: Shuffle P99 FCT")

    plt.subplot(1, 2, 2)
    rr_vals = [reorder_by_algo[a] for a in algo_names]
    plt.bar(x, rr_vals)
    plt.xticks(x, algo_names)
    plt.ylabel("Reorder ratio")
    plt.title("ExpB2: Shuffle reorder ratio")

    plt.tight_layout()
    fig_path = os.path.join(PIC_DIR, "exp2b_shuffle_fct_reorder.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n[ExpB2] Figure saved as {fig_path}")

    params = {
        "Algorithms": ", ".join(a.upper() for a in algo_names),
        "Flow size (bytes)": str(size_bytes),
        "Topology": "4 leaves, 4 spines, 4 hosts/leaf, 100 Gbps",
    }
    headers = ["Algo", "P99_FCT(ms)", "Reorder"]
    rows: List[List[str]] = []
    for algo in algo_names:
        rows.append([
            algo.upper(),
            f"{p99_by_algo[algo]*1e3:.3f}",
            f"{reorder_by_algo[algo]:.4f}",
        ])

    append_report_section(
        "Experiment B2: Shuffle (all-to-all) traffic",
        params,
        headers,
        rows,
    )


# ------------ Experiment C: REPS-plus ladder sensitivity ------------

def experiment_c_reps_plus_ladder():
    print_section("Experiment C: REPS-plus RTT ladder sensitivity")

    ladder_configs = {
        "default": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 10.0],
        "tight":   [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 10.0],
        "coarse":  [1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 5.0, 10.0],
    }

    p99_by_cfg = {}
    reorder_by_cfg = {}

    print("\n[ExpC] REPS-plus ladder configs:")
    print("  config      P99 FCT (ms)   reorder ratio")
    print("  ------      ------------   -------------")

    for cfg_name, thresholds in ladder_configs.items():
        lb = REPSLB(mode="reps_plus", rtt_thresholds=thresholds)
        sim = Simulator(lb=lb,
                        num_leaves=4,
                        num_spines=4,
                        hosts_per_leaf=4,
                        link_capacity_gbps=100.0)
        hosts = sim.topology.hosts

        flows = generate_random_flows(
            hosts=hosts,
            num_flows=40,
            mice_fraction=0.7,
            mice_size_bytes=128 * 1024,
            elephant_size_bytes=16 * 1024 * 1024,
            max_start_time=0.00002,
        )
        sim.add_flows(flows)
        stats = sim.run()
        fcts = extract_fct_array(stats)
        p99 = np.percentile(fcts, 99) if fcts.size > 0 else 0.0
        rratio = extract_reorder_ratio(stats)

        p99_by_cfg[cfg_name] = p99
        reorder_by_cfg[cfg_name] = rratio

        append_raw(
            f"ExpC cfg={cfg_name} thresholds={','.join(map(str, thresholds))} "
            f"p99_s={p99:.9f} reorder={rratio:.9f}"
        )

        print(f"  {cfg_name:<8s}  {p99*1e3:12.3f}   {rratio:13.4f}")

    cfg_names = list(ladder_configs.keys())
    x = np.arange(len(cfg_names))

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    p99_vals = [p99_by_cfg[c] * 1e3 for c in cfg_names]
    plt.bar(x, p99_vals)
    plt.xticks(x, cfg_names)
    plt.ylabel("P99 FCT (ms)")
    plt.title("ExpC: REPS-plus RTT ladder (P99 FCT)")

    plt.subplot(1, 2, 2)
    rr_vals = [reorder_by_cfg[c] for c in cfg_names]
    plt.bar(x, rr_vals)
    plt.xticks(x, cfg_names)
    plt.ylabel("Reorder ratio")
    plt.title("ExpC: REPS-plus RTT ladder (reorder)")

    plt.tight_layout()
    fig_path = os.path.join(PIC_DIR, "exp3_reps_plus_ladder_p99.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n[ExpC] Figure saved as {fig_path}")

    params = {
        "Configs": ", ".join(cfg_names),
        "Flows": "40 random flows (mixed mice/elephants)",
        "Topology": "4 leaves, 4 spines, 4 hosts/leaf, 100 Gbps",
    }
    headers = ["Config", "P99_FCT(ms)", "Reorder"]
    rows: List[List[str]] = []
    for cfg_name in cfg_names:
        rows.append([
            cfg_name,
            f"{p99_by_cfg[cfg_name]*1e3:.3f}",
            f"{reorder_by_cfg[cfg_name]:.4f}",
        ])

    append_report_section(
        "Experiment C: REPS-plus RTT ladder sensitivity",
        params,
        headers,
        rows,
    )


if __name__ == "__main__":
    init_io()
    experiment_a_random_load_curve()
    experiment_b_incast_outcast()
    experiment_b2_shuffle()
    experiment_c_reps_plus_ladder()
