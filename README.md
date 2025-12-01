# SGLB & REPS Network Simulator

This project is a packet-level network simulator based on `SimPy`. It implements and compares load balancing algorithms—specifically **SGLB**, **REPS**, and **REPS+**—within a Leaf-Spine data center topology. The simulator allows you to run various traffic scenarios (Random, Incast, Outcast, Shuffle) and generate performance reports and visualization plots.

## Table of Contents

1.  [Prerequisites & Installation](https://www.google.com/search?q=%23prerequisites--installation)
2.  [Project Structure](https://www.google.com/search?q=%23project-structure)
3.  [Running Experiments](https://www.google.com/search?q=%23running-experiments)
4.  [Understanding the Output](https://www.google.com/search?q=%23understanding-the-output)
5.  [Configuration & Customization](https://www.google.com/search?q=%23configuration--customization)
6.  [Adding New Load Balancers](https://www.google.com/search?q=%23adding-new-load-balancers)

-----

## Prerequisites & Installation

### System Requirements

  * **Python 3.10+**
  * Recommended: A virtual environment (venv or conda).

### Dependencies

The project relies on `simpy` for the discrete-event simulation engine, `numpy` for calculations, and `matplotlib` for plotting results.

1.  **Clone or download** the repository.
2.  **Install dependencies** using pip:

<!-- end list -->

```bash
pip install simpy numpy matplotlib
```

-----

## Project Structure

  * **`experiments.py`**: The main entry point. Contains the simulation definitions (Exp A, B, B2, C) and executes the runs.
  * **`sim_engine.py`**: The core simulator class that manages the environment, topology, and flow generation.
  * **`topology.py`**: Defines the Leaf-Spine network topology and packet routing logic.
  * **`traffic.py`**: Helper functions to generate different traffic patterns (Random, Incast, All-to-All).
  * **`lb_reps.py`**: Implementation of the REPS and REPS+ load balancing algorithms.
  * **`lb_sglb.py`**: Implementation of the SGLB load balancing algorithm.
  * **`transport.py`**: Implements the flow control (CC), packet handling, and Ack logic.
  * **`metrics.py`**: Collects statistics like FCT (Flow Completion Time) and packet reordering.

-----

## Running Experiments

The project is designed to run all defined experiments sequentially via the `experiments.py` script.

To start the simulation:

```bash
python experiments.py
```

### Included Experiments

When you run the script, it executes the following scenarios automatically:

1.  **Experiment A: Random Traffic Load Curve**

      * Varies the network load (0.3, 0.6, 0.9).
      * Measures P99 Flow Completion Time (FCT) for SGLB, REPS, and REPS+.
      * Generates: `pic/exp1_fct_p99_vs_load.png`.

2.  **Experiment B: Incast & Outcast**

      * **Incast:** Multiple senders targeting one receiver.
      * **Outcast:** One sender targeting multiple receivers.
      * Measures P99 FCT and Packet Reordering ratios.
      * Generates: `pic/exp2_incast_fct_reorder.png`.

3.  **Experiment B2: Shuffle (All-to-All)**

      * Every host sends traffic to every other host.
      * Generates: `pic/exp2b_shuffle_fct_reorder.png`.

4.  **Experiment C: REPS+ Sensitivity**

      * Tests REPS+ with different RTT threshold "ladders" (Default, Tight, Coarse).
      * Generates: `pic/exp3_reps_plus_ladder_p99.png`.

-----

## Understanding the Output

Upon running the simulation, the script creates two directories in the root folder: `log/` and `pic/`.

### 1\. Visualizations (`pic/`)

  * **`exp1_fct_p99_vs_load.png`**: Line chart comparing P99 FCT latency across load levels.
  * **`exp2_incast_fct_reorder.png`**: Bar charts for Incast scenarios showing FCT and reordering.
  * **`exp2b_shuffle_fct_reorder.png`**: Bar charts for the All-to-All shuffle scenario.
  * **`exp3_reps_plus_ladder_p99.png`**: Comparison of REPS+ performance under different configuration parameters.

### 2\. Logs (`log/`)

  * **`report.txt`**: A human-readable summary table. It contains the exact numerical results (Mean P99, Standard Deviation, Reorder Ratios) formatted for easy reading.
  * **`raw.txt`**: A raw dump of every experiment run, including seed values. Useful for debugging or parsing data for custom plotting.

-----

## Configuration & Customization

Currently, simulation parameters are defined directly within `experiments.py`. To customize the simulation, open `experiments.py` and modify the following variables:

### Changing Network Topology

The default topology is a 4-Leaf, 4-Spine network with 4 hosts per leaf. To change this, find the `Simulator` initialization in `experiments.py` (inside `run_scenario` or specific experiment functions):

```python
sim = Simulator(
    lb=lb,
    num_leaves=8,         # Increase number of leaf switches
    num_spines=8,         # Increase number of spine switches
    hosts_per_leaf=8,     # Hosts per rack
    link_capacity_gbps=40.0 # Change link bandwidth
)
```

### Adjusting Traffic

Inside specific experiment functions (e.g., `experiment_a_random_load_curve`), you can tweak traffic generation parameters:

```python
flows = generate_random_flows(
    hosts=hosts,
    num_flows=100,              # Number of flows to simulate
    mice_fraction=0.9,          # Percentage of small flows
    mice_size_bytes=10 * 1024,  # Size of small flows
    elephant_size_bytes=50 * 1024 * 1024 # Size of large flows
)
```

### Tuning REPS+ Parameters

To modify the RTT thresholds for REPS+, navigate to `experiment_c_reps_plus_ladder` in `experiments.py` and modify the `ladder_configs` dictionary:

```python
ladder_configs = {
    "my_custom_config": [1.05, 1.1, 1.2, 1.5, 2.0, 5.0, 10.0],
    # ...
}
```

-----

## Adding New Load Balancers

To implement a new Load Balancing algorithm (e.g., ECMP or LetFlow), follow the pattern in `lb_sglb.py` or `lb_reps.py`.

1.  **Create a new class** (e.g., `MyNewLB`).
2.  **Implement the required methods**:
      * `choose_path(self, flow, paths, now)`: Returns the selected path (list of links) and an `ev` (flow label/cookie).
      * `on_ack(self, flow, ev, rtt_sample, ecn_marked)`: Updates internal state based on packet feedback.
3.  **Register the LB**:
    In `experiments.py`, update the `make_lb` function to include your new class:

<!-- end list -->

```python
def make_lb(name: str, **kwargs):
    if name == "sglb":
        return SGLBLB(**kwargs)
    # ...
    elif name == "my_new_lb":
        return MyNewLB(**kwargs) # Add this
```