ad-hoc-time-sync
=================
Simulation code for the **Defending Every Tick: Comprehensive Attack Mitigation and Feasibility Analysis for Ad Hoc Time Synchronization** paper

This repo contains a single-file Python simulator for ad-hoc time-sync networks with multiple security-by-design suites (A–D; E optional for ablation). It measures overhead bytes, airtime, CPU time, energy, convergence, and overhead message counts attributable to robustness mechanisms.

It produces:
  1) A topology plot of the random geometric graph (RGG)
  2) An energy histogram and a convergence curve
  3) Four continuous degree→resource curves (neighbors vs total energy / CPU time / CPU energy / overhead messages)
  4) Per-suite CSV artifacts (summary_*.csv, per_node_*.csv) and overlay plots for all suites

---------------------------------------------------------------------
Quick start
---------------------------------------------------------------------

(optional) create and activate a virtual env
--------------------------------------------
```bash
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
```

Install dependencies
--------------------
```bash
python3 -m pip install --upgrade pip
python3 -m pip install numpy pandas matplotlib
```

Run with defaults (compare A–D on the same topology)
----------------------------------------------------
```bash
python3 sim.py --compare-suites
```

Include Suite E as an ablation (optional)
----------------------------------------
```bash
python3 sim.py --compare-suites --suites suite-a,suite-b,suite-c,suite-d,suite-e
```

Run a single suite (e.g., Suite A) – plots + CSVs
-------------------------------------------------
```bash
python3 sim.py --profile suite-a
```

Show node IDs on the topology plot (single-suite or first suite in compare)
---------------------------------------------------------------------------
```bash
python3 sim.py --profile suite-a --show-ids
```

---------------------------------------------------------------------
Command-line options
---------------------------------------------------------------------

```text
--nodes N                 number of nodes (default: 100)
--epochs E                number of epochs (default: 60)
--range R                 radio range in meters (default: 120)
--loss P                  baseline packet loss probability [0..1] (default: 0.02)
--seed S                  RNG seed (default: 42)

--profile suite-x         run a single suite
                          choices: suite-a, suite-b, suite-c, suite-d, suite-e

--compare-suites          compare multiple suites on the same topology
--suites LIST             comma-separated suites to compare
                          default: suite-a,suite-b,suite-c,suite-d

--msg-overhead-bcast B    override: extra bytes added to EVERY beacon (all suites)
--msg-overhead-uni U      override: extra bytes added to EVERY RTT unicast (all suites)

--show-ids                annotate node IDs on the topology plot
--no-topology             skip topology plotting
```

Examples
--------
```bash
# larger network, more loss
python3 sim.py --compare-suites --nodes 200 --range 100 --loss 0.05

# quick ablation: add global measured overhead bytes
python3 sim.py --compare-suites --msg-overhead-bcast 6 --msg-overhead-uni 3
```

---------------------------------------------------------------------
Countermeasure suites (profiles)
---------------------------------------------------------------------

All suites are evaluated under a common security-by-design floor; differences are in how they realize that floor and what overheads they incur.

- Suite A — Software Baseline
  Auth + anti-replay, neighbor consistency (median+MAD) with bounded corrections, RTT ON, CAD ON.
  Default message overhead bytes: beacon=0, unicast=0.

- Suite B — PHY-Hardened
  PHY distance-bounding/integrity in beacons (replaces RTT), Auth + anti-replay, CAD ON.
  Adds PHY bytes + per-beacon PHY CPU.
  Defaults: beacon OH=4, unicast OH=4.

- Suite C — Identity-Aware
  NISA/PASID-style identity profiling (clock skew/RSSI/LQI), Auth + anti-replay, RTT ON, CAD ON.
  Adds identity bytes + per-sample identity CPU.
  Defaults: beacon OH=4, unicast OH=2.

- Suite D — Trust-Validated & Conservative
  Self-stabilizing sync, Auth + anti-replay + per-message trust validation, RTT ON, CAD OFF, lower correction gain.
  Defaults: beacon OH=2, unicast OH=6.

- Suite E — Lightweight / Energy-Savvy (optional ablation)
  Minimal extras, low-rate RTT, CAD OFF, light TSPM CPU.
  Defaults: beacon OH=0, unicast OH=0.

Why Suite B often shows fewer “overhead messages”: it disables RTT; its cost shifts into extra PHY bytes and per-beacon PHY CPU.

---------------------------------------------------------------------
What the simulator does
---------------------------------------------------------------------

Topology
--------
Builds an RGG in a square field; nodes are neighbors if distance ≤ radio_range. Rare isolates are “healed” by linking to nearest neighbor. The same seed/topology is used across suites in --compare-suites.

Per-epoch sync loop
-------------------
- Authenticated beacons + anti-replay (bytes + sign/verify CPU).
- Neighbor consistency (median with MAD outlier rejection; bounded-rate corrections).
- RTT checks (challenge/response, unicast overhead) when enabled.
- CAD (channel-aware detection): sliding loss estimate; beacon multiplier increases up to 3× under loss (A,B,C). D keeps fixed rate.

Message overhead — two notions
------------------------------
1) Overhead messages (count) = CAD-added beacons beyond 1 beacon/epoch/node + all RTT unicast messages.
2) Per-packet overhead bytes = extra bytes added to every beacon/unicast (suite-specific; measured/estimated). These increase airtime/energy linearly and are tracked separately.

Cost model
----------
- Radio energy: electronics/bit + amplifier·distance².
- CPU energy: constant MCU power × per-action service time (sign/verify, fusion, RTT proc, PHY check, identity/trust).

Convergence metric
------------------
Median absolute residual to the network median per epoch; convergence epoch = first epoch < 1 ms.

---------------------------------------------------------------------
Outputs
---------------------------------------------------------------------

Per suite (e.g., suite-a):
- Topology
  topology_suite-a.png — RGG nodes and neighbor edges (drawn once per compare run).

- CSVs
  summary_suite-a.csv — network totals/averages, energy split, convergence proxy, overhead message/byte totals.
  per_node_suite-a.csv — per-node degree, TX/RX, CPU/Radio time, energy, final beacon multiplier, overhead message/byte counts.

- Per-suite plots
  degree_vs_energy_curve.png — neighbors vs total energy (mJ)
  degree_vs_cpu_time_curve.png — neighbors vs CPU time (s)
  degree_vs_cpu_energy_curve.png — neighbors vs CPU energy (mJ)
  degree_vs_overhead_msgs_curve.png — neighbors vs overhead messages (count)
  Plus interactive: Energy histogram and Convergence curve.

- Overlay plots (selected suites)
  compare_convergence.png
  compare_degree_vs_energy.png
  compare_degree_vs_cpu_time.png
  compare_degree_vs_cpu_energy.png
  compare_degree_vs_overhead_msgs.png

---------------------------------------------------------------------
Configuration (edit in code)
---------------------------------------------------------------------

All tunables live in SimConfig at the top of sim.py:
- Field size, radio_range, epochs, bitrate_bps
- Packet sizes (payload/header/auth/tag), μTESLA overhead
- Per-suite extras: identity bytes, PHY bytes, message overhead bytes (beacon/unicast)
- CPU timings (verify/sign/fusion/RTT/PHY/identity/trust), radio energy coefficients
- CAD window/threshold/max multiplier, RTT probing rate, correction gain

Per-suite defaults are set in make_profile_cfg(...).

Override measured per-packet overhead bytes at runtime:
```bash
python3 sim.py --compare-suites --msg-overhead-bcast 6 --msg-overhead-uni 3
```

More examples
-------------
```bash
# heavier loss
python3 sim.py --compare-suites --loss 0.05

# larger network / different connectivity
python3 sim.py --compare-suites --nodes 200 --range 100
```

---------------------------------------------------------------------
Notes & limitations
---------------------------------------------------------------------
- First-order radio model only; no CSMA/CCA or interference.
- CPU model uses fixed per-action times; no cache/memory effects.
- Loss is i.i.d. Bernoulli per link.
- Security evaluation is architectural (by design, via traceability), not an empirical red-team test.

---------------------------------------------------------------------
License
---------------------------------------------------------------------
Choose a license (e.g., MIT) and add it here.
