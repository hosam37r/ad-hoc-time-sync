# ad-hoc-time-sync
Simulation code for the Defending Every Tick: Comprehensive Attack Mitigation and Feasibility Analysis for Ad Hoc Time Synchronization paper

This repo contains a single-file Python simulator for a 100-node ad-hoc time-sync network. It measures the overhead (bytes, airtime, CPU time, energy) of enabling a minimal bundle of countermeasures that, together, defend against the full attack surface discussed in the paper.

It produces

1- A topology plot of the random geometric graph (RGG)

2- An energy histogram and a convergence curve

3- Three degree–resource scatter plots
    (neighbors vs total energy / CPU time / CPU energy)

4- CSV artifacts (summary.csv, per_node.csv) for analysis

# (optional) create and activate a virtual env
	python -m venv .venv
	source .venv/bin/activate               # on Windows: .venv\Scripts\activate

# install dependencies
	pip install numpy pandas matplotlib

# run with defaults (100 nodes, 60 epochs) – plots + CSVs
	python adhoc_time_sync_sim.py

# show node IDs on the topology plot
	python adhoc_time_sync_sim.py --show-ids

#Command-line options

	--nodes N            number of nodes (default: 100)
	--epochs E           number of epochs (default: 60)
	--range R            radio range in meters (default: 120)
	--loss P             baseline packet loss probability [0..1] (default: 0.02)
	--seed S             RNG seed (default: 42)
    
    --show-ids           annotate node IDs on the topology plot
    --no-topology        skip topology plotting

# Countermeasure toggles
	--no-auth            disable authenticated messaging + replay protection
	--no-consistency     disable neighbor consistency & majority voting
	--no-rtt             disable RTT wormhole/delay checks
	--no-cad             disable CAD (beacon multiplier fixed at 1)


# compare overheads by disabling countermeasures

	python adhoc_time_sync_sim.py --no-auth --no-consistency --no-rtt --no-cad

What the simulator does

Topology: Builds an RGG in a square field; nodes are neighbors if distance ≤ radio_range. Rare isolated nodes are “healed” by linking to their nearest neighbor.

Per-epoch sync loop:

        Authenticated beacons + replay protection (modeled bytes + CPU)
        (toggle: --no-auth)

        Neighbor consistency & majority voting
        Median with MAD outlier rejection; partial corrections
        (toggle: --no-consistency)

        RTT checks (wormhole/delay proxy)
        Unicast challenge/response overhead
        (toggle: --no-rtt)

        Channel-Aware Detection (CAD)
        Tracks loss in a sliding window, increases beaconing up to 3× under loss
        (toggle: --no-cad)

Cost model:
    Radio energy = electronics/bit + amplifier·distance²;
    CPU energy from verify/sign/fusion/RTT processing.



Outputs

After a run you’ll get:

    topology.png — RGG nodes and neighbor edges

    summary.csv — network totals/averages, energy split, convergence proxy

    per_node.csv — per-node degree, TX/RX, CPU/Radio time, energy, final beacon multiplier

    degree_vs_energy.png — neighbors vs total energy used (mJ)

    degree_vs_cpu_time.png — neighbors vs CPU calculation time (s)

    degree_vs_cpu_energy.png — neighbors vs CPU energy (mJ)

    Interactive plots: Per-Node Energy Consumed (histogram) and Median Absolute Residual Offset (convergence)



How to read the key graphs

    Per-Node Energy Consumed (histogram):
    X = energy used by a node over the run (mJ).
    Y = count of nodes per bin. A bar near 120 mJ with height 8 means 8 nodes each used ≈120 mJ.

    Median Absolute Residual Offset (convergence):
    Per epoch, we compute each node’s time error relative to the network median, take absolute values, then the median across nodes.
    Lower → tighter sync. Crossing 1 ms quickly indicates fast convergence.

    Neighbors vs Total Energy / CPU Time / CPU Energy:
    Degree (number of 1-hop neighbors) on X. More neighbors → more RX verifies, fusion work, and often higher TX, so these trends usually slope upward.


Configuration (edit in code)

All tunables live in SimConfig at the top of adhoc_time_sync_sim.py:

    Field size, radio_range, epochs, bitrate_bps

    Packet sizes (payload/header/auth/tag), µTESLA overhead

    CPU timings (verify/sign/fusion/RTT), radio energy coefficients

    CAD window/threshold/max multiplier

    RTT probing rate
Examples:
	
	# heavier loss
	python adhoc_time_sync_sim.py --loss 0.05
	# larger network / different connectivity
	python adhoc_time_sync_sim.py --nodes 200 --range 100



