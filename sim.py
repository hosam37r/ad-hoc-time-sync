#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ad-hoc time synchronization network simulator (extensively commented + 3 extra plots)

NEW PLOTS
---------
1) Degree (neighbors) vs TOTAL energy used (mJ)  -> "power consumed" proxy
2) Degree (neighbors) vs CPU time (s)            -> "calculation time"
3) Degree (neighbors) vs CPU energy used (mJ)    -> "processing power used" proxy

WHAT'S MODELED
--------------
Topology: Random Geometric Graph (RGG) in a square field; neighbors within radio_range.
Time sync per epoch:
  (1) Authenticated beacons (+µTESLA-like overhead)  [toggle: --no-auth]
  (2) Neighbor consistency (median + MAD outlier rejection) [--no-consistency]
  (3) RTT challenge/response checks (wormhole/delay proxy)  [--no-rtt]
  (4) Channel-Aware Detection (CAD) adapts beacon rate with loss [--no-cad]

OUTPUTS
-------
- topology.png
- summary.csv, per_node.csv
- plots: energy histogram, convergence curve
- NEW: degree_vs_energy.png, degree_vs_cpu_time.png, degree_vs_cpu_energy.png

Dependencies: numpy, pandas, matplotlib
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SimConfig:
    # RNG & size
    seed: int = 42
    n_nodes: int = 100
    area_size: float = 500.0
    radio_range: float = 120.0

    # Time / epochs
    epochs: int = 60
    epoch_duration_s: float = 1.0

    # Link / PHY model
    bitrate_bps: int = 250_000  # 802.15.4-ish

    # Packet sizing (beacons)
    base_packet_bytes: int = 16
    header_bytes: int = 12
    mac_tag_bytes: int = 8
    tesla_key_overhead_bytes: int = 8

    # Packet sizing (RTT)
    rtt_payload_bytes: int = 12

    # Channel loss + CAD
    base_packet_loss: float = 0.02
    cad_window: int = 5
    cad_loss_threshold: float = 0.10
    cad_max_multiplier: int = 3

    # CPU timing/energy model
    mcu_power_mw: float = 15.0
    verify_time_s: float = 0.0002
    sign_time_s: float = 0.00025
    neighbor_fusion_time_s_per_sample: float = 0.00005
    rtt_proc_time_s: float = 0.00015

    # Radio energy model
    e_elec_j_per_bit: float = 50e-9
    e_amp_j_per_bit_m2: float = 100e-12

    # Battery (per node)
    battery_mj: float = 5000.0

    # RTT probing
    rtt_checks_per_epoch_frac: float = 0.2

    # Feature toggles
    enable_auth: bool = True
    enable_consistency: bool = True
    enable_rtt: bool = True
    enable_cad: bool = True


# =============================================================================
# Node model
# =============================================================================

class Node:
    def __init__(self, nid: int, x: float, y: float,
                 skew_ppm: float, offset_s: float, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.nid = nid
        self.x = x
        self.y = y

        # Local clock: offset + (1+skew)*t
        self.clock_skew_ppm = skew_ppm
        self.clock_offset_s = offset_s

        # 1-hop neighbors
        self.neighbors: List[int] = []

        # Accounting
        self.battery_mj = cfg.battery_mj
        self.tx_bytes = 0
        self.rx_bytes = 0
        self.tx_msgs = 0
        self.rx_msgs = 0
        self.cpu_time_s = 0.0
        self.radio_time_s = 0.0

        # Minimal replay/auth state
        self.seq_out = 0
        self.seq_in_max: Dict[int, int] = {}

        # CAD sliding window
        self.cad_window_rx_expected: List[int] = []
        self.cad_window_rx_ok: List[int] = []
        self.beacon_multiplier = 1

        # Neighbor consistency buffer
        self.recent_offsets: List[float] = []

    # Clock helpers
    def local_time(self, t_global: float) -> float:
        skew = 1.0 + self.clock_skew_ppm * 1e-6
        return self.clock_offset_s + skew * t_global

    def update_offset(self, correction: float) -> None:
        self.clock_offset_s += correction

    # Energy/time helpers
    def spend_cpu(self, seconds: float) -> None:
        self.cpu_time_s += seconds
        # Energy (mJ) = Power(W) * time(s) * 1000
        self.battery_mj -= (self.cfg.mcu_power_mw * 1e-3) * seconds * 1000.0

    def spend_radio_tx(self, bits: int, distance_m: float) -> None:
        # Electronics + amplifier*d^2
        e = self.cfg.e_elec_j_per_bit * bits + self.cfg.e_amp_j_per_bit_m2 * bits * (distance_m ** 2)
        self.battery_mj -= e * 1000.0
        self.radio_time_s += bits / self.cfg.bitrate_bps

    def spend_radio_rx(self, bits: int) -> None:
        e = self.cfg.e_elec_j_per_bit * bits
        self.battery_mj -= e * 1000.0
        self.radio_time_s += bits / self.cfg.bitrate_bps


# =============================================================================
# Network + Simulation
# =============================================================================

class Network:
    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.nodes: List[Node] = []
        self.dist_mat: Optional[np.ndarray] = None
        self._build()

    # ---- Build RGG topology ----
    def _build(self) -> None:
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        coords = np.random.rand(self.cfg.n_nodes, 2) * self.cfg.area_size
        skews_ppm = np.random.normal(loc=20.0, scale=10.0, size=self.cfg.n_nodes)
        offsets_s = np.random.uniform(-0.5, 0.5, size=self.cfg.n_nodes)

        for i in range(self.cfg.n_nodes):
            self.nodes.append(
                Node(
                    nid=i,
                    x=float(coords[i, 0]),
                    y=float(coords[i, 1]),
                    skew_ppm=float(skews_ppm[i]),
                    offset_s=float(offsets_s[i]),
                    cfg=self.cfg,
                )
            )

        self.dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        for i in range(self.cfg.n_nodes):
            nbrs = np.where((self.dist_mat[i] > 0) & (self.dist_mat[i] <= self.cfg.radio_range))[0]
            self.nodes[i].neighbors = list(map(int, nbrs))

        # Heal isolates (rare): connect to nearest neighbor
        isolated = [n.nid for n in self.nodes if not n.neighbors]
        for i in isolated:
            d = self.dist_mat[i]
            j = int(np.argsort(d)[1])  # 0 is itself; 1 is nearest other
            self.nodes[i].neighbors.append(j)
            self.nodes[j].neighbors.append(i)

    # ---- Packet sizes ----
    def broadcast_size_bits(self) -> int:
        bytes_total = self.cfg.base_packet_bytes + self.cfg.header_bytes
        if self.cfg.enable_auth:
            bytes_total += self.cfg.mac_tag_bytes + self.cfg.tesla_key_overhead_bytes
        return bytes_total * 8

    def unicast_size_bits(self) -> int:
        bytes_total = self.cfg.rtt_payload_bytes + self.cfg.header_bytes
        if self.cfg.enable_auth:
            bytes_total += self.cfg.mac_tag_bytes
        return bytes_total * 8

    # ---- Simulation loop ----
    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, List[float]]:
        epoch_err_median: List[float] = []
        for epoch in range(self.cfg.epochs):
            t = epoch * self.cfg.epoch_duration_s
            self._epoch_broadcasts(epoch, t)
            if self.cfg.enable_consistency:
                self._epoch_neighbor_consistency(epoch, t)
            else:
                for n in self.nodes:
                    n.recent_offsets = []
            if self.cfg.enable_rtt:
                self._epoch_rtt_checks(epoch, t)
            if self.cfg.enable_cad:
                self._epoch_cad_update(epoch)
            else:
                for n in self.nodes:
                    n.beacon_multiplier = 1
                    n.cad_window_rx_expected = []
                    n.cad_window_rx_ok = []

            # Convergence proxy
            residuals = [node.local_time(t) - self.nodes[0].local_time(t) for node in self.nodes]
            residuals = np.array(residuals) - np.median(residuals)
            epoch_err_median.append(float(np.median(np.abs(residuals))))
        return self._summarize(epoch_err_median)

    # ---- Epoch steps ----
    def _epoch_broadcasts(self, epoch: int, t: float) -> None:
        bits = self.broadcast_size_bits()
        for tx in self.nodes:
            n_beacons = max(1, min(tx.beacon_multiplier, self.cfg.cad_max_multiplier))
            max_distance = self.cfg.radio_range
            for _ in range(n_beacons):
                tx.tx_msgs += 1
                tx.tx_bytes += bits // 8
                tx.spend_radio_tx(bits, max_distance)
                if self.cfg.enable_auth:
                    tx.spend_cpu(self.cfg.sign_time_s)

            for nbr_id in tx.neighbors:
                rx = self.nodes[nbr_id]

                # CAD window bookkeeping
                if self.cfg.enable_cad:
                    while len(rx.cad_window_rx_expected) < self.cfg.cad_window:
                        rx.cad_window_rx_expected.append(0)
                        rx.cad_window_rx_ok.append(0)
                    if epoch % self.cfg.cad_window == 0 and epoch > 0:
                        rx.cad_window_rx_expected = rx.cad_window_rx_expected[1:] + [0]
                        rx.cad_window_rx_ok = rx.cad_window_rx_ok[1:] + [0]
                    rx.cad_window_rx_expected[-1] += n_beacons

                for _ in range(n_beacons):
                    if np.random.rand() < self.cfg.base_packet_loss:
                        continue
                    rx.rx_msgs += 1
                    rx.rx_bytes += bits // 8
                    rx.spend_radio_rx(bits)
                    if self.cfg.enable_auth:
                        rx.spend_cpu(self.cfg.verify_time_s)
                        # Simple replay state (overhead model)
                        tx.seq_out += 1
                        seq = tx.seq_out
                        prev = rx.seq_in_max.get(tx.nid, -1)
                        if seq > prev:
                            rx.seq_in_max[tx.nid] = seq

                    # Time-offset sample for fusion
                    neighbor_time = tx.local_time(t)
                    my_time = rx.local_time(t)
                    rx.recent_offsets.append(neighbor_time - my_time)
                    if len(rx.recent_offsets) > 50:
                        rx.recent_offsets = rx.recent_offsets[-50:]

                    if self.cfg.enable_cad:
                        rx.cad_window_rx_ok[-1] += 1

    def _epoch_neighbor_consistency(self, epoch: int, t: float) -> None:
        for node in self.nodes:
            if not node.recent_offsets:
                continue
            samples = np.array(node.recent_offsets)
            median = np.median(samples)
            mad = np.median(np.abs(samples - median)) + 1e-9
            keep = np.abs(samples - median) <= 3.0 * mad
            kept = samples[keep]
            if kept.size == 0:
                node.recent_offsets = []
                continue
            node.spend_cpu(self.cfg.neighbor_fusion_time_s_per_sample * len(samples))
            correction = float(np.median(kept)) * 0.5
            node.update_offset(correction)
            node.recent_offsets = []

    def _epoch_rtt_checks(self, epoch: int, t: float) -> None:
        n_checks = int(self.cfg.n_nodes * self.cfg.rtt_checks_per_epoch_frac)
        if n_checks == 0:
            return
        bits = self.unicast_size_bits()
        selected = np.random.choice(range(self.cfg.n_nodes), size=n_checks, replace=False)
        for nid in selected:
            a = self.nodes[nid]
            if not a.neighbors:
                continue
            b_id = int(np.random.choice(a.neighbors))
            b = self.nodes[b_id]
            d = min(self.cfg.radio_range, float(self.dist_mat[a.nid, b_id]))

            # A -> B challenge
            a.tx_msgs += 1
            a.tx_bytes += bits // 8
            a.spend_radio_tx(bits, d)
            if self.cfg.enable_auth:
                a.spend_cpu(self.cfg.sign_time_s)

            if np.random.rand() >= self.cfg.base_packet_loss:
                # B receives
                b.rx_msgs += 1
                b.rx_bytes += bits // 8
                b.spend_radio_rx(bits)
                if self.cfg.enable_auth:
                    b.spend_cpu(self.cfg.verify_time_s)

                # B -> A response
                b.tx_msgs += 1
                b.tx_bytes += bits // 8
                b.spend_radio_tx(bits, d)
                if self.cfg.enable_auth:
                    b.spend_cpu(self.cfg.sign_time_s)

                if np.random.rand() >= self.cfg.base_packet_loss:
                    a.rx_msgs += 1
                    a.rx_bytes += bits // 8
                    a.spend_radio_rx(bits)
                    if self.cfg.enable_auth:
                        a.spend_cpu(self.cfg.verify_time_s)
                    a.spend_cpu(self.cfg.rtt_proc_time_s)

    def _epoch_cad_update(self, epoch: int) -> None:
        if (epoch + 1) % self.cfg.cad_window != 0:
            return
        for n in self.nodes:
            expd = sum(n.cad_window_rx_expected) if n.cad_window_rx_expected else 0
            ok = sum(n.cad_window_rx_ok) if n.cad_window_rx_ok else 0
            loss = 0.0 if expd == 0 else max(0.0, 1.0 - (ok / max(expd, 1)))
            if loss > self.cfg.cad_loss_threshold:
                n.beacon_multiplier = min(self.cfg.cad_max_multiplier, n.beacon_multiplier + 1)
            else:
                n.beacon_multiplier = max(1, n.beacon_multiplier - 1)
            n.cad_window_rx_expected = []
            n.cad_window_rx_ok = []

    # ---- Summaries ----
    def _summarize(self, epoch_err_median: List[float]) -> tuple[pd.DataFrame, pd.DataFrame, List[float]]:
        total_tx_bytes = sum(n.tx_bytes for n in self.nodes)
        total_rx_bytes = sum(n.rx_bytes for n in self.nodes)
        total_tx_msgs = sum(n.tx_msgs for n in self.nodes)
        total_rx_msgs = sum(n.rx_msgs for n in self.nodes)
        cpu_time_total_s = sum(n.cpu_time_s for n in self.nodes)
        radio_time_total_s = sum(n.radio_time_s for n in self.nodes)

        cpu_energy_mj = sum((self.cfg.mcu_power_mw * 1e-3) * n.cpu_time_s * 1000.0 for n in self.nodes)
        battery_remaining_mj = sum(n.battery_mj for n in self.nodes)
        total_energy_used_mj = self.cfg.battery_mj * self.cfg.n_nodes - battery_remaining_mj
        radio_energy_mj = max(0.0, total_energy_used_mj - cpu_energy_mj)

        avg_energy_per_node_mj = total_energy_used_mj / self.cfg.n_nodes
        avg_cpu_time_per_node_s = cpu_time_total_s / self.cfg.n_nodes
        avg_radio_time_per_node_s = radio_time_total_s / self.cfg.n_nodes
        avg_beacon_multiplier = float(np.mean([n.beacon_multiplier for n in self.nodes]))

        # Convergence proxy
        thresh = 0.001
        conv_epoch = next((i for i, v in enumerate(epoch_err_median) if v < thresh), None)
        final_residual = epoch_err_median[-1] if epoch_err_median else float('nan')

        summary_rows = [
            ("nodes", self.cfg.n_nodes),
            ("epochs", self.cfg.epochs),
            ("bitrate_bps", self.cfg.bitrate_bps),
            ("base_packet_loss", self.cfg.base_packet_loss),
            ("enable_auth", self.cfg.enable_auth),
            ("enable_consistency", self.cfg.enable_consistency),
            ("enable_rtt", self.cfg.enable_rtt),
            ("enable_cad", self.cfg.enable_cad),
            ("total_tx_msgs", total_tx_msgs),
            ("total_rx_msgs", total_rx_msgs),
            ("total_tx_bytes", total_tx_bytes),
            ("total_rx_bytes", total_rx_bytes),
            ("total_energy_used_mJ", round(total_energy_used_mj, 2)),
            ("cpu_energy_mJ", round(cpu_energy_mj, 2)),
            ("radio_energy_mJ", round(radio_energy_mj, 2)),
            ("avg_energy_per_node_mJ", round(avg_energy_per_node_mj, 2)),
            ("avg_cpu_time_per_node_s", round(avg_cpu_time_per_node_s, 6)),
            ("avg_radio_time_per_node_s", round(avg_radio_time_per_node_s, 6)),
            ("avg_beacon_multiplier_final", round(avg_beacon_multiplier, 3)),
            ("convergence_epoch_<1ms", conv_epoch if conv_epoch is not None else -1),
            ("final_median_residual_s", round(final_residual, 6)),
        ]
        summary_df = pd.DataFrame(summary_rows, columns=["metric", "value"])

        # Per-node details
        per_node_rows = []
        for n in self.nodes:
            per_node_rows.append({
                "node": n.nid,
                "degree": len(n.neighbors),
                "tx_msgs": n.tx_msgs,
                "rx_msgs": n.rx_msgs,
                "tx_bytes": n.tx_bytes,
                "rx_bytes": n.rx_bytes,
                "cpu_time_s": round(n.cpu_time_s, 6),
                "radio_time_s": round(n.radio_time_s, 6),
                "battery_remaining_mJ": round(n.battery_mj, 2),
                "beacon_multiplier_final": n.beacon_multiplier
            })
        per_node_df = pd.DataFrame(per_node_rows)

        return summary_df, per_node_df, epoch_err_median


# =============================================================================
# Plotting helpers
# =============================================================================

def plot_topology(net: Network, annotate: bool = False, filename: Optional[str] = "topology.png") -> None:
    coords = np.array([[n.x, n.y] for n in net.nodes])
    plt.figure(figsize=(6.6, 6.6))
    ax = plt.gca()
    # Draw edges once (i<j)
    for i, ni in enumerate(net.nodes):
        for j in ni.neighbors:
            if j > i:
                nj = net.nodes[j]
                ax.plot([ni.x, nj.x], [ni.y, nj.y], linewidth=0.5, alpha=0.35)
    ax.scatter(coords[:, 0], coords[:, 1], s=25)
    if annotate:
        for n in net.nodes:
            ax.text(n.x, n.y, str(n.nid), fontsize=7)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Ad-hoc Topology (N={len(net.nodes)}, range={net.cfg.radio_range} m)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=200)
    plt.show()


# =============================================================================
# CLI / Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Ad-hoc time sync security overhead simulator")
    # Topology / scale
    parser.add_argument("--nodes", type=int, default=100, help="number of nodes")
    parser.add_argument("--epochs", type=int, default=60, help="number of epochs")
    parser.add_argument("--range", type=float, default=120.0, help="radio range (m)")
    parser.add_argument("--loss", type=float, default=0.02, help="baseline packet loss probability")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")

    # Plotting
    parser.add_argument("--show-ids", action="store_true", help="annotate node IDs on topology plot")
    parser.add_argument("--no-topology", action="store_true", help="skip topology plotting")

    # Feature toggles (countermeasures)
    parser.add_argument("--no-auth", action="store_true", help="disable authenticated messaging + replay protection")
    parser.add_argument("--no-consistency", action="store_true", help="disable neighbor consistency & majority voting")
    parser.add_argument("--no-rtt", action="store_true", help="disable RTT wormhole/delay checks")
    parser.add_argument("--no-cad", action="store_true", help="disable CAD (beacon multiplier always 1)")

    args = parser.parse_args()

    # Build config with flags applied
    cfg = SimConfig(
        n_nodes=args.nodes,
        epochs=args.epochs,
        radio_range=args.range,
        base_packet_loss=args.loss,
        seed=args.seed,
        enable_auth=not args.no_auth,
        enable_consistency=not args.no_consistency,
        enable_rtt=not args.no_rtt,
        enable_cad=not args.no_cad,
    )

    # Build network + topology plot
    net = Network(cfg)
    if not args.no_topology:
        plot_topology(net, annotate=args.show_ids, filename="topology.png")

    # Run simulation
    summary_df, per_node_df, residuals = net.run()

    # Enhance per-node with extra derived metrics for plotting
    per_node_df["energy_used_mJ"] = cfg.battery_mj - per_node_df["battery_remaining_mJ"]
    per_node_df["cpu_energy_mJ"] = (cfg.mcu_power_mw * 1e-3) * per_node_df["cpu_time_s"] * 1000.0  # J->mJ

    # Save CSVs
    summary_df.to_csv("summary.csv", index=False)
    per_node_df.to_csv("per_node.csv", index=False)

    # Console summary
    print("\n=== Simulation Summary ===")
    print(summary_df.to_string(index=False))

    # Plot: per-node energy histogram
    plt.figure(figsize=(7, 4))
    plt.hist(per_node_df["energy_used_mJ"].values, bins=20)
    plt.title("Per-Node Energy Consumed (mJ)")
    plt.xlabel("Energy Consumed (mJ)")
    plt.ylabel("Nodes")
    plt.tight_layout()
    plt.show()

    # Plot: convergence curve
    plt.figure(figsize=(7, 4))
    plt.plot(range(len(residuals)), residuals)
    plt.title("Median Absolute Residual Offset vs Network Median")
    plt.xlabel("Epoch")
    plt.ylabel("Residual (s)")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------------
    # NEW SCATTER PLOTS (degree vs energy/time/cpu-energy)
    # ---------------------------------------------------------------------
    deg = per_node_df["degree"].values
    energy_used = per_node_df["energy_used_mJ"].values
    cpu_time = per_node_df["cpu_time_s"].values
    cpu_energy = per_node_df["cpu_energy_mJ"].values

    # (1) Degree vs total energy used (mJ)
    plt.figure(figsize=(7, 4))
    plt.scatter(deg, energy_used, s=28)
    plt.title("Neighbors vs Total Energy Used")
    plt.xlabel("Number of Neighbors (Degree)")
    plt.ylabel("Total Energy Used (mJ)")
    plt.tight_layout()
    plt.savefig("degree_vs_energy.png", dpi=200)
    plt.show()

    # (2) Degree vs CPU time (s)
    plt.figure(figsize=(7, 4))
    plt.scatter(deg, cpu_time, s=28)
    plt.title("Neighbors vs CPU Calculation Time")
    plt.xlabel("Number of Neighbors (Degree)")
    plt.ylabel("CPU Time (s)")
    plt.tight_layout()
    plt.savefig("degree_vs_cpu_time.png", dpi=200)
    plt.show()

    # (3) Degree vs CPU energy (mJ)
    plt.figure(figsize=(7, 4))
    plt.scatter(deg, cpu_energy, s=28)
    plt.title("Neighbors vs CPU Energy (Processing Power Used)")
    plt.xlabel("Number of Neighbors (Degree)")
    plt.ylabel("CPU Energy (mJ)")
    plt.tight_layout()
    plt.savefig("degree_vs_cpu_energy.png", dpi=200)
    plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
