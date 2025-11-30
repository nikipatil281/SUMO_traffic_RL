#!/usr/bin/env python3
import os
import sys
import time
import csv
import subprocess
from pathlib import Path
import random  # NEW: for stochastic vehicle injection

# Project root = parent of this 'scripts' folder
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
NETDIR = ROOT / "envs" / "grid2x2"
CFG = NETDIR / "grid.sumocfg"


def ensure_grid_net():
    """Build the small grid network if it doesn't exist yet."""
    if not CFG.exists():
        print("grid.sumocfg not found, building grid network...")
        subprocess.check_call([sys.executable, str(SCRIPTS / "make_grid_net.py")])
    else:
        print("Found existing grid:", CFG)


def build_edge_groups(env, traci, num_groups: int = 4):
    """
    Discover all incoming edges from env._incoming_lanes and partition them
    into num_groups buckets. We don't try to guess real 'N/E/S/W' directions;
    we just want different sets of edges so we can shift traffic emphasis
    over time.

    This makes the script network-agnostic: it will work with your 2x2 grid
    (and likely small variants) without hard-coding edge IDs.
    """
    incoming_edges = []
    for aid in env.agent_ids:
        for lane_id in env._incoming_lanes[aid]:
            edge_id = traci.lane.getEdgeID(lane_id)
            if edge_id not in incoming_edges:
                incoming_edges.append(edge_id)

    incoming_edges.sort()
    if not incoming_edges:
        print("WARNING: No incoming edges discovered from env._incoming_lanes!")

    groups = [[] for _ in range(num_groups)]
    for i, edge_id in enumerate(incoming_edges):
        g = i % num_groups
        groups[g].append(edge_id)

    print("Discovered incoming edges:")
    for i, g in enumerate(groups):
        print(f"  Group {i}: {g}")

    return groups


def make_route_manager(traci):
    """
    Returns a helper function ensure_route(edge_id) that will define a simple
    one-edge route 'r_<edge_id>' on demand and cache its ID.

    We keep routes minimal: each route starts on a specific incoming edge and
    then SUMO decides how vehicles leave the network.
    """
    created_routes = set()

    def ensure_route(edge_id: str) -> str:
        rid = f"r_{edge_id}"
        if rid not in created_routes:
            # Simple route: vehicle enters on this edge and then follows
            # the network connections out.
            traci.route.add(rid, [edge_id])
            created_routes.add(rid)
        return rid

    return ensure_route


def inject_nonstationary_traffic(
    t: int,
    steps: int,
    edge_groups,
    ensure_route,
    traci,
    edge_spawn_counters,
    base_prob: float = 0.02,
    heavy_prob: float = 0.25,
):
    """
    Inject vehicles in a time-varying pattern:

    - Phase 1 (0 to steps/3): heavy on groups 0 and 2
    - Phase 2 (steps/3 to 2*steps/3): heavy on groups 1 and 3
    - Phase 3 (2*steps/3 to steps): heavy on groups 0 and 1

    'base_prob' gives a small background flow, 'heavy_prob' gives a much
    larger flow where we want congestion. This makes static fixed-time control
    badly misaligned with demand, while RL can react to queues.
    """
    phase1_end = steps // 3
    phase2_end = 2 * steps // 3

    if t < phase1_end:
        # "NS-heavy" analogue: emphasize some groups (0, 2)
        heavy_groups = {0, 2}
    elif t < phase2_end:
        # "EW-heavy" analogue: different groups (1, 3)
        heavy_groups = {1, 3}
    else:
        # "Diagonal/mixed": yet another pattern (0, 1)
        heavy_groups = {0, 1}

    for gi, edges in enumerate(edge_groups):
        if not edges:
            continue

        p = heavy_prob if gi in heavy_groups else base_prob

        for edge_id in edges:
            if random.random() < p:
                rid = ensure_route(edge_id)
                key = edge_id
                edge_spawn_counters[key] = edge_spawn_counters.get(key, 0) + 1
                veh_idx = edge_spawn_counters[key]
                veh_id = f"dyn_{edge_id}_{t}_{veh_idx}"

                # Let SUMO choose departLane / departSpeed; we just pick the route.
                traci.vehicle.add(
                    veh_id,
                    rid,
                    depart=str(t),        # depart at current time step
                    departLane="best",
                    departSpeed="max",
                )


def main(steps=3600, seed=42, dynamic_traffic=False):
    ensure_grid_net()

    # Import after SUMO env is presumably set up (SUMO_HOME, PATH)
    from marddpg.env.sumo_multi_tl_env import EnvConfig, SumoMultiTrafficLightEnv
    import traci

    # Make Python's RNG deterministic for reproducible traffic patterns
    random.seed(seed)

    # Where to store outputs on your Mac
    runs_root = ROOT / "mac_runs"
    runs_root.mkdir(exist_ok=True)

    run_dir = runs_root / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir()
    print("Writing outputs to:", run_dir)

    tripinfo_path = run_dir / "tripinfo.xml"
    summary_path = run_dir / "summary.xml"
    metrics_path = run_dir / "metrics.csv"

    cfg = EnvConfig(
        sumo_cfg=str(CFG),
        gui=False,
        max_steps=steps,
        warmup_steps=0,
        min_green=8,
        extend_green=3,
        control_interval=1,
        reward_mode="queue_wait",
        tripinfo_output=str(tripinfo_path),
        summary_output=str(summary_path),
        seed=seed,
    )

    env = SumoMultiTrafficLightEnv(cfg)

    # Start SUMO (no RL actions; just let fixed-time program run)
    obs0, _ = env.reset()

    # Build non-stationary traffic machinery if requested
    if dynamic_traffic:
        print("\n[Dynamic traffic] Enabling non-stationary waves to stress static control.")
        edge_groups = build_edge_groups(env, traci, num_groups=4)
        ensure_route = make_route_manager(traci)
        edge_spawn_counters = {}  # edge_id -> count of injected vehicles
    else:
        edge_groups = None
        ensure_route = None
        edge_spawn_counters = None

    rows = []
    for t in range(steps):
        # 1) Optionally inject dynamic traffic **before** stepping simulation
        if dynamic_traffic and edge_groups:
            inject_nonstationary_traffic(
                t=t,
                steps=steps,
                edge_groups=edge_groups,
                ensure_route=ensure_route,
                traci=traci,
                edge_spawn_counters=edge_spawn_counters,
                base_prob=0.02,
                heavy_prob=0.25,
            )

        # 2) Advance simulation
        traci.simulationStep()

        # 3) Collect metrics (same as your original script)
        total_q = 0.0
        total_w = 0.0
        arrived = len(traci.simulation.getArrivedIDList())

        # sum queue + waiting time over all incoming lanes for all TLS
        for aid in env.agent_ids:
            for ln in env._incoming_lanes[aid]:
                total_q += traci.lane.getLastStepHaltingNumber(ln)
                total_w += traci.lane.getWaitingTime(ln)

        rows.append((t + 1, float(total_q), float(total_w), int(arrived)))

    env.close()

    # Write per-step metrics to CSV
    with metrics_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "sum_queue", "sum_wait", "arrivals"])
        w.writerows(rows)

    print("\n=== SUMO data generated on Mac ===")
    print("Run directory:", run_dir)
    print("tripinfo.xml:", tripinfo_path)
    print("summary.xml:", summary_path)
    print("metrics.csv:", metrics_path)
    if dynamic_traffic:
        print("Dynamic, non-stationary traffic was enabled.")
    else:
        print("Static (original) demand only; no extra injection.")
    print("You can now copy this folder to Ubuntu for RL training.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--steps",
        type=int,
        default=3600,
        help="Number of simulation steps to run",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed to pass to SUMO and Python's RNG",
    )
    ap.add_argument(
        "--dynamic-traffic",
        action="store_true",
        help="Inject time-varying traffic waves to make static control struggle",
    )
    args = ap.parse_args()
    main(steps=args.steps, seed=args.seed, dynamic_traffic=args.dynamic_traffic)
