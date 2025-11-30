#!/usr/bin/env python3
import os
import sys
import time
import csv
import subprocess
from pathlib import Path

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

def main(steps=3600, seed=42):
    ensure_grid_net()

    # Import after SUMO env is presumably set up (SUMO_HOME, PATH)
    from marddpg.env.sumo_multi_tl_env import EnvConfig, SumoMultiTrafficLightEnv
    import traci

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

    rows = []
    for t in range(steps):
        traci.simulationStep()

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
    print("You can now copy this folder to Ubuntu for RL training.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3600,
                    help="Number of simulation steps to run")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed to pass to SUMO")
    args = ap.parse_args()
    main(steps=args.steps, seed=args.seed)