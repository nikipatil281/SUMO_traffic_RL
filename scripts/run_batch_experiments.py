import os
import sys
import csv

import numpy as np
import torch
import traci
from sumolib import checkBinary

# Ensure project root on sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl_attention.envs.multi_tl_wrapper import MultiTLSumoGymWrapper
from rl_attention.policies.lane_attention_gaussian_policy import LaneAttentionGaussianPolicy
from rl_attention.regions.dynamic_clustering import cluster_tls_by_embedding

SUMO_CFG = os.path.join(PROJECT_ROOT, "envs", "grid4x4_rush_hour", "grid.sumocfg")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, "batch_results.csv")


# ---------- 1. Static SUMO (no RL) ----------
def run_static_sumo(steps: int, seed: int):
    """
    Run SUMO with fixed-time signals, step via TraCI,
    and collect avg waiting time and avg queue length.
    """
    sumoBinary = checkBinary("sumo")
    cmd = [
        sumoBinary,
        "-c", SUMO_CFG,
        "--no-step-log", "true",
        "--waiting-time-memory", "3600",
        "--seed", str(seed),
    ]
    traci.start(cmd)

    lane_ids = traci.lane.getIDList()
    waits = []
    halts = []
    vehicles_completed = 0

    try:
        for t in range(steps):
            traci.simulationStep()

            vehicles_completed += traci.simulation.getArrivedNumber()

            step_waits = []
            step_halts = []
            for lane in lane_ids:
                h = traci.lane.getLastStepHaltingNumber(lane)
                w = traci.lane.getWaitingTime(lane)
                step_halts.append(h)
                step_waits.append(w)

            waits.append(float(np.mean(step_waits)))
            halts.append(float(np.mean(step_halts)))
    finally:
        traci.close()

    avg_wait = float(np.mean(waits))
    avg_queue = float(np.mean(halts))
    sim_time = len(waits)
    return avg_wait, avg_queue, vehicles_completed, sim_time


# ---------- 3. Lane-attention RL (single-critic) ----------
def run_lane_attn_agent(steps: int, seed: int):
    device = torch.device("cpu")

    env = MultiTLSumoGymWrapper(gui=False, max_steps=steps + 10, seed=seed)
    obs = env.reset()
    tls_ids = env.tls_ids

    policy = LaneAttentionGaussianPolicy(
        incoming_lanes=env.incoming_lanes,
        n_phases=env.n_phases,
        embed_dim=32,
        num_heads=2,
        hidden_dim=64,
        init_log_std=-0.5,
    ).to(device)

    # Load flat (non-regional) trained weights
    weights_path = os.path.join(PROJECT_ROOT, "models", "lane_attn_policy.pt")
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        policy.load_state_dict(state_dict)
        print(f"Loaded lane-attn policy weights from {weights_path}")
    else:
        print("WARNING: Trained lane_attn_policy.pt not found, using untrained policy.")

    waits = []
    halts = []
    vehicles_completed = 0

    try:
        for t in range(steps):
            action_dict, _ = policy.act_and_log_prob(obs, tls_ids, device=device)
            obs, rew, done, info = env.step(action_dict)

            m = env.get_metrics()
            avg_wait = float(np.mean([v["total_waiting"] for v in m.values()]))
            avg_halt = float(np.mean([v["total_halting"] for v in m.values()]))

            vehicles_completed += traci.simulation.getArrivedNumber()

            waits.append(avg_wait)
            halts.append(avg_halt)

            if done:
                break
    finally:
        env.close()

    avg_wait = float(np.mean(waits))
    avg_queue = float(np.mean(halts))
    sim_time = len(waits)
    return avg_wait, avg_queue, vehicles_completed, sim_time


# ---------- 4. Lane-attention REGION-AWARE RL ----------
def run_lane_attn_region_agent(steps: int, seed: int, num_regions: int = 3):
    device = torch.device("cpu")

    env = MultiTLSumoGymWrapper(gui=False, max_steps=steps + 10, seed=seed)
    obs = env.reset()
    tls_ids = env.tls_ids

    policy = LaneAttentionGaussianPolicy(
        incoming_lanes=env.incoming_lanes,
        n_phases=env.n_phases,
        embed_dim=32,
        num_heads=2,
        hidden_dim=64,
        init_log_std=-0.5,
        num_regions=num_regions,
        region_embed_dim=8,
    ).to(device)

    # Load region-aware trained weights
    weights_path = os.path.join(PROJECT_ROOT, "models", "lane_attn_region_policy.pt")
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        policy.load_state_dict(state_dict)
        print(f"Loaded REGION-AWARE policy weights from {weights_path}")
    else:
        print("WARNING: lane_attn_region_policy.pt not found, using untrained region policy.")

    # Compute region_ids once from initial lane-context
    with torch.no_grad():
        ctx = policy.compute_lane_context(obs, tls_ids, device=device)
    region_ids = cluster_tls_by_embedding(
        ctx,
        n_clusters=num_regions,
        random_state=seed,
    )
    print(f"Region IDs for evaluation: {region_ids}")

    waits = []
    halts = []
    vehicles_completed = 0

    try:
        for t in range(steps):
            action_dict, _ = policy.act_and_log_prob(
                obs,
                tls_ids,
                device=device,
                region_ids=region_ids,
            )
            obs, rew, done, info = env.step(action_dict)

            m = env.get_metrics()
            avg_wait = float(np.mean([v["total_waiting"] for v in m.values()]))
            avg_halt = float(np.mean([v["total_halting"] for v in m.values()]))

            vehicles_completed += traci.simulation.getArrivedNumber()

            waits.append(avg_wait)
            halts.append(avg_halt)

            if done:
                break
    finally:
        env.close()

    avg_wait = float(np.mean(waits))
    avg_queue = float(np.mean(halts))
    sim_time = len(waits)
    return avg_wait, avg_queue, vehicles_completed, sim_time


# ---------- 5. Batch runner ----------
def main():
    # You can tweak this list of scenarios
    scenarios = [
        {"name": "short_seed1", "steps": 600, "seed": 1},
        {"name": "medium_seed2", "steps": 1200, "seed": 2},
        {"name": "long_seed3", "steps": 1800, "seed": 3},
    ]

    fieldnames = [
        "scenario",
        "controller",
        "steps",
        "seed",
        "avg_wait_time",
        "avg_queue_length",
        "vehicles_completed",
        "sim_time",
    ]

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sc in scenarios:
            name = sc["name"]
            steps = sc["steps"]
            seed = sc["seed"]

            print(f"\n=== Scenario: {name} (steps={steps}, seed={seed}) ===")

            # Static SUMO
            print("  -> Static SUMO")
            sw, sq, sveh, stime = run_static_sumo(steps, seed)
            writer.writerow({
                "scenario": name,
                "controller": "static",
                "steps": steps,
                "seed": seed,
                "avg_wait_time": sw,
                "avg_queue_length": sq,
                "vehicles_completed": sveh,
                "sim_time": stime,
            })

            # Lane-attention (flat)
            print("  -> Lane-attention RL (single-critic)")
            lw, lq, lveh, ltime = run_lane_attn_agent(steps, seed)
            writer.writerow({
                "scenario": name,
                "controller": "lane_attn",
                "steps": steps,
                "seed": seed,
                "avg_wait_time": lw,
                "avg_queue_length": lq,
                "vehicles_completed": lveh,
                "sim_time": ltime,
            })

            # Lane-attention REGION-AWARE
            print("  -> Lane-attention REGION RL")
            rw2, rq2, rveh2, rtime2 = run_lane_attn_region_agent(steps, seed)
            writer.writerow({
                "scenario": name,
                "controller": "lane_attn_region",
                "steps": steps,
                "seed": seed,
                "avg_wait_time": rw2,
                "avg_queue_length": rq2,
                "vehicles_completed": rveh2,
                "sim_time": rtime2,
            })

    print(f"\nAll results written to {CSV_PATH}")


if __name__ == "__main__":
    main()
