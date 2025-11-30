import os
import sys

import torch

# Ensure project root on sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl_attention.envs.multi_tl_wrapper import MultiTLSumoGymWrapper
from rl_attention.policies.lane_attention_gaussian_policy import LaneAttentionGaussianPolicy
from rl_attention.regions.dynamic_clustering import cluster_tls_by_embedding


def main():
    device = torch.device("cpu")

    env = MultiTLSumoGymWrapper(gui=False, max_steps=200)
    print("Created MultiTLSumoGymWrapper")

    try:
        print("Calling env.reset() ...")
        obs = env.reset()
        print("Reset done.")

        tls_ids = env.tls_ids
        print(f"TLS IDs: {tls_ids}")
        print(f"incoming_lanes: {{tid: len(lanes) for tid, lanes in env.incoming_lanes.items()}} -> "
              f"{ {tid: len(lanes) for tid, lanes in env.incoming_lanes.items()} }")

        # Build lane-attention policy (untrained)
        policy = LaneAttentionGaussianPolicy(
            incoming_lanes=env.incoming_lanes,
            n_phases=env.n_phases,
            embed_dim=32,
            num_heads=2,
            hidden_dim=64,
            init_log_std=-0.5,
        ).to(device)
        print("Created LaneAttentionGaussianPolicy")

        # Compute lane-context embeddings
        lane_contexts = policy.compute_lane_context(obs, tls_ids, device=device)

        print("\nLane-context embedding shapes:")
        for tid in tls_ids:
            print(f"  {tid}: {lane_contexts[tid].shape}")

        # Cluster into K regions (e.g., 2)
        K = 2
        region_ids = cluster_tls_by_embedding(lane_contexts, n_clusters=K, random_state=0)

        print(f"\nClustering into {K} regions...")
        print("Region assignments (region_id per TLS):")
        for tid in tls_ids:
            print(f"  {tid}: region {region_ids[tid]}")

        # Optional: invert mapping to see members per region
        regions = {}
        for tid, rid in region_ids.items():
            regions.setdefault(rid, []).append(tid)

        print("\nRegions -> member intersections:")
        for rid, members in regions.items():
            print(f"  Region {rid}: {members}")

    finally:
        print("\nClosing env...")
        env.close()
        print("Env closed.")


if __name__ == "__main__":
    main()

