import os
import sys

import numpy as np
import torch

# Ensure project root on sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl_attention.envs.multi_tl_wrapper import MultiTLSumoGymWrapper
from rl_attention.policies.lane_attention_gaussian_policy import LaneAttentionGaussianPolicy
from rl_attention.graph.neighbor_attention import IntersectionNeighborMixer
from scripts.build_tls_graph import build_tls_graph  # reuse your helper


def main():
    device = torch.device("cpu")

    # Paths
    net_path = os.path.join(PROJECT_ROOT, "envs", "grid2x2", "net.net.xml")

    # Build neighbor graph once
    neighbor_graph = build_tls_graph(net_path)
    print("Neighbor graph:")
    print(neighbor_graph)

    env = MultiTLSumoGymWrapper(gui=False, max_steps=200)
    print("Created MultiTLSumoGymWrapper")

    try:
        print("Calling env.reset() ...")
        obs = env.reset()
        print("Reset done.")

        tls_ids = env.tls_ids
        print(f"TLS IDs: {tls_ids}")

        # Build lane-attention policy
        policy = LaneAttentionGaussianPolicy(
            incoming_lanes=env.incoming_lanes,
            n_phases=env.n_phases,
            embed_dim=32,
            num_heads=2,
            hidden_dim=64,
            init_log_std=-0.5,
        ).to(device)
        print("Created LaneAttentionGaussianPolicy")

        # Compute lane-context embeddings at t=0
        lane_contexts = policy.compute_lane_context(obs, tls_ids, device=device)

        print("\nOriginal lane-context embeddings per TLS:")
        for tid in tls_ids:
            emb = lane_contexts[tid]
            print(f"  {tid}: shape={emb.shape}, first3={emb[:3]}")

        # Build neighbor mixer
        mixer = IntersectionNeighborMixer(embed_dim=32, hidden_dim=64).to(device)
        print("\nCreated IntersectionNeighborMixer")

        # Apply neighbor mixing
        mixed_contexts = mixer(lane_contexts, neighbor_graph, tls_ids, device=device)

        print("\nMixed (neighbor-aware) embeddings per TLS:")
        for tid in tls_ids:
            emb = mixed_contexts[tid]
            print(f"  {tid}: shape={emb.shape}, first3={emb[:3]}")

    finally:
        print("\nClosing env...")
        env.close()
        print("Env closed.")


if __name__ == "__main__":
    main()

