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
        print(f"n_phases: {env.n_phases}")

        if not isinstance(obs, dict) or not tls_ids:
            raise RuntimeError("Unexpected obs from env.reset()")

        # Build attention-based policy
        policy = LaneAttentionGaussianPolicy(
            incoming_lanes=env.incoming_lanes,
            n_phases=env.n_phases,
            embed_dim=32,
            num_heads=2,
            hidden_dim=64,
            init_log_std=-0.5,
        ).to(device)
        print("Created LaneAttentionGaussianPolicy")

        # Compute lane-context embeddings (one step)
        lane_contexts = policy.compute_lane_context(obs, tls_ids, device=device)

        print("\nLane-context embeddings per TLS:")
        for tid in tls_ids:
            emb = lane_contexts[tid]
            print(f"  {tid}: shape={emb.shape}, values={emb}")

    finally:
        print("Closing env...")
        env.close()
        print("Env closed.")


if __name__ == "__main__":
    main()

