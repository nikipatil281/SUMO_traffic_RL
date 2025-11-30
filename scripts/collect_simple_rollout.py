import os
import sys

import numpy as np
import torch

# Ensure project root is on sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl_attention.envs.multi_tl_wrapper import MultiTLSumoGymWrapper
from rl_attention.policies.simple_shared_policy import SimpleSharedMLPPolicy


def main():
    device = torch.device("cpu")

    # How many episodes / steps to collect
    NUM_EPISODES = 3
    EP_MAX_STEPS = 50

    env = MultiTLSumoGymWrapper(gui=False, max_steps=EP_MAX_STEPS + 10)
    print("Created MultiTLSumoGymWrapper for rollout collection")

    try:
        # One initial reset to inspect obs dims and build the policy
        print("Initial env.reset() ...")
        obs = env.reset()
        print("Initial reset done.")

        tls_ids = env.tls_ids
        print(f"TLS IDs: {tls_ids}")

        if not isinstance(obs, dict) or not tls_ids:
            raise RuntimeError("Unexpected observation structure from env.reset()")

        # Compute per-TL obs dims and choose min dim (as in test_simple_policy_inference.py)
        dims = {tls_id: obs[tls_id].shape[0] for tls_id in tls_ids}
        print(f"Per-TL obs dims (initial): {dims}")

        min_dim = min(dims.values())
        print(f"Using min obs dim across TLS = {min_dim}")

        # Build shared policy
        policy = SimpleSharedMLPPolicy(obs_dim=min_dim, hidden_dim=64).to(device)
        print("Created SimpleSharedMLPPolicy for rollout collection")

        print("\nStarting rollout collection...\n")

        for ep in range(NUM_EPISODES):
            obs = env.reset()
            done = False
            ep_return = 0.0
            ep_steps = 0

            while not done and ep_steps < EP_MAX_STEPS:
                # Use policy to compute actions
                action_dict = policy.act(obs, tls_ids, device=device)

                # Step env
                obs, rew, done, info = env.step(action_dict)

                # Aggregate reward across all TLs (mean)
                mean_reward = float(np.mean(list(rew.values())))
                ep_return += mean_reward
                ep_steps += 1

            avg_return = ep_return / max(ep_steps, 1)
            print(
                f"Episode {ep} finished: steps={ep_steps}, "
                f"mean-reward-per-step={avg_return:.3f}"
            )

        print("\nRollout collection finished successfully.")

    finally:
        print("Closing env...")
        env.close()
        print("Env closed.")


if __name__ == "__main__":
    main()

