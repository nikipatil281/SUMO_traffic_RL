import os
import sys

import numpy as np
import torch

# Make sure project root is on sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl_attention.envs.multi_tl_wrapper import MultiTLSumoGymWrapper
from rl_attention.policies.simple_shared_policy import SimpleSharedMLPPolicy


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

        if not isinstance(obs, dict) or not tls_ids:
            raise RuntimeError("Unexpected observation structure from env.reset()")

        # Check per-TL obs dims
        dims = {tls_id: obs[tls_id].shape[0] for tls_id in tls_ids}
        print(f"Per-TL obs dims: {dims}")

        min_dim = min(dims.values())
        print(f"Using min obs dim across TLS = {min_dim}")

        first_id = tls_ids[0]
        first_obs = obs[first_id]
        if not isinstance(first_obs, np.ndarray):
            raise RuntimeError(f"Expected numpy array for obs, got {type(first_obs)}")

        # Build the shared policy using min_dim
        policy = SimpleSharedMLPPolicy(obs_dim=min_dim, hidden_dim=64).to(device)
        print("Created SimpleSharedMLPPolicy")

        print("\nRolling out a few steps using the policy...\n")

        for step_i in range(10):
            # Use the policy to get actions for all TLS
            action_dict = policy.act(obs, tls_ids, device=device)

            # Step the env
            obs, rew, done, info = env.step(action_dict)

            a0 = action_dict[first_id]
            r0 = rew[first_id]
            print(
                f"[step {step_i:02d}] t={info.get('t')} "
                f"first_tls={first_id} action={a0:.3f} reward={r0:.3f}"
            )

            if done:
                print("Env signaled done=True, stopping early.")
                break

        print("\nPolicy-based rollout finished successfully.")

    finally:
        print("Closing env...")
        env.close()
        print("Env closed.")


if __name__ == "__main__":
    main()
