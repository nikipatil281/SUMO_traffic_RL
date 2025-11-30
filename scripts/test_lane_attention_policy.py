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
from rl_attention.policies.lane_attention_policy import LaneAttentionPolicy


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
            raise RuntimeError("Unexpected observation structure from env.reset()")

        # Build attention policy using env metadata
        policy = LaneAttentionPolicy(
            incoming_lanes=env.incoming_lanes,
            n_phases=env.n_phases,
            embed_dim=32,
            num_heads=2,
            hidden_dim=64,
        ).to(device)
        print("Created LaneAttentionPolicy")

        # Just to verify obs parsing works for one TLS
        first_id = tls_ids[0]
        first_obs = obs[first_id]
        print(f"Obs[{first_id}] shape = {first_obs.shape}, value = {first_obs}")

        print("\nRolling out a few steps using the attention policy...\n")

        for step_i in range(10):
            action_dict = policy.act(obs, device=device)

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
        
        # ---- Save trained policy ----
        models_dir = os.path.join(PROJECT_ROOT, "models")
        os.makedirs(models_dir, exist_ok=True)
        save_path = os.path.join(models_dir, "lane_attn_policy.pt")
        torch.save(policy.state_dict(), save_path)
        print(f"\nSaved trained LaneAttentionGaussianPolicy to: {save_path}")


        print("\nAttention policy rollout finished successfully.")

    finally:
        print("Closing env...")
        env.close()
        print("Env closed.")


if __name__ == "__main__":
    main()

