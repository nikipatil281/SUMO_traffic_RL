import os
import sys
import time

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

    # Run with GUI on, more steps for a nice demo
    EP_MAX_STEPS = 200

    env = MultiTLSumoGymWrapper(gui=True, max_steps=EP_MAX_STEPS + 10)
    print("Created MultiTLSumoGymWrapper with GUI=True")

    try:
        obs = env.reset()
        print("Reset done.")

        tls_ids = env.tls_ids
        print(f"TLS IDs: {tls_ids}")

        # Build (untrained or trained) attention policy
        policy = LaneAttentionGaussianPolicy(
            incoming_lanes=env.incoming_lanes,
            n_phases=env.n_phases,
            embed_dim=32,
            num_heads=2,
            hidden_dim=64,
            init_log_std=-0.5,
        ).to(device)
        print("Created LaneAttentionGaussianPolicy")

        # If you later save a trained model, you can load it here with:
        # policy.load_state_dict(torch.load("models/lane_attn_policy.pt", map_location=device))

        print("\nStarting GUI rollout. Close the SUMO-GUI window to stop.\n")

        done = False
        step_i = 0
        while not done and step_i < EP_MAX_STEPS:
            # Get actions from policy
            action_dict, _ = policy.act_and_log_prob(obs, tls_ids, device=device)

            obs, rew, done, info = env.step(action_dict)

            # Just print a minimal summary
            mean_rew = sum(rew.values()) / len(rew)
            print(
                f"[step {step_i:03d}] t={info.get('t')} "
                f"mean_reward={mean_rew:.3f}"
            )

            step_i += 1

            # Slow down so you can actually see things
            time.sleep(0.1)

        print("\nGUI rollout finished.")

    finally:
        print("Closing env...")
        env.close()
        print("Env closed.")


if __name__ == "__main__":
    main()

