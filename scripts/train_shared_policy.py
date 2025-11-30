import os
import sys

import numpy as np
import torch
from torch.optim import Adam

# Ensure project root on sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl_attention.envs.multi_tl_wrapper import MultiTLSumoGymWrapper
from rl_attention.policies.shared_gaussian_policy import SharedGaussianPolicy


def main():
    device = torch.device("cpu")

    # Hyperparameters (keep small / simple for now)
    NUM_EPISODES = 10
    EP_MAX_STEPS = 50
    GAMMA = 0.99
    LR = 1e-3

    env = MultiTLSumoGymWrapper(gui=False, max_steps=EP_MAX_STEPS + 10)
    print("Created MultiTLSumoGymWrapper for training")

    try:
        # Initial reset to get obs dimensions and TLS IDs
        print("Initial env.reset() ...")
        obs = env.reset()
        print("Initial reset done.")

        tls_ids = env.tls_ids
        print(f"TLS IDs: {tls_ids}")

        if not isinstance(obs, dict) or not tls_ids:
            raise RuntimeError("Unexpected observation structure from env.reset()")

        # Compute per-TL obs dims and choose min dim
        dims = {tls_id: obs[tls_id].shape[0] for tls_id in tls_ids}
        print(f"Per-TL obs dims (initial): {dims}")
        min_dim = min(dims.values())
        print(f"Using min obs dim across TLS for policy = {min_dim}")

        # Build policy and optimizer
        policy = SharedGaussianPolicy(obs_dim=min_dim, hidden_dim=64).to(device)
        optimizer = Adam(policy.parameters(), lr=LR)

        print("\nStarting training loop...\n")

        for ep in range(NUM_EPISODES):
            obs = env.reset()
            done = False

            # Storage for REINFORCE
            rewards = []
            step_log_probs = []

            ep_steps = 0

            while not done and ep_steps < EP_MAX_STEPS:
                # Sample actions and log_probs from policy
                action_dict, log_prob_dict = policy.act_and_log_prob(
                    obs, tls_ids, device=device
                )

                # Step the environment
                obs, rew, done, info = env.step(action_dict)

                # Mean reward across TLS
                r_t = float(np.mean(list(rew.values())))
                rewards.append(r_t)

                # Mean log_prob across TLS for this step
                log_probs_tensor = torch.stack(list(log_prob_dict.values()))  # [num_tls]
                mean_log_prob = log_probs_tensor.mean()
                step_log_probs.append(mean_log_prob)

                ep_steps += 1

            # Compute discounted returns
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + GAMMA * G
                returns.insert(0, G)

            returns = torch.tensor(returns, dtype=torch.float32, device=device)

            # Normalize returns (helps training stability)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Stack log_probs
            log_probs = torch.stack(step_log_probs)  # [ep_steps]

            # Policy gradient loss: -E[log pi(a|s) * return]
            loss = -(log_probs * returns).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_return = float(returns.mean().cpu().item())
            raw_return = float(np.mean(rewards)) if rewards else 0.0

            print(
                f"Episode {ep:03d}: steps={ep_steps}, "
                f"mean reward (raw)={raw_return:.3f}, "
                f"mean normalized return={avg_return:.3f}, "
                f"loss={loss.item():.4f}"
            )

        print("\nTraining loop finished.")

    finally:
        print("Closing env...")
        env.close()
        print("Env closed.")


if __name__ == "__main__":
    main()

