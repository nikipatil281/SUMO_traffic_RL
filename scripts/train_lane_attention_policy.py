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
from rl_attention.policies.lane_attention_gaussian_policy import LaneAttentionGaussianPolicy


def main():
    device = torch.device("cpu")

    # Hyperparameters
    NUM_EPISODES = 10
    EP_MAX_STEPS = 50
    GAMMA = 0.99
    LR = 1e-3

    env = MultiTLSumoGymWrapper(gui=False, max_steps=EP_MAX_STEPS + 10)
    print("Created MultiTLSumoGymWrapper for attention training")

    try:
        # Initial reset to get metadata
        print("Initial env.reset() ...")
        obs = env.reset()
        print("Initial reset done.")

        tls_ids = env.tls_ids
        print(f"TLS IDs: {tls_ids}")
        print(f"incoming_lanes: {{tid: len(lanes) for tid, lanes in env.incoming_lanes.items()}} -> "
              f"{ {tid: len(lanes) for tid, lanes in env.incoming_lanes.items()} }")
        print(f"n_phases: {env.n_phases}")

        if not isinstance(obs, dict) or not tls_ids:
            raise RuntimeError("Unexpected observation structure from env.reset()")

        # Build attention-based Gaussian policy
        policy = LaneAttentionGaussianPolicy(
            incoming_lanes=env.incoming_lanes,
            n_phases=env.n_phases,
            embed_dim=32,
            num_heads=2,
            hidden_dim=64,
            init_log_std=-0.5,
        ).to(device)
        optimizer = Adam(policy.parameters(), lr=LR)

        print("\nStarting attention-based training loop...\n")

        for ep in range(NUM_EPISODES):
            obs = env.reset()
            done = False

            rewards = []
            step_log_probs = []

            ep_steps = 0

            while not done and ep_steps < EP_MAX_STEPS:
                # Policy produces actions + log_probs for each TLS
                action_dict, log_prob_dict = policy.act_and_log_prob(
                    obs, tls_ids, device=device
                )

                obs, rew, done, info = env.step(action_dict)

                # Mean reward across TLS
                r_t = float(np.mean(list(rew.values())))
                rewards.append(r_t)

                # Mean log_prob across TLS
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

            # Normalize returns
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            log_probs = torch.stack(step_log_probs)  # [ep_steps]

            # REINFORCE loss
            loss = -(log_probs * returns).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_return = float(returns.mean().cpu().item())
            raw_return = float(np.mean(rewards)) if rewards else 0.0

            print(
                f"[ATTN] Episode {ep:03d}: steps={ep_steps}, "
                f"mean reward (raw)={raw_return:.3f}, "
                f"mean normalized return={avg_return:.3f}, "
                f"loss={loss.item():.4f}"
            )

        print("\nAttention-based training loop finished.")

        # 🔽 Save once at the end
        os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
        weights_path = os.path.join(PROJECT_ROOT, "models", "lane_attn_policy.pt")
        torch.save(policy.state_dict(), weights_path)
        print(f"Saved lane-attention policy weights to {weights_path}")

    finally:
        print("Closing env...")
        env.close()
        print("Env closed.")


if __name__ == "__main__":
    main()
