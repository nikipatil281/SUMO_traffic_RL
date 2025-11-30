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
from rl_attention.regions.dynamic_clustering import cluster_tls_by_embedding


def main():
    device = torch.device("cpu")

    # Hyperparameters
    NUM_EPISODES = 20          # increase a bit since now learning more structure
    EP_MAX_STEPS = 60          # slightly longer episodes help
    GAMMA = 0.99
    LR = 1e-3
    NUM_REGIONS = 3            # you can tune this
    CLUSTER_EVERY = 5          # re-cluster every 5 episodes

    env = MultiTLSumoGymWrapper(gui=False, max_steps=EP_MAX_STEPS + 10)
    print("Created MultiTLSumoGymWrapper for REGION training")

    try:
        print("Initial env.reset() ...")
        obs = env.reset()
        print("Initial reset done.")

        tls_ids = env.tls_ids
        print(f"TLS IDs: {tls_ids}")

        # -----------------------------
        # Build REGION-AWARE policy
        # -----------------------------
        policy = LaneAttentionGaussianPolicy(
            incoming_lanes=env.incoming_lanes,
            n_phases=env.n_phases,
            embed_dim=32,
            num_heads=2,
            hidden_dim=64,
            init_log_std=-0.5,
            num_regions=NUM_REGIONS,
            region_embed_dim=8,
        ).to(device)

        optimizer = Adam(policy.parameters(), lr=LR)
        region_ids = {tid: 0 for tid in tls_ids}    # start with everyone in region 0

        print("\nStarting REGION-AWARE training loop...\n")

        for ep in range(NUM_EPISODES):

            # ----------------------------------------
            # (A) Dynamic Re-Clustering every few episodes
            # ----------------------------------------
            if ep % CLUSTER_EVERY == 0:
                print(f"\nRe-clustering at episode {ep}...")

                # To cluster, get lane-context embeddings
                with torch.no_grad():
                    ctx = policy.compute_lane_context(obs, tls_ids, device=device)

                region_ids = cluster_tls_by_embedding(
                    ctx,
                    n_clusters=NUM_REGIONS,
                    random_state=ep,
                )
                print(f"New region assignments: {region_ids}")

            # ----------------------------------------
            # (B) Episode rollout
            # ----------------------------------------
            obs = env.reset()
            done = False
            rewards = []
            log_probs = []

            ep_steps = 0

            while not done and ep_steps < EP_MAX_STEPS:

                # NOTE: now passing region_ids into the policy
                action_dict, log_prob_dict = policy.act_and_log_prob(
                    obs, tls_ids, device=device, region_ids=region_ids
                )

                obs, rew, done, info = env.step(action_dict)

                # reward = mean over TLS
                r_t = float(np.mean(list(rew.values())))
                rewards.append(r_t)

                # log_prob = mean over TLS
                lp = torch.stack(list(log_prob_dict.values())).mean()
                log_probs.append(lp)

                ep_steps += 1

            # ----------------------------------------
            # (C) Compute Returns
            # ----------------------------------------
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + GAMMA * G
                returns.insert(0, G)

            returns = torch.tensor(returns, dtype=torch.float32, device=device)

            # Normalize returns
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            log_probs = torch.stack(log_probs)

            loss = -(log_probs * returns).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ----------------------------------------
            # Print progress
            # ----------------------------------------
            print(
                f"[REGION] Episode {ep:03d}: "
                f"steps={ep_steps}, "
                f"mean reward={np.mean(rewards):.3f}, "
                f"loss={loss.item():.4f}"
            )

        # ----------------------------------------
        # Save region-aware policy
        # ----------------------------------------
        os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
        save_path = os.path.join(PROJECT_ROOT, "models", "lane_attn_region_policy.pt")
        torch.save(policy.state_dict(), save_path)
        print(f"\nSaved REGION-AWARE policy to {save_path}")

    finally:
        print("Closing env...")
        env.close()
        print("Env closed.")


if __name__ == "__main__":
    main()
