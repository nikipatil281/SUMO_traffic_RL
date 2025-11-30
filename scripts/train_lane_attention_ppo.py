import os
import random
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from rl_attention.envs.multi_tl_wrapper import MultiTLSumoGymWrapper
from rl_attention.policies.lane_attention_ppo_policy import LaneAttentionPPOPolicy
from rl_attention.ppo.ppo_buffer import PPORolloutBuffer
from rl_attention.regions.dynamic_clustering import cluster_tls_by_embedding


# ----------------------------------------------------------------------
# Utility: seeding
# ----------------------------------------------------------------------
def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# Utility: compute lane-context embeddings from PPO policy
#          (used for region clustering)
# ----------------------------------------------------------------------
def compute_lane_context_embeddings(
    policy: LaneAttentionPPOPolicy,
    obs: Dict[str, np.ndarray],
    incoming_lanes: Dict[str, int],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Rebuild per-TLS lane-context embeddings from the PPO policy's
    lane encoder + self-attention, for use with cluster_tls_by_embedding.

    Returns:
        dict[tid] -> np.ndarray (lane_context vector)
    """
    ctx: Dict[str, np.ndarray] = {}

    for tid in policy.tls_ids:
        obs_vec_np = obs[tid]
        n_lanes = incoming_lanes[tid]

        # Convert obs to tensor
        obs_vec = torch.as_tensor(
            obs_vec_np, dtype=torch.float32, device=device
        )

        # Use the policy's internal decoding helper (already defined)
        decoded = policy._decode_obs_for_tls(obs_vec, n_lanes)  # type: ignore[attr-defined]
        queues = decoded["queues"]        # [L]
        waits = decoded["waits"]          # [L]
        elapsed_norm = decoded["elapsed_norm"]  # scalar

        # Build per-lane features (queue, wait, elapsed_norm)
        lane_feats = []
        for i in range(n_lanes):
            lane_feats.append(
                torch.stack(
                    [
                        queues[i],
                        waits[i],
                        elapsed_norm,
                    ],
                    dim=0,
                )  # [3]
            )

        lane_feats = torch.stack(lane_feats, dim=0)  # [L, 3]
        lane_emb = policy.lane_mlp(lane_feats)       # [L, D]

        # Self-attention over lanes
        lane_emb_batch = lane_emb.unsqueeze(0)       # [1, L, D]
        attn_out, _ = policy.attn(
            lane_emb_batch, lane_emb_batch, lane_emb_batch
        )
        lane_context = attn_out.mean(dim=1).squeeze(0)  # [D]

        ctx[tid] = lane_context.detach().cpu().numpy()

    return ctx


# ----------------------------------------------------------------------
# Utility: recompute region IDs via dynamic clustering
# ----------------------------------------------------------------------
def recompute_region_ids(
    policy: LaneAttentionPPOPolicy,
    obs: Dict[str, np.ndarray],
    incoming_lanes: Dict[str, int],
    num_regions: int,
    device: torch.device,
    random_state: int,
) -> Dict[str, int]:
    """
    Use current lane-context embeddings + KMeans clustering to
    assign each TLS to a region.
    """
    embeddings = compute_lane_context_embeddings(policy, obs, incoming_lanes, device)
    region_ids = cluster_tls_by_embedding(
        embeddings, n_clusters=num_regions, random_state=random_state
    )
    return region_ids


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
def train_lane_attention_ppo() -> None:
    # -------------------- Config --------------------
    SUMO_CFG = "envs/grid4x4_ns_heavy/grid.sumocfg"  # change scenario if desired
    USE_GUI = False
    MAX_STEPS_PER_EPISODE = 1800

    NUM_EPISODES = 10
    NUM_PPO_EPOCHS = 5
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPS = 0.2
    LR = 3e-4
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.5
    NUM_REGIONS = 3
    CLUSTER_EVERY = 1   # recompute regions every N episodes

    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_DIR = "models"
    MODEL_PATH = os.path.join(MODEL_DIR, "lane_attn_region_ppo_policy.pt")
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")
    set_global_seeds(SEED)

    # -------------------- Env + Policy --------------------
    env = MultiTLSumoGymWrapper(
        sumocfg_path=SUMO_CFG,
        gui=USE_GUI,
        max_steps=MAX_STEPS_PER_EPISODE,
        seed=SEED,
    )

    obs = env.reset()
    tls_ids = env.tls_ids

    # incoming_lanes_raw: dict[tid] -> list of incoming lane IDs
    incoming_lanes_raw = env.incoming_lanes
    incoming_lanes = {tid: len(lanes) for tid, lanes in incoming_lanes_raw.items()}

    print(f"Discovered {len(tls_ids)} TLS: {tls_ids}")
    print(f"Incoming lanes per TLS (int): {incoming_lanes}")


    policy = LaneAttentionPPOPolicy(
        tls_ids=tls_ids,
        incoming_lanes_per_tls=incoming_lanes,
        num_regions=NUM_REGIONS,
    ).to(DEVICE)


    optimizer = Adam(policy.parameters(), lr=LR)
    buffer = PPORolloutBuffer(gamma=GAMMA, gae_lambda=GAE_LAMBDA)

    # Initialize default region IDs
    region_ids: Dict[str, int] = {tid: 0 for tid in tls_ids}

    # -------------------- Training loop --------------------
    for ep in range(NUM_EPISODES):
        buffer.clear()
        obs = env.reset()

        # Recompute region IDs every CLUSTER_EVERY episodes
        if (ep % CLUSTER_EVERY) == 0:
            region_ids = recompute_region_ids(
                policy=policy,
                obs=obs,
                incoming_lanes=incoming_lanes,
                num_regions=NUM_REGIONS,
                device=DEVICE,
                random_state=SEED + ep,
            )
            print(f"[Episode {ep:03d}] Region IDs: {region_ids}")

        ep_rewards: List[float] = []
        done = False

        step_count = 0
        while not done and step_count < MAX_STEPS_PER_EPISODE:
            step_count += 1

            # Act with current policy
            with torch.no_grad():
                action_dict, log_prob, value = policy.act(
                    obs, region_ids=region_ids, device=DEVICE
                )

            # env.step returns reward as dict[tid] -> float
            next_obs, rew_dict, done, info = env.step(action_dict)

            # Aggregate reward exactly like your existing trainers:
            # mean over all TLS rewards
            reward_scalar = float(np.mean(list(rew_dict.values())))

            buffer.add(
                obs=obs,
                action=action_dict,
                log_prob=log_prob.item(),
                value=value.item(),
                reward=reward_scalar,
                done=done,
                region_ids=region_ids.copy(),
            )

            ep_rewards.append(reward_scalar)
            obs = next_obs


        # Bootstrap value at end of episode (0 if terminal)
        if done:
            last_value = 0.0
        else:
            with torch.no_grad():
                means, log_std, values_per_tls = policy.forward(
                    obs, region_ids=region_ids, device=DEVICE
                )
                last_value = float(values_per_tls.mean().item())

        buffer.compute_returns_and_advantages(last_value=last_value, device=DEVICE)
        data = buffer.get_training_tensors(device=DEVICE)

        old_log_probs = data["log_probs"]             # [T]
        old_values = data["values"]                   # [T]
        returns = data["returns"]                     # [T]
        advantages = data["advantages"]               # [T]
        batch_obs = data["obs"]                       # list[dict]
        batch_actions = data["actions"]               # list[dict]
        batch_region_ids = data["region_ids"]         # list[dict or None]

        T = old_log_probs.shape[0]

        # -------------------- PPO Update --------------------
        policy.train()
        for epoch in range(NUM_PPO_EPOCHS):
            # For simplicity: use the full trajectory as one batch
            curr_log_probs, values_pred, entropy = policy.evaluate(
                batch_obs,
                batch_region_ids,
                batch_actions,
                device=DEVICE,
            )

            ratios = torch.exp(curr_log_probs - old_log_probs)  # [T]
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values_pred, returns)
            entropy_loss = -entropy.mean()

            loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

        mean_ep_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        print(
            f"[Episode {ep:03d}] steps={step_count:4d}, "
            f"mean_reward={mean_ep_reward:.3f}, "
            f"last_value={last_value:.3f}"
        )

        # Optionally save periodically
        if (ep + 1) % 10 == 0:
            torch.save(policy.state_dict(), MODEL_PATH)
            print(f"Saved intermediate PPO model to {MODEL_PATH}")
        
    try:
        env.close()
    except Exception as e:
        print(f"env.close() raised {e}, ignoring.")

    # Final save
    torch.save(policy.state_dict(), MODEL_PATH)
    print(f"Training finished. Final PPO model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_lane_attention_ppo()
