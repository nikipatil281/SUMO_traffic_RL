import os
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

# Ensure project root on sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl_attention.envs.multi_tl_wrapper import MultiTLSumoGymWrapper
from rl_attention.policies.lane_attention_ddpg import (
    LaneAttentionDDPGActor,
    LaneAttentionDDPGCritic,
)
from rl_attention.regions.dynamic_clustering import cluster_tls_by_embedding


# -------------------- Utilities --------------------
def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_lane_context_embeddings(
    actor: LaneAttentionDDPGActor,
    obs: Dict[str, np.ndarray],
    incoming_lanes: Dict[str, int],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Build per-TLS lane-context embeddings using the actor's lane encoder
    (for region clustering).
    """
    ctx: Dict[str, np.ndarray] = {}

    for tid in actor.tls_ids:
        obs_vec_np = obs[tid]
        n_lanes = incoming_lanes[tid]
        obs_vec = torch.as_tensor(obs_vec_np, dtype=torch.float32, device=device)

        decoded = actor._decode_obs_for_tls(obs_vec, n_lanes)  # type: ignore[attr-defined]
        queues = decoded["queues"]
        waits = decoded["waits"]
        elapsed_norm = decoded["elapsed_norm"]

        lane_feats = []
        for i in range(n_lanes):
            lane_feats.append(
                torch.stack(
                    [queues[i], waits[i], elapsed_norm],
                    dim=0,
                )
            )
        lane_feats = torch.stack(lane_feats, dim=0)
        lane_emb = actor.lane_mlp(lane_feats)

        lane_emb_batch = lane_emb.unsqueeze(0)
        attn_out, _ = actor.attn(
            lane_emb_batch, lane_emb_batch, lane_emb_batch
        )
        lane_context = attn_out.mean(dim=1).squeeze(0)

        ctx[tid] = lane_context.detach().cpu().numpy()

    return ctx


def recompute_region_ids(
    actor: LaneAttentionDDPGActor,
    obs: Dict[str, np.ndarray],
    incoming_lanes: Dict[str, int],
    num_regions: int,
    device: torch.device,
    random_state: int,
) -> Dict[str, int]:
    embeddings = compute_lane_context_embeddings(actor, obs, incoming_lanes, device)
    region_ids = cluster_tls_by_embedding(
        embeddings, n_clusters=num_regions, random_state=random_state
    )
    return region_ids


# -------------------- Replay Buffer --------------------
class DDPGReplayBuffer:
    def __init__(self, max_size: int):
        self.max_size = int(max_size)
        self.storage: deque = deque(maxlen=self.max_size)

    def add(
        self,
        obs: Dict[str, np.ndarray],
        region_ids: Dict[str, int],
        action: Dict[str, float],
        reward: float,
        next_obs: Dict[str, np.ndarray],
        next_region_ids: Dict[str, int],
        done: bool,
    ) -> None:
        self.storage.append(
            (obs, region_ids, action, reward, next_obs, next_region_ids, done)
        )

    def size(self) -> int:
        return len(self.storage)

    def sample(
        self,
        batch_size: int,
    ) -> List[Tuple[Dict[str, np.ndarray], Dict[str, int], Dict[str, float], float,
                   Dict[str, np.ndarray], Dict[str, int], bool]]:
        batch_size = min(batch_size, len(self.storage))
        idxs = np.random.choice(len(self.storage), size=batch_size, replace=False)
        batch = [self.storage[i] for i in idxs]
        return batch


# -------------------- Training Loop --------------------
def train_lane_attention_ddpg() -> None:
    # Config
    SUMO_CFG = "envs/grid4x4_ns_heavy/grid.sumocfg"  # change if needed
    USE_GUI = False
    MAX_STEPS_PER_EPISODE = 150

    NUM_EPISODES = 5
    GAMMA = 0.99
    TAU = 0.005

    ACTOR_LR = 1e-4
    CRITIC_LR = 1e-3

    REPLAY_SIZE = 50_000
    BATCH_SIZE = 64
    WARMUP_STEPS = 200  # collect some transitions before updates
    UPDATES_PER_STEP = 1

    NUM_REGIONS = 3
    CLUSTER_EVERY = 1

    NOISE_STD_START = 0.3
    NOISE_STD_END = 0.05

    SEED = 7
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_DIR = "models"
    ACTOR_PATH = os.path.join(MODEL_DIR, "lane_attn_region_ddpg_actor.pt")
    CRITIC_PATH = os.path.join(MODEL_DIR, "lane_attn_region_ddpg_critic.pt")
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")
    set_global_seeds(SEED)

    # Env
    env = MultiTLSumoGymWrapper(
        sumocfg_path=SUMO_CFG,
        gui=USE_GUI,
        max_steps=MAX_STEPS_PER_EPISODE,
        seed=SEED,
    )

    obs = env.reset()
    tls_ids = env.tls_ids

    incoming_lanes_raw = env.incoming_lanes  # dict[tid] -> list of lane IDs
    incoming_lanes = {tid: len(lanes) for tid, lanes in incoming_lanes_raw.items()}

    print(f"TLS IDs: {tls_ids}")
    print(f"Incoming lanes per TLS: {incoming_lanes}")

    # Actor & Critic
    actor = LaneAttentionDDPGActor(
        tls_ids=tls_ids,
        incoming_lanes_per_tls=incoming_lanes,
        num_regions=NUM_REGIONS,
        max_action=1.0,
    ).to(DEVICE)

    critic = LaneAttentionDDPGCritic(
        tls_ids=tls_ids,
        incoming_lanes_per_tls=incoming_lanes,
        num_regions=NUM_REGIONS,
    ).to(DEVICE)

    actor_target = LaneAttentionDDPGActor(
        tls_ids=tls_ids,
        incoming_lanes_per_tls=incoming_lanes,
        num_regions=NUM_REGIONS,
        max_action=1.0,
    ).to(DEVICE)
    critic_target = LaneAttentionDDPGCritic(
        tls_ids=tls_ids,
        incoming_lanes_per_tls=incoming_lanes,
        num_regions=NUM_REGIONS,
    ).to(DEVICE)

    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = Adam(actor.parameters(), lr=ACTOR_LR)
    critic_optimizer = Adam(critic.parameters(), lr=CRITIC_LR)

    replay_buffer = DDPGReplayBuffer(max_size=REPLAY_SIZE)

    total_steps = 0

    # Training episodes
    for ep in range(NUM_EPISODES):
        obs = env.reset()

        # region IDs (dynamic clustering each episode)
        if (ep % CLUSTER_EVERY) == 0:
            region_ids = recompute_region_ids(
                actor=actor,
                obs=obs,
                incoming_lanes=incoming_lanes,
                num_regions=NUM_REGIONS,
                device=DEVICE,
                random_state=SEED + ep,
            )
        else:
            # keep previous (if you want)
            region_ids = {tid: 0 for tid in tls_ids}

        print(f"[Ep {ep:03d}] Region IDs: {region_ids}")

        done = False
        ep_rewards: List[float] = []
        step_count = 0

        while not done and step_count < MAX_STEPS_PER_EPISODE:
            step_count += 1
            total_steps += 1

            # Exploration noise decay
            frac = min(1.0, total_steps / (NUM_EPISODES * MAX_STEPS_PER_EPISODE))
            noise_std = NOISE_STD_START + frac * (NOISE_STD_END - NOISE_STD_START)

            with torch.no_grad():
                action_dict = actor.act_with_noise(
                    obs=obs,
                    region_ids=region_ids,
                    noise_std=noise_std,
                    device=DEVICE,
                )

            next_obs, rew_dict, done, info = env.step(action_dict)
            print(f"[Ep {ep:03d}] raw step={step_count}, done={done}, t={info.get('t')}")
            reward_scalar = float(np.mean(list(rew_dict.values())))
            ep_rewards.append(reward_scalar)
            # DEBUG: show progress every 200 steps
            if step_count % 200 == 0:
                print(
                    f"[Ep {ep:03d}] step={step_count}, "
                    f"reward={reward_scalar:.3f}, "
                    f"buffer={replay_buffer.size()}"
                )

            # For simplicity keep region_ids constant within episode
            next_region_ids = region_ids.copy()

            # Store in replay buffer
            replay_buffer.add(
                obs=obs,
                region_ids=region_ids.copy(),
                action=action_dict,
                reward=reward_scalar,
                next_obs=next_obs,
                next_region_ids=next_region_ids,
                done=done,
            )

            obs = next_obs

            # DDPG updates
            if replay_buffer.size() >= WARMUP_STEPS:
                for _ in range(UPDATES_PER_STEP):
                    batch = replay_buffer.sample(BATCH_SIZE)
                    critic_loss, actor_loss = ddpg_update(
                        actor,
                        critic,
                        actor_target,
                        critic_target,
                        actor_optimizer,
                        critic_optimizer,
                        batch,
                        tls_ids,
                        gamma=GAMMA,
                        tau=TAU,
                        device=DEVICE,
                    )

        mean_ep_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        print(
            f"[Ep {ep:03d}] steps={step_count}, "
            f"mean_reward={mean_ep_reward:.3f}, "
            f"buffer_size={replay_buffer.size()}"
        )

        if (ep + 1) % 10 == 0:
            torch.save(actor.state_dict(), ACTOR_PATH)
            torch.save(critic.state_dict(), CRITIC_PATH)
            print(f"Saved intermediate DDPG models to {ACTOR_PATH}, {CRITIC_PATH}")

    # Final save
    torch.save(actor.state_dict(), ACTOR_PATH)
    torch.save(critic.state_dict(), CRITIC_PATH)
    print(f"Training finished. Final DDPG models saved to {ACTOR_PATH}, {CRITIC_PATH}")

    try:
        env.close()
    except Exception as e:
        print(f"env.close() raised {e}, ignoring.")


# -------------------- DDPG Update --------------------
def ddpg_update(
    actor: LaneAttentionDDPGActor,
    critic: LaneAttentionDDPGCritic,
    actor_target: LaneAttentionDDPGActor,
    critic_target: LaneAttentionDDPGCritic,
    actor_optimizer: Adam,
    critic_optimizer: Adam,
    batch,
    tls_ids: List[str],
    gamma: float,
    tau: float,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Perform one DDPG update from a minibatch.
    """
    critic_losses = []
    actor_losses = []

    for (
        obs,
        region_ids,
        action_dict,
        reward,
        next_obs,
        next_region_ids,
        done,
    ) in batch:
        # Build action tensor in tls_ids order
        action_vec = torch.tensor(
            [action_dict[tid] for tid in tls_ids],
            dtype=torch.float32,
            device=device,
        )

        # Current Q(s, a)
        q_val = critic(
            obs=obs,
            actions=action_vec,
            region_ids=region_ids,
            device=device,
        )

        # Target Q(s', a')
        with torch.no_grad():
            next_action_vec = actor_target.forward(
                obs=next_obs,
                region_ids=next_region_ids,
                device=device,
            )
            q_next = critic_target(
                obs=next_obs,
                actions=next_action_vec,
                region_ids=next_region_ids,
                device=device,
            )
            target_q = reward + gamma * (0.0 if done else 1.0) * q_next.item()
            target_q = torch.tensor(target_q, dtype=torch.float32, device=device)

        # Critic loss
        critic_loss = F.mse_loss(q_val, target_q)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        critic_losses.append(critic_loss.item())

        # Actor loss: -Q(s, a(s))
        actor_optimizer.zero_grad()
        pred_action_vec = actor.forward(
            obs=obs,
            region_ids=region_ids,
            device=device,
        )
        actor_q = critic(
            obs=obs,
            actions=pred_action_vec,
            region_ids=region_ids,
            device=device,
        )
        actor_loss = -actor_q
        actor_loss.backward()
        actor_optimizer.step()

        actor_losses.append(actor_loss.item())

        # Soft update targets
        soft_update(actor, actor_target, tau)
        soft_update(critic, critic_target, tau)

    return float(np.mean(critic_losses)), float(np.mean(actor_losses))


def soft_update(source_net: torch.nn.Module, target_net: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for param, target_param in zip(source_net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data
            )


if __name__ == "__main__":
    train_lane_attention_ddpg()
