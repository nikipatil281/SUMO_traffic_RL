import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class LaneAttentionPPOPolicy(nn.Module):
    """
    PPO-ready lane-attention policy with region support and continuous actions.

    Assumed per-TLS observation structure (per intersection):
        obs[tid] = [
            queue_lane_0, ..., queue_lane_{L-1},
            wait_lane_0,  ..., wait_lane_{L-1},
            phase_onehot_0, phase_onehot_1, phase_onehot_2, phase_onehot_3,
            elapsed_norm
        ]

    Where:
        - L = number of incoming lanes for this TLS
        - 'phase_onehot_*' is a 4-dim one-hot vector for the current phase
        - 'elapsed_norm' is a scalar in [0, 1] (time in current phase / max_phase_duration)

    Architecture (per TLS):
        lanes -> lane_mlp -> lane embeddings -> MultiheadAttention over lanes -> lane_context
        phase features -> phase_mlp -> phase_context
        region_id -> nn.Embedding -> region_emb
        fused = [lane_context, phase_context, region_emb] -> fusion_mlp -> h
        action_mean = linear(h)
        value       = linear(h)

    PPO usage:
        - act(...)     -> returns action_dict, log_prob (scalar), value (scalar)
        - evaluate(...) -> used by PPO update to recompute log_probs, values, entropy
    """

    def __init__(
        self,
        tls_ids: Sequence[str],
        incoming_lanes_per_tls: Dict[str, int],
        num_regions: int,
        lane_embed_dim: int = 32,
        lane_hidden_dim: int = 32,
        phase_hidden_dim: int = 32,
        fusion_hidden_dim: int = 64,
        num_attn_heads: int = 4,
        region_embed_dim: int = 8,
        action_std_init: float = 0.3,
    ) -> None:
        super().__init__()

        self.tls_ids: List[str] = list(tls_ids)
        self.incoming_lanes_per_tls = dict(incoming_lanes_per_tls)
        self.num_regions = num_regions

        # ----- Lane encoder -----
        # Each lane feature = (queue, wait, elapsed_norm) -> R^3
        self.lane_mlp = nn.Sequential(
            nn.Linear(3, lane_hidden_dim),
            nn.ReLU(),
            nn.Linear(lane_hidden_dim, lane_embed_dim),
            nn.ReLU(),
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=lane_embed_dim,
            num_heads=num_attn_heads,
            batch_first=True,  # input shape: [batch=L_lanes, dim]
        )

        # ----- Phase encoder -----
        # phase_onehot (4) + elapsed_norm (1) -> 5-dim
        self.phase_mlp = nn.Sequential(
            nn.Linear(5, phase_hidden_dim),
            nn.ReLU(),
            nn.Linear(phase_hidden_dim, phase_hidden_dim),
            nn.ReLU(),
        )

        # ----- Region embedding -----
        self.region_embedding = nn.Embedding(num_regions, region_embed_dim)

        # ----- Fusion MLP -----
        fusion_input_dim = lane_embed_dim + phase_hidden_dim + region_embed_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
        )

        # ----- Action and value heads -----
        self.action_mean_head = nn.Linear(fusion_hidden_dim, 1)
        self.value_head = nn.Linear(fusion_hidden_dim, 1)

        # Shared log-std parameter (continuous actions)
        action_std = float(action_std_init)
        self.log_std = nn.Parameter(torch.ones(1) * math.log(action_std))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _decode_obs_for_tls(
        self,
        obs_vec: torch.Tensor,
        n_lanes: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Given a 1D tensor obs_vec for a single intersection and its number of
        incoming lanes, decode into:
            queues: [L]
            waits: [L]
            phase_onehot: [4]
            elapsed_norm: scalar tensor
        """
        L = n_lanes
        # Shape assumptions:
        # [0:L]     -> queues
        # [L:2L]    -> waits
        # [2L:2L+4] -> phase_onehot
        # [-1]      -> elapsed_norm
        queues = obs_vec[0:L]
        waits = obs_vec[L : 2 * L]
        phase_onehot = obs_vec[2 * L : 2 * L + 4]
        elapsed_norm = obs_vec[-1]
        return {
            "queues": queues,
            "waits": waits,
            "phase_onehot": phase_onehot,
            "elapsed_norm": elapsed_norm,
        }

    def _build_tls_embeddings(
        self,
        obs: Dict[str, np.ndarray],
        region_ids: Optional[Dict[str, int]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        For a dict of obs[tid] -> np.array, build fused embeddings h[tid] for all
        tls_ids, returned as a tensor of shape [N_tls, fusion_hidden_dim].
        """
        fused_list = []

        for tid in self.tls_ids:
            obs_vec_np = obs[tid]
            n_lanes = self.incoming_lanes_per_tls[tid]

            # Convert to tensor
            obs_vec = torch.as_tensor(
                obs_vec_np, dtype=torch.float32, device=device
            )

            decoded = self._decode_obs_for_tls(obs_vec, n_lanes)
            queues = decoded["queues"]
            waits = decoded["waits"]
            phase_onehot = decoded["phase_onehot"]
            elapsed_norm = decoded["elapsed_norm"]

            # ----- Lane embeddings -----
            # Build per-lane features: (queue_i, wait_i, elapsed_norm)
            lane_feats = []
            for i in range(n_lanes):
                lane_feat = torch.stack(
                    [
                        queues[i],
                        waits[i],
                        elapsed_norm,  # same scalar for all lanes
                    ],
                    dim=0,
                )  # shape [3]
                lane_feats.append(lane_feat)

            lane_feats = torch.stack(lane_feats, dim=0)  # [L, 3]
            lane_emb = self.lane_mlp(lane_feats)  # [L, lane_embed_dim]

            # Multihead attention over lanes (self-attention)
            # attn expects [batch_size, seq_len, embed_dim], here batch_size=1
            lane_emb_batch = lane_emb.unsqueeze(0)  # [1, L, D]
            attn_out, _ = self.attn(
                lane_emb_batch, lane_emb_batch, lane_emb_batch
            )
            lane_context = attn_out.mean(dim=1).squeeze(0)  # [D]

            # ----- Phase embedding -----
            # concat phase_onehot (4) + elapsed_norm (1)
            phase_feat = torch.cat(
                [phase_onehot, elapsed_norm.unsqueeze(0)], dim=0
            )  # [5]
            phase_emb = self.phase_mlp(phase_feat)  # [phase_hidden_dim]

            # ----- Region embedding -----
            if region_ids is not None and tid in region_ids:
                region_id = region_ids[tid]
            else:
                # default region 0 if not provided
                region_id = 0

            region_id_tensor = torch.tensor(
                [region_id], dtype=torch.long, device=device
            )
            region_emb = self.region_embedding(region_id_tensor).squeeze(0)

            # ----- Fuse -----
            fused = torch.cat([lane_context, phase_emb, region_emb], dim=-1)
            h = self.fusion_mlp(fused)  # [fusion_hidden_dim]
            fused_list.append(h)

        fused_tensor = torch.stack(fused_list, dim=0)  # [N_tls, fusion_hidden_dim]
        return fused_tensor

    # ------------------------------------------------------------------
    # Public API: forward / act / evaluate
    # ------------------------------------------------------------------
    def forward(
        self,
        obs: Dict[str, np.ndarray],
        region_ids: Optional[Dict[str, int]] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Forward pass:
            obs       : dict[tid] -> np.array
            region_ids: dict[tid] -> int region index (optional)
        Returns:
            means  : tensor [N_tls]
            log_std: tensor [N_tls]
            values : tensor [N_tls]
        """
        if device is None:
            device = next(self.parameters()).device

        h = self._build_tls_embeddings(obs, region_ids, device)
        means = self.action_mean_head(h).squeeze(-1)  # [N_tls]
        values = self.value_head(h).squeeze(-1)       # [N_tls]

        # Expand shared log_std to match shape of means
        log_std = self.log_std.expand_as(means)       # [N_tls]
        return means, log_std, values

    @torch.no_grad()
    def act(
        self,
        obs: Dict[str, np.ndarray],
        region_ids: Optional[Dict[str, int]] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Sample continuous actions for all TLS, and return:
            action_dict: dict[tid] -> float
            log_prob   : scalar tensor (sum over all TLS)
            value      : scalar tensor (mean value over TLS)
        """
        if device is None:
            device = next(self.parameters()).device

        means, log_std, values_per_tls = self.forward(
            obs, region_ids=region_ids, device=device
        )
        std = torch.exp(log_std)
        dist = Normal(means, std)

        actions = dist.sample()  # [N_tls]
        log_probs = dist.log_prob(actions)  # [N_tls]

        # Aggregate per-TLS into a single scalar for PPO
        total_log_prob = log_probs.sum()       # scalar
        state_value = values_per_tls.mean()    # scalar

        # Convert back to dict for environment
        action_dict: Dict[str, float] = {
            tid: actions[i].item() for i, tid in enumerate(self.tls_ids)
        }

        return action_dict, total_log_prob, state_value

    def evaluate(
        self,
        batch_obs: List[Dict[str, np.ndarray]],
        batch_region_ids: Optional[List[Dict[str, int]]],
        batch_actions: List[Dict[str, float]],
        device: Optional[torch.device] = None,
    ):
        """
        Evaluate a batch of (obs, actions) pairs for PPO.

        Args:
            batch_obs       : list over time, each item is dict[tid] -> np.array
            batch_region_ids: list over time, each item is dict[tid] -> int (or None)
            batch_actions   : list over time, each item is dict[tid] -> float

        Returns:
            log_probs: tensor [T]  (scalar log_prob per time step)
            values   : tensor [T]  (scalar value per time step)
            entropy  : tensor [T]  (scalar entropy per time step)
        """
        if device is None:
            device = next(self.parameters()).device

        T = len(batch_obs)
        assert len(batch_actions) == T
        if batch_region_ids is not None:
            assert len(batch_region_ids) == T
        else:
            batch_region_ids = [None] * T  # type: ignore

        log_probs_list = []
        values_list = []
        entropy_list = []

        for t in range(T):
            obs_t = batch_obs[t]
            region_ids_t = batch_region_ids[t]
            actions_t = batch_actions[t]

            means, log_std, values_per_tls = self.forward(
                obs_t, region_ids=region_ids_t, device=device
            )
            std = torch.exp(log_std)
            dist = Normal(means, std)

            # Build action tensor in tls_ids order
            a_vec = torch.tensor(
                [actions_t[tid] for tid in self.tls_ids],
                dtype=torch.float32,
                device=device,
            )

            log_prob_t = dist.log_prob(a_vec).sum()     # scalar
            entropy_t = dist.entropy().sum()            # scalar
            value_t = values_per_tls.mean()             # scalar

            log_probs_list.append(log_prob_t)
            values_list.append(value_t)
            entropy_list.append(entropy_t)

        log_probs = torch.stack(log_probs_list, dim=0)  # [T]
        values = torch.stack(values_list, dim=0)        # [T]
        entropy = torch.stack(entropy_list, dim=0)      # [T]

        return log_probs, values, entropy
