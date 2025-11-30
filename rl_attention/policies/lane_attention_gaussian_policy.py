import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LaneAttentionGaussianPolicy(nn.Module):
    """
    Lane-level self-attention + Gaussian policy over continuous actions,
    with optional region embeddings.

    Assumes per-TL observation structure:
      - let L = number of incoming lanes for this TL
      - obs[tid] is a 1D array of length 2*L + 4 + 1:
          [ queue[0..L-1],
            waiting[0..L-1],
            phase_onehot[0..3],
            elapsed_norm ]

    This matches the shapes you saw: 11 dims for L=3, 13 dims for L=4, etc.

    Region support (for Step B):
      - You can optionally pass a dict region_ids: tid -> int
      - We embed region_id via nn.Embedding and append it to the feature vector
        before the Gaussian head.
      - If region_ids is None or tid not in dict, region index defaults to 0.
    """

    def __init__(
        self,
        incoming_lanes: Dict[str, List[str]],
        n_phases: Dict[str, int],
        embed_dim: int = 32,
        num_heads: int = 2,
        hidden_dim: int = 64,
        init_log_std: float = -0.5,
        num_regions: int = 4,
        region_embed_dim: int = 8,
    ):
        super().__init__()

        self.incoming_lanes = incoming_lanes
        self.n_phases = n_phases

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Lane encoder: (queue, waiting, elapsed_norm) -> lane embedding
        self.lane_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

        # Self-attention over lanes
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Global (phase one-hot + elapsed) embedding
        # phase_onehot(4) + elapsed(1) = 5 dims
        self.phase_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Region embedding
        self.num_regions = num_regions
        self.region_embed_dim = region_embed_dim
        self.region_embedding = nn.Embedding(num_regions, region_embed_dim)

        # Final head: [lane_context, phase_emb, region_emb] -> mean action
        policy_input_dim = embed_dim + hidden_dim + region_embed_dim
        self.mlp_head = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Log std parameter (shared across TLs)
        self.log_std = nn.Parameter(torch.ones(1) * init_log_std)

    # -------------------------------------------------------------------------
    #  Helpers to parse the observation vector for a single TL
    # -------------------------------------------------------------------------
    def _split_obs_for_tls(
        self,
        obs_vec: torch.Tensor,
        num_lanes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given obs_vec (1D tensor) and number of incoming lanes L,
        returns:
          queue:  shape [L]
          waiting: shape [L]
          phase_onehot: shape [4]
          elapsed_norm: shape [] scalar
        """
        L = num_lanes
        assert obs_vec.ndim == 1, "obs_vec must be 1D"

        queue = obs_vec[0:L]
        waiting = obs_vec[L : 2 * L]
        phase_onehot = obs_vec[2 * L : 2 * L + 4]
        elapsed_norm = obs_vec[2 * L + 4]

        return queue, waiting, phase_onehot, elapsed_norm

    # -------------------------------------------------------------------------
    #  Core lane-context computation in torch (with gradients)
    # -------------------------------------------------------------------------
    def _lane_context_torch(
        self,
        obs: Dict[str, np.ndarray],
        tls_ids: List[str],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute lane-context embeddings (one per TL) as torch tensors with grads.
        """
        ctx: Dict[str, torch.Tensor] = {}

        for tid in tls_ids:
            vec = obs[tid]
            num_lanes = len(self.incoming_lanes[tid])

            vec_t = torch.as_tensor(vec, dtype=torch.float32, device=device)
            queue, waiting, phase_onehot, elapsed_norm = self._split_obs_for_tls(
                vec_t, num_lanes
            )

            # Build lane features: [L, 3] = (queue, waiting, elapsed_norm per lane)
            elapsed_rep = elapsed_norm.expand(num_lanes)
            lane_feats = torch.stack([queue, waiting, elapsed_rep], dim=-1)  # [L, 3]

            lane_emb = self.lane_mlp(lane_feats)  # [L, embed_dim]

            # Self-attention over lanes
            lane_emb_batch = lane_emb.unsqueeze(0)  # [1, L, embed_dim]
            attn_out, _ = self.attn(
                lane_emb_batch,
                lane_emb_batch,
                lane_emb_batch,
                need_weights=False,
            )  # [1, L, embed_dim]

            # Average lanes to get context
            ctx_tid = attn_out.mean(dim=1).squeeze(0)  # [embed_dim]
            ctx[tid] = ctx_tid

        return ctx

    # -------------------------------------------------------------------------
    #  Public: compute_lane_context for inspection / debugging (no grads)
    # -------------------------------------------------------------------------
    def compute_lane_context(
        self,
        obs: Dict[str, np.ndarray],
        tls_ids: List[str],
        device: torch.device,
    ) -> Dict[str, np.ndarray]:
        """
        Same as _lane_context_torch, but returns numpy arrays and does not
        track gradients. This is what your test_lane_embeddings.py uses.
        """
        with torch.no_grad():
            ctx_torch = self._lane_context_torch(obs, tls_ids, device)

        return {tid: v.detach().cpu().numpy() for tid, v in ctx_torch.items()}

    # -------------------------------------------------------------------------
    #  Gaussian policy: act and log_prob
    # -------------------------------------------------------------------------
    def act_and_log_prob(
        self,
        obs: Dict[str, np.ndarray],
        tls_ids: List[str],
        device: torch.device,
        region_ids: Optional[Dict[str, int]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """
        Compute actions and log_probs for each TLS.

        Args:
          obs: dict[tid] -> np.ndarray
          tls_ids: list of TL IDs in a fixed order
          device: torch.device
          region_ids: optional dict[tid] -> int region index

        Returns:
          action_dict: dict[tid] -> float (scalar action)
          log_prob_dict: dict[tid] -> torch.Tensor scalar (log prob)
        """
        self.train()  # ensure we're in train mode by default

        lane_ctx = self._lane_context_torch(obs, tls_ids, device)

        feats = []
        phase_feats = []

        # Build per-TL feature vectors
        for tid in tls_ids:
            vec = obs[tid]
            num_lanes = len(self.incoming_lanes[tid])
            vec_t = torch.as_tensor(vec, dtype=torch.float32, device=device)

            _, _, phase_onehot, elapsed_norm = self._split_obs_for_tls(vec_t, num_lanes)

            # phase + elapsed -> global embedding
            phase_elapsed = torch.cat(
                [phase_onehot, elapsed_norm.unsqueeze(0)], dim=-1
            )  # [5]
            phase_emb = self.phase_mlp(phase_elapsed)  # [hidden_dim]

            # region embedding
            rid = 0
            if region_ids is not None and tid in region_ids:
                rid = int(region_ids[tid])
            rid = max(0, min(rid, self.num_regions - 1))
            rid_t = torch.tensor(rid, dtype=torch.long, device=device)
            region_emb = self.region_embedding(rid_t)  # [region_embed_dim]

            # full feature: [lane_context, phase_emb, region_emb]
            ctx_tid = lane_ctx[tid]  # [embed_dim]
            feat_tid = torch.cat([ctx_tid, phase_emb, region_emb], dim=-1)
            feats.append(feat_tid)

        feats_tensor = torch.stack(feats, dim=0)  # [num_tls, D]
        means = self.mlp_head(feats_tensor).squeeze(-1)  # [num_tls]

        # Gaussian policy
        log_std = self.log_std.expand_as(means)
        std = log_std.exp()

        # Sample actions
        noise = torch.randn_like(means)
        actions = means + std * noise  # unconstrained
        # Optional: clamp to [-1, 1] since your env expects that range
        actions_clamped = torch.clamp(actions, -1.0, 1.0)

        # Log probability of the (unclamped) sample under N(means, std^2)
        # We compute log_prob of the *sampled* actions (before clamp).
        var = std.pow(2)
        log_prob = -0.5 * (((actions - means) ** 2) / var + 2 * log_std + math.log(2 * math.pi))
        # log_prob shape: [num_tls]

        action_dict: Dict[str, float] = {}
        log_prob_dict: Dict[str, torch.Tensor] = {}

        for i, tid in enumerate(tls_ids):
            action_dict[tid] = float(actions_clamped[i].detach().cpu().item())
            log_prob_dict[tid] = log_prob[i]

        return action_dict, log_prob_dict

    # -------------------------------------------------------------------------
    #  Convenience wrapper if some scripts only need actions
    # -------------------------------------------------------------------------
    def act(
        self,
        obs: Dict[str, np.ndarray],
        tls_ids: List[str],
        device: torch.device,
        region_ids: Optional[Dict[str, int]] = None,
    ) -> Dict[str, float]:
        """
        Return only actions, ignoring log_probs.
        """
        actions, _ = self.act_and_log_prob(obs, tls_ids, device, region_ids=region_ids)
        return actions
