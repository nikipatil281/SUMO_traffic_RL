import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Ensure project root on sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RL_ROOT = os.path.dirname(THIS_DIR)      # .../rl_attention
PROJECT_ROOT = os.path.dirname(RL_ROOT)  # .../generating_traffic_mac
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class LaneAttentionPolicy(nn.Module):
    """
    Intra-intersection attention policy.

    For each TLS tid, the obs vector from SumoMultiTrafficLightEnv has layout:

        [ q_0, ..., q_{L-1},
          w_0, ..., w_{L-1},
          phase_onehot(0..n_ph-1),
          elapsed_norm ]

    where:
        L           = number of incoming lanes for that TLS
        n_ph        = number of phases for that TLS
        q_i         = halting vehicles on lane i
        w_i * 0.1   = scaled waiting time on lane i
        elapsed_norm= (time since phase start) / (min_green + 20)

    We:
      1. reconstruct lane-level features [q_i, w_i, elapsed_norm]
      2. embed them with a small MLP
      3. run self-attention over lanes (per TLS)
      4. pool attended lane embeddings to one vector per TLS
      5. concatenate with phase onehot (padded) and elapsed_norm
      6. output scalar action in [-1, 1] for each TLS.
    """

    def __init__(
        self,
        incoming_lanes: Dict[str, List[str]],
        n_phases: Dict[str, int],
        embed_dim: int = 32,
        num_heads: int = 2,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # Keep metadata
        self.tls_ids: List[str] = sorted(incoming_lanes.keys())
        self.incoming_lanes = incoming_lanes
        self.n_phases = n_phases

        # Compute max lanes and max phases for padding
        self.max_lanes = max(len(v) for v in incoming_lanes.values())
        self.max_phases = max(n_phases[tid] for tid in self.tls_ids)

        # Lane feature: [queue, waiting_time, elapsed_norm] -> embed_dim
        self.lane_feat_dim = 3
        self.lane_encoder = nn.Sequential(
            nn.Linear(self.lane_feat_dim, embed_dim),
            nn.ReLU(),
        )

        # Multi-head self-attention across lanes (batch_first for [B, T, C])
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Final head: lane-context + padded phase onehot + elapsed -> 1 action
        # lane_context: embed_dim
        # phase_onehot_padded: max_phases
        # elapsed: 1
        final_in_dim = embed_dim + self.max_phases + 1

        self.head = nn.Sequential(
            nn.Linear(final_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # keep action in [-1, 1]
        )

    def _parse_obs_tls(
        self, tid: str, obs_vec: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Given TLS id and its obs vector, recover:
          lane_feats: [L, 3]  (queue, waiting_time, elapsed_norm)
          phase_onehot: [max_phases] padded
          elapsed_norm: float
        """
        L = len(self.incoming_lanes[tid])
        n_ph = self.n_phases[tid]

        # Extract segments
        q = obs_vec[0:L]
        w = obs_vec[L:2 * L]
        phase_onehot = obs_vec[2 * L : 2 * L + n_ph]
        elapsed_norm = float(obs_vec[2 * L + n_ph])

        # Lane features: stack [q_i, w_i, elapsed_norm]
        elapsed_vec = np.full_like(q, elapsed_norm, dtype=np.float32)
        lane_feats = np.stack([q, w, elapsed_vec], axis=-1)  # [L, 3]

        # Pad phase_onehot to max_phases
        phase_padded = np.zeros(self.max_phases, dtype=np.float32)
        phase_padded[:n_ph] = phase_onehot

        return lane_feats, phase_padded, elapsed_norm

    @torch.no_grad()
    def act(
        self,
        obs_dict: Dict[str, np.ndarray],
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, float]:
        """
        Inference: maps obs_dict to action_dict (deterministic).
        """
        self.eval()

        # Build lane tensors for all TLS in a fixed order
        lane_feat_batches = []
        lane_masks = []
        phase_batches = []
        elapsed_batches = []

        for tid in self.tls_ids:
            obs_vec = obs_dict[tid]
            lane_feats, phase_padded, elapsed_norm = self._parse_obs_tls(tid, obs_vec)

            L = lane_feats.shape[0]
            # Pad lanes to max_lanes
            pad_len = self.max_lanes - L
            if pad_len > 0:
                pad_block = np.zeros((pad_len, self.lane_feat_dim), dtype=np.float32)
                lane_feats_padded = np.concatenate([lane_feats, pad_block], axis=0)
                mask = np.array([1] * L + [0] * pad_len, dtype=np.float32)
            else:
                lane_feats_padded = lane_feats
                mask = np.ones(self.max_lanes, dtype=np.float32)

            lane_feat_batches.append(lane_feats_padded)
            lane_masks.append(mask)
            phase_batches.append(phase_padded)
            elapsed_batches.append([elapsed_norm])

        lane_feat_array = np.stack(lane_feat_batches, axis=0)   # [B, max_lanes, 3]
        mask_array = np.stack(lane_masks, axis=0)               # [B, max_lanes]
        phase_array = np.stack(phase_batches, axis=0)           # [B, max_phases]
        elapsed_array = np.array(elapsed_batches, dtype=np.float32)  # [B, 1]

        lane_feat = torch.from_numpy(lane_feat_array).float().to(device)
        mask = torch.from_numpy(mask_array).bool().to(device)  # True for valid, False for padded
        phase = torch.from_numpy(phase_array).float().to(device)
        elapsed = torch.from_numpy(elapsed_array).float().to(device)

        # Encode lanes
        lane_emb = self.lane_encoder(lane_feat)  # [B, T, C]

        # Multi-head self-attention across lanes
        # key_padding_mask: True for positions that should be ignored
        key_padding_mask = ~mask  # invert: padded=True
        attn_out, _ = self.attn(
            lane_emb, lane_emb, lane_emb,
            key_padding_mask=key_padding_mask
        )  # [B, T, C]

        # Pool attended lane embeddings using mask (mean over valid lanes)
        mask_float = mask.float().unsqueeze(-1)  # [B, T, 1]
        summed = (attn_out * mask_float).sum(dim=1)          # [B, C]
        counts = mask_float.sum(dim=1).clamp(min=1.0)        # [B, 1]
        lane_context = summed / counts                       # [B, C]

        # Build final input: [lane_context, phase_onehot_padded, elapsed]
        x = torch.cat([lane_context, phase, elapsed], dim=-1)  # [B, C + max_phases + 1]

        actions = self.head(x).squeeze(-1)  # [B]

        # Map back to dict[tls_id] -> float
        actions_cpu = actions.cpu().numpy()
        return {tid: float(a) for tid, a in zip(self.tls_ids, actions_cpu)}

