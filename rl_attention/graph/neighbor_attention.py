import os
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

# Ensure project root on sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RL_ROOT = os.path.dirname(THIS_DIR)      # .../rl_attention
PROJECT_ROOT = os.path.dirname(RL_ROOT)  # .../generating_traffic_mac
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class IntersectionNeighborMixer(nn.Module):
    """
    Simple intersection-to-intersection "attention-like" mixer.

    For each intersection i:
      - self embedding: e_i
      - neighbor embeddings: {e_j | j in N(i)}

    It computes:
      neighbor_mean_i = average_j e_j  (or e_i if no neighbors)
      concat_i = [e_i, neighbor_mean_i]
      new_e_i  = MLP(concat_i)

    This already captures "who are my critical neighbors?" in a learnable way.
    Later, you can replace neighbor_mean with actual attention weights if needed.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        emb_dict: Dict[str, np.ndarray],
        neighbor_graph: Dict[str, List[str]],
        tls_ids: List[str],
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, np.ndarray]:
        """
        Args:
          emb_dict: dict[tid] -> np.ndarray of shape [D]
          neighbor_graph: dict[tid] -> List[neighbor_tid]
          tls_ids: list of tids defining order

        Returns:
          new_emb_dict: dict[tid] -> np.ndarray of shape [D]
        """
        # Stack embeddings in fixed order
        emb_list = [emb_dict[tid] for tid in tls_ids]
        emb_array = np.stack(emb_list, axis=0)  # [B, D]
        emb = torch.from_numpy(emb_array).float().to(device)  # [B, D]

        B, D = emb.shape
        assert D == self.embed_dim, f"Expected embed_dim={self.embed_dim}, got {D}"

        # Build neighbor_mean tensor [B, D]
        neighbor_means = []
        for i, tid in enumerate(tls_ids):
            neighbors = neighbor_graph.get(tid, [])
            if not neighbors:
                # No neighbors: just use self
                neighbor_means.append(emb[i].unsqueeze(0))
            else:
                # Average neighbor embeddings
                n_embs = [emb[tls_ids.index(n_tid)] for n_tid in neighbors]
                n_stack = torch.stack(n_embs, dim=0)  # [deg, D]
                n_mean = n_stack.mean(dim=0, keepdim=True)  # [1, D]
                neighbor_means.append(n_mean)

        neighbor_means_tensor = torch.cat(neighbor_means, dim=0)  # [B, D]

        # Concatenate self and neighbor_mean
        concat = torch.cat([emb, neighbor_means_tensor], dim=-1)  # [B, 2D]

        # Apply MLP
        new_emb = self.mlp(concat)  # [B, D]

        # Convert back to dict
        new_emb_np = new_emb.detach().cpu().numpy()
        return {tid: new_emb_np[i] for i, tid in enumerate(tls_ids)}
