import os
import sys
from typing import Dict, List

import torch
import torch.nn as nn

# Ensure project root is on sys.path (so we can import the wrapper if needed)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RL_ROOT = os.path.dirname(THIS_DIR)      # .../rl_attention
PROJECT_ROOT = os.path.dirname(RL_ROOT)  # .../generating_traffic_mac
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class SimpleSharedMLPPolicy(nn.Module):
    """
    Simple shared policy:
      - one MLP shared across all traffic lights
      - input: per-TL observation vector (shape [obs_dim])
      - output: scalar action in [-1, 1] (via tanh)

    NOTE: For now we handle different obs lengths across TLs by
    truncating all obs vectors to the minimum length across TLS.
    This is just for bootstrapping the pipeline.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # ensure action in [-1, 1]
        )

    def forward(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """
        obs_batch: tensor of shape [num_tls, obs_dim]
        returns: tensor of shape [num_tls, 1] with actions in [-1, 1]
        """
        return self.net(obs_batch)

    @torch.no_grad()
    def act(
        self,
        obs_dict: Dict[str, "np.ndarray"],
        tls_ids: List[str],
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, float]:
        """
        Convenience method:
          - stacks obs_dict[tls_id] into a tensor
          - truncates obs to the minimum length across TLS
          - runs forward pass
          - returns a dict[tls_id] -> float action
        """
        import numpy as np  # local import to avoid hard dependency if not needed

        # Compute lengths per TLS
        lengths = [obs_dict[tls_id].shape[0] for tls_id in tls_ids]
        min_dim = min(lengths)

        if min_dim < self.obs_dim:
            # For now we expect obs_dim (used to build the network)
            # to equal the minimum length across TLS.
            # If not, we truncate further to self.obs_dim.
            min_dim = self.obs_dim

        # Truncate each obs vector to min_dim
        obs_list = [obs_dict[tls_id][:min_dim] for tls_id in tls_ids]
        obs_array = np.stack(obs_list, axis=0)  # [num_tls, obs_dim]

        obs_tensor = torch.from_numpy(obs_array).float().to(device)
        actions_tensor = self.forward(obs_tensor)  # [num_tls, 1]

        actions_tensor = actions_tensor.squeeze(-1).cpu().numpy()  # [num_tls]
        return {tls_id: float(a) for tls_id, a in zip(tls_ids, actions_tensor)}
