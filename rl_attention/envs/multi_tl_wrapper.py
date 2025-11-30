import os
import sys
from typing import Dict, Any, Optional

import numpy as np

# Ensure project root on sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RL_ROOT = os.path.dirname(THIS_DIR)      # .../rl_attention
PROJECT_ROOT = os.path.dirname(RL_ROOT)  # .../generating_traffic_mac
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from marddpg.env.sumo_multi_tl_env import EnvConfig, SumoMultiTrafficLightEnv


DEFAULT_CFG_PATH = os.path.join(PROJECT_ROOT, "envs", "grid2x2", "grid.sumocfg")


class MultiTLSumoGymWrapper:
    """
    Correct wrapper for your SUMO environment.
    Uses internal fields:
      - _tls_ids
      - _incoming_lanes
      - _n_phases
    These ARE the correct names shown in all earlier tests.
    """

    def __init__(
        self,
        gui: bool = False,
        max_steps: int = 200,
        sumocfg_path: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        if sumocfg_path is None:
            sumocfg_path = DEFAULT_CFG_PATH

        cfg = EnvConfig(
            sumo_cfg=sumocfg_path,
            gui=gui,
            max_steps=max_steps,
            seed=seed,
        )

        self._env = SumoMultiTrafficLightEnv(cfg)

        # Will be populated after reset()
        self.tls_ids = []
        self.incoming_lanes: Dict[str, list] = {}
        self.n_phases: Dict[str, int] = {}

    def reset(self):
        obs, info = self._env.reset()

        # FIX: use internal attributes that exist
        if not self.tls_ids:
            self.tls_ids = list(self._env._tls_ids)
            self.incoming_lanes = dict(self._env._incoming_lanes)
            self.n_phases = dict(self._env._n_phases)

        return obs

    def step(self, actions: Dict[str, float]):
        return self._env.step(actions)

    def close(self):
        self._env.close()


    def get_metrics(self):
        """Forward the metrics dictionary from the env."""
        return self._env.get_metrics()
