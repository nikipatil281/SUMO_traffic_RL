import os
import sys
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl_attention.envs.multi_tl_wrapper import MultiTLSumoGymWrapper
from rl_attention.policies.lane_attention_gaussian_policy import LaneAttentionGaussianPolicy


def main():
    device = torch.device("cpu")

    EPISODES = 5
    STEPS = 100

    results = []

    for ep in range(EPISODES):
        env = MultiTLSumoGymWrapper(gui=False, max_steps=STEPS+10)
        obs = env.reset()

        tls_ids = env.tls_ids

        # Policy (load your trained weights if available)
        policy = LaneAttentionGaussianPolicy(
            incoming_lanes=env.incoming_lanes,
            n_phases=env.n_phases,
        ).to(device)

        # Load if you saved after training:
        # policy.load_state_dict(torch.load("models/attn_policy.pt", map_location=device))

        ep_wait = 0
        ep_halt = 0

        for t in range(STEPS):
            action, _ = policy.act_and_log_prob(obs, tls_ids, device=device)
            obs, rew, done, info = env.step(action)

            # Queue / waiting metrics
            m = env.get_metrics()
            avg_wait = np.mean([v["total_waiting"] for v in m.values()])
            avg_halt = np.mean([v["total_halting"] for v in m.values()])

            ep_wait += avg_wait
            ep_halt += avg_halt

            if done:
                break

        env.close()
        results.append((ep_wait/STEPS, ep_halt/STEPS))

    waits, halts = zip(*results)
    print("\nLane Attention RL:")
    print(f"Avg waiting time: {np.mean(waits):.3f}")
    print(f"Avg queue length (halting vehicles): {np.mean(halts):.3f}")


if __name__ == "__main__":
    main()

