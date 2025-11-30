import os
import sys
from pprint import pprint

# Ensure project root is on sys.path so "marddpg" can be imported
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)  # .../generating_traffic_mac
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from marddpg.env.sumo_multi_tl_env import EnvConfig, SumoMultiTrafficLightEnv


def main():
    # Absolute path to the small grid SUMO config
    sumocfg_path = os.path.join(ROOT_DIR, "envs", "grid2x2", "grid.sumocfg")
    if not os.path.isfile(sumocfg_path):
        raise FileNotFoundError(f"SUMO config not found at: {sumocfg_path}")

    print(f"Using SUMO config: {sumocfg_path}")

    # IMPORTANT: EnvConfig expects "sumo_cfg", not "sumocfg_path"
    cfg = EnvConfig(
        sumo_cfg=sumocfg_path,
        gui=False,      # headless
        max_steps=200,  # short rollout just to test
        # everything else uses defaults from EnvConfig
    )

    # Create env
    env = SumoMultiTrafficLightEnv(cfg)
    print("Created SumoMultiTrafficLightEnv")

    try:
        # Reset (starts SUMO and does warmup)
        print("Calling env.reset() ...")
        obs = env.reset()
        print("Reset done.")

        # Try to peek into internal TLS IDs
        tls_ids = list(getattr(env, "_tls_ids", []))
        print(f"Discovered {len(tls_ids)} traffic lights: {tls_ids}")

        # Inspect observation structure
        if isinstance(obs, dict) and tls_ids:
            first_id = tls_ids[0]
            first_obs = obs[first_id]
            print(
                f"Obs for first TL '{first_id}': "
                f"shape={getattr(first_obs, 'shape', None)}, value={first_obs}"
            )
        else:
            print("Observation is not a dict or TLS list is empty; obs=")
            pprint(obs)

        print("\nRolling out a few deterministic steps (zero actions)...\n")

        # Run a short rollout with simple deterministic actions (no random policy)
        for step_i in range(10):
            # Here we choose a neutral action: 0.0 for every TLS
            # You can later replace this with your learned policy's actions.
            action_dict = {tid: 0.0 for tid in tls_ids}

            obs, rew, done, info = env.step(action_dict)

            first_id = tls_ids[0] if tls_ids else None
            if first_id is not None:
                a0 = action_dict[first_id]
                r0 = rew[first_id]
                print(
                    f"[step {step_i:02d}] t={info.get('t')} "
                    f"first_tls={first_id} action={a0:.3f} reward={r0:.3f}"
                )
            else:
                print(f"[step {step_i:02d}] t={info.get('t')} reward_dict={rew}")

            if done:
                print("Environment signaled done=True, stopping early.")
                break

        print("\nTest rollout finished successfully.")

    finally:
        # Always close TraCI / SUMO
        print("Closing environment...")
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    main()
