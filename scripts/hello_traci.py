# scripts/hello_traci.py
import os, sys, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NETDIR = ROOT / "envs" / "grid2x2"
CFG = NETDIR / "grid.sumocfg"

def ensure_files():
    if not CFG.exists():
        # build the tiny grid
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "make_grid_net.py")])

def main():
    ensure_files()
    # import traci *after* SUMO tools are on PYTHONPATH (set in ~/.zshrc)
    import traci
    sumo_cmd = ["sumo", "-c", str(CFG), "--no-step-log", "true"]
    print("$", " ".join(sumo_cmd))
    traci.start(sumo_cmd)
    try:
        tls_ids = traci.trafficlight.getIDList()
        print("TLS:", tls_ids)
        for step in range(50):
            if step % 10 == 0:
                # print phase and vehicle counts near first tls
                if tls_ids:
                    tid = tls_ids[0]
                    phase = traci.trafficlight.getPhase(tid)
                    state = traci.trafficlight.getRedYellowGreenState(tid)
                    print(f"t={step:02d} tls[{tid}] phase={phase} state={state}")
            traci.simulationStep()
        print("Simulation OK.")
    finally:
        traci.close()

if __name__ == "__main__":
    main()
