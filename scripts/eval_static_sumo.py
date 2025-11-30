import os
import subprocess
import xml.etree.ElementTree as ET

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
SUMO_CFG = os.path.join(PROJECT_ROOT, "envs/grid2x2/grid.sumocfg")
OUT_FILE = os.path.join(PROJECT_ROOT, "envs/grid2x2/tripinfo_static.xml")

def run_static():
    # Remove old file
    if os.path.exists(OUT_FILE):
        os.remove(OUT_FILE)

    cmd = [
        "sumo",
        "-c", SUMO_CFG,
        "--quit-on-end",
    ]

    print("Running static SUMO...")
    subprocess.run(cmd, check=True)
    print("SUMO static run completed.")

def parse_tripinfo():
    tree = ET.parse(OUT_FILE)
    root = tree.getroot()

    total_wait = 0.0
    total_time_loss = 0.0
    count = 0

    for trip in root.findall("tripinfo"):
        wait = float(trip.get("waitingTime"))
        loss = float(trip.get("timeLoss"))
        total_wait += wait
        total_time_loss += loss
        count += 1

    return {
        "avg_wait_time": total_wait / max(1,count),
        "avg_delay": total_time_loss / max(1,count),
        "vehicles_completed": count
    }

def main():
    run_static()
    stats = parse_tripinfo()
    print("\nStatic Baseline Metrics:")
    for k,v in stats.items():
        print(f"{k}: {v:.3f}")

if __name__ == "__main__":
    main()

