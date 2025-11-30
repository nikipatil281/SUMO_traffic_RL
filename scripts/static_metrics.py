#!/usr/bin/env python3
import csv
import argparse
from pathlib import Path


def compute_stats_from_metrics(metrics_path: Path):
    """
    Reads metrics.csv (t, sum_queue, sum_wait, arrivals)
    and returns (steps, avg_queue_length, avg_wait_time, vehicles_completed, sim_time).
    """
    steps = 0
    total_queue = 0.0
    total_wait = 0.0
    vehicles_completed = 0

    with metrics_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps += 1
            total_queue += float(row["sum_queue"])
            total_wait += float(row["sum_wait"])
            vehicles_completed += int(row["arrivals"])

    if steps == 0:
        raise ValueError(f"No rows found in {metrics_path}")

    avg_queue_length = total_queue / steps
    if vehicles_completed > 0:
        avg_wait_time = total_wait / vehicles_completed
    else:
        avg_wait_time = 0.0

    sim_time = steps  # assuming 1s per step

    return steps, avg_queue_length, avg_wait_time, vehicles_completed, sim_time


def main():
    parser = argparse.ArgumentParser(
        description="Compute static controller stats from mac_runs/metrics.csv")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to a single run directory (e.g. mac_runs/run_20251126_125147)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Scenario name to put in the CSV (e.g. dynamic_seed1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed used for this run (for the 'seed' column)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="batch_results_static.csv",
        help="Where to write the output CSV (appended if exists)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    metrics_path = run_dir / "metrics.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found at {metrics_path}")

    (steps,
     avg_queue_length,
     avg_wait_time,
     vehicles_completed,
     sim_time) = compute_stats_from_metrics(metrics_path)

    output_path = Path(args.output).resolve()
    write_header = not output_path.exists()

    with output_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "scenario",
                "controller",
                "steps",
                "seed",
                "avg_wait_time",
                "avg_queue_length",
                "vehicles_completed",
                "sim_time",
            ])

        writer.writerow([
            args.scenario,
            "static",
            steps,
            args.seed,
            round(avg_wait_time, 3),
            round(avg_queue_length, 3),
            vehicles_completed,
            sim_time,
        ])

    print(f"Stats written to {output_path}")
    print("Row:")
    print(
        args.scenario,
        "static",
        steps,
        args.seed,
        round(avg_wait_time, 3),
        round(avg_queue_length, 3),
        vehicles_completed,
        sim_time,
    )


if __name__ == "__main__":
    main()
