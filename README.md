# SUMO Traffic RL – Project README

This repository contains a small, reproducible pipeline for:

1. Building SUMO grid networks (2×2 and 4×4 variants).
2. Generating fixed-time traffic data on macOS.
3. Training and testing multiple RL controllers (shared policy, lane‑attention, region‑aware) on Ubuntu.

The goal is to let you quickly re‑build the environment, regenerate data, and re‑run all RL experiments.

---
## 1. High‑Level Folder Layout

Very roughly, the project looks like this:

- `envs/`
  - `grid2x2/` – small 3×3 grid generating a 2×2 core of intersections; used by the classic `marddpg` environment.
  - `grid4x4_uniform/` – 4×4 grid with balanced flows.
  - `grid4x4_ns_heavy/` – 4×4 grid with heavier North–South flows.
  - `grid4x4_rush_hour/` – 4×4 grid with time‑varying (rush‑hour) patterns.
- `scripts/`
  - Network builders for each scenario (see below).
  - Environment checks / TraCI sanity scripts.
- `mac_runs/`
  - Each run folder (e.g. `run_YYYYMMDD_HHMMSS/`) contains `tripinfo.xml`, `summary.xml`, `metrics.csv` generated on macOS.
- `marddpg/` and `rl_attention/`
  - RL environments and policy implementations.
- `train_*.py` / `test_*.py`
  - Entrypoints for training and evaluating different RL policies.

---
## 2. Dependencies & Setup

1. **Python**: 3.12.
2. **SUMO** (and `netgenerate`, `netconvert`, `sumolib` on your `PATH`).
3. **Python packages**: `numpy`, `torch`, and any extra RL / plotting libraries you used.
4. **Environment variables**: set `SUMO_HOME` and ensure `$SUMO_HOME/tools` is on `PYTHONPATH` or `PATH` so TraCI works.

After installing, clone this repo and work from the project root.

---
## 3. Build SUMO Networks

These scripts live in `scripts/` and rebuild the corresponding `envs/...` folders.

### 3.1 Small grid for classic experiments

- **Script**: `scripts/make_grid_net.py`
- **Output folder**: `envs/grid2x2/`
- **What it does**: builds a small grid network and writes:
  - `net.net.xml` – the network topology
  - `routes.rou.xml` – basic, roughly uniform flows
  - `grid.sumocfg` – SUMO configuration file

**Run:**
```bash
python scripts/make_grid_net.py
```

### 3.2 4×4 grid – uniform demand

- **Script**: `scripts/make_grid4x4_uniform.py`
- **Output folder**: `envs/grid4x4_uniform/`
- **What it does**: builds a 4×4 grid with similar flows in all directions (balanced demand).

**Run:**
```bash
python scripts/make_grid4x4_uniform.py
```

### 3.3 4×4 grid – North–South heavy

- **Script**: `scripts/make_grid4x4_ns_heavy.py`
- **Output folder**: `envs/grid4x4_ns_heavy/`
- **What it does**: same 4×4 geometry but with heavier North–South flows, lighter East–West.

**Run:**
```bash
python scripts/make_grid4x4_ns_heavy.py
```

### 3.4 4×4 grid – Rush‑hour scenario

- **Script**: `scripts/make_grid4x4_rush_hour.py`
- **Output folder**: `envs/grid4x4_rush_hour/`
- **What it does**: 4×4 grid where traffic demand changes over time (e.g. morning vs midday vs evening).

**Run:**
```bash
python scripts/make_grid4x4_rush_hour.py
```

You can rebuild any scenario at any time – the scripts overwrite the old network and route files.

---
## 4. Generate Fixed‑Time Data on macOS

Once the small grid (grid2x2) exists, you can generate per‑step traffic metrics using a fixed‑time controller.

- **Script**: `mac_generate_data.py`
- **What it does:**
  - Ensures `envs/grid2x2/grid.sumocfg` exists (builds it if needed).
  - Creates a new subfolder in `mac_runs/` with a timestamp.
  - Runs SUMO with the `marddpg` multi‑traffic‑light environment.
  - Logs per‑step aggregate metrics to `metrics.csv` (queue length, waiting time, arrivals), plus `tripinfo.xml` and `summary.xml`.

**Run (example):**
```bash
python mac_generate_data.py --steps 3600 --seed 42
```

You can copy any finished `mac_runs/run_...` folder to your Ubuntu machine and reuse the XML/CSV outputs there if needed.

---
## 5. Basic Environment Sanity Checks

These scripts are for quickly verifying that SUMO + TraCI + the RL environment are wired correctly.

- **`scripts/test_multi_tl_env.py`**
  - Uses the `marddpg` multi‑traffic‑light environment directly.
  - Resets the env, prints discovered traffic lights and a sample observation.
  - Steps the environment for a few iterations with simple dummy actions.

Run this before any RL training to confirm that SUMO starts and the observations/rewards look sane.

```bash
python scripts/test_multi_tl_env.py
```

You may also have small helper scripts like `check_env.py` / `hello_traci.py` for very early debugging; they are optional once the main env tests pass.

---
## 6. RL Baseline – Shared Policy

### 6.1 Train a simple shared Gaussian policy

- **Script**: `train_shared_policy.py`
- **Idea**: one small MLP policy shared by all intersections.
- **What it does**:
  - Wraps the SUMO environment in a small Gym‑style wrapper.
  - Runs a REINFORCE loop for a few episodes.
  - Uses the same network weights for every traffic light.

**Run:**
```bash
python train_shared_policy.py
```

### 6.2 Test the shared policy

- **Script**: `test_simple_policy_inference.py`
- **What it does**:
  - Loads the trained shared policy.
  - Rolls out a short episode and prints actions + rewards per step.

**Run:**
```bash
python test_simple_policy_inference.py
```

This gives you a very simple baseline controller to compare against more advanced attention‑based variants.

---
## 7. Lane‑Attention Policy

### 7.1 Train attention‑based lane policy

- **Script**: `train_lane_attention_policy.py`
- **Idea**: each intersection gets an attention module over its incoming lanes.
- **What it does**:
  - Builds a `LaneAttentionGaussianPolicy` using lane‑level embeddings.
  - Trains it with REINFORCE for several episodes.
  - Saves the resulting weights to `models/lane_attn_policy.pt`.

**Run:**
```bash
python train_lane_attention_policy.py
```

### 7.2 Inspect lane embeddings

- **Script**: `test_lane_embeddings.py`
- **What it does**:
  - Resets the environment and computes lane‑context embeddings.
  - Prints shapes / sample values per traffic light (for debugging and understanding what the attention is producing).

**Run:**
```bash
python test_lane_embeddings.py
```

### 7.3 Quick rollout using lane‑attention policy

- **Script**: `test_lane_attention_policy.py`
- **What it does**:
  - Constructs a `LaneAttentionPolicy` and runs a short rollout.
  - Saves the policy weights for later reuse.

**Run:**
```bash
python test_lane_attention_policy.py
```

---
## 8. Region‑Aware Lane‑Attention

Here we add a second layer of structure: intersections can be grouped into a small number of **regions**, and the policy can learn shared patterns per region.

### 8.1 Dynamic region clustering (offline test)

- **Script**: `test_dynamic_region_clustering.py`
- **What it does**:
  - Builds an untrained `LaneAttentionGaussianPolicy` and computes lane‑context embeddings.
  - Clusters intersections into `K` regions using these embeddings.
  - Prints the region assignments and per‑region members.

**Run:**
```bash
python test_dynamic_region_clustering.py
```

### 8.2 Train region‑aware policy

- **Script**: `train_region_aware_policy.py`
- **What it does**:
  - Uses lane‑context embeddings + clustering to assign each intersection to a region.
  - Periodically re‑clusters intersections every few episodes.
  - Trains a region‑aware attention policy with REINFORCE.
  - Saves the trained weights to `models/lane_attn_region_policy.pt`.

**Run:**
```bash
python train_region_aware_policy.py
```

This is the more advanced controller that uses both lane‑level attention and dynamic region structure.

---
## 9. Neighbor‑Graph Mixing Between Intersections

- **Script**: `test_intersection_neighbor_mixer.py`
- **What it does**:
  - Builds a graph of neighboring intersections from the SUMO network.
  - Computes lane‑context embeddings for each intersection.
  - Applies an `IntersectionNeighborMixer` to produce neighbor‑aware embeddings.
  - Prints both original and mixed embeddings for inspection.

**Run:**
```bash
python test_intersection_neighbor_mixer.py
```

This is a building block for future experiments where intersections share information with their immediate neighbors.

---
## 10. Minimal “Replicate Everything” Order

If someone wants to roughly redo the whole pipeline, here is a very short, practical order:

1. **Build networks** (on any machine):
   - `python scripts/make_grid_net.py`  (2×2 base)
   - `python scripts/make_grid4x4_uniform.py`
   - `python scripts/make_grid4x4_ns_heavy.py`
   - `python scripts/make_grid4x4_rush_hour.py`
2. **Generate fixed‑time data (Mac)**:
   - `python mac_generate_data.py --steps 3600 --seed 42`
   - Copy `mac_runs/run_...` to your Ubuntu machine if needed.
3. **Sanity check SUMO + env**:
   - `python scripts/test_multi_tl_env.py`
4. **Train + test simple shared policy**:
   - `python train_shared_policy.py`
   - `python test_simple_policy_inference.py`
5. **Train + inspect lane‑attention policy**:
   - `python train_lane_attention_policy.py`
   - `python test_lane_embeddings.py`
   - `python test_lane_attention_policy.py`
6. **Region‑aware + neighbor experiments**:
   - `python test_dynamic_region_clustering.py`
   - `python train_region_aware_policy.py`
   - `python test_intersection_neighbor_mixer.py`

You can pick and choose from this list depending on which part of the project you want to demonstrate (simple baseline vs. attention vs. region‑aware coordination).

