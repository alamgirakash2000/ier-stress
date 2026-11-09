# Reproducing the results

Every command runs from the repository root. The saved runs are in `results/raw/`. To check a result, retrain it into a fresh folder and compare it with them.

## 1. Setup

You need an NVIDIA GPU whose driver supports CUDA 12.8, and conda. Linux and Windows both work. One group of 20 seeds needs at most about 5 GB of GPU memory.

```bash
conda env create -f environment.yml
conda activate ierstress
python scripts/check_env.py
```

The environment installs the exact versions behind every result, pinned in [`requirements-lock.txt`](../requirements-lock.txt). On a machine without a display, set `MUJOCO_GL=egl` (or `osmesa`) so that the DeepMind Control tasks load.

## 2. Check a reported result by retraining it

Retrain the runs behind one result into a fresh folder, analyse them with the same code, and compare with the saved analysis:

```bash
python scripts/run_main.py --item A1 --tasks walker-walk --processes 2 --out check/raw
python scripts/aggregate.py --raw check/raw --out check/aggregated
```

Then compare `check/aggregated/contrasts.csv` with [`results/aggregated/contrasts.csv`](../results/aggregated/contrasts.csv), row by row (`id`, `task`, `condition`). `gains.csv` and `cells.csv` compare the same way.

| Result | Command (add `--processes 2 --out check/raw`) | Rows | GPU hours, one RTX 3090 |
|---|---|---|---|
| IER against SAC alone | `run_main.py --item A1 --tasks walker-walk` | P1 | 4 |
| Paper schedule against released code | `run_main.py --items A1 R1 --tasks HalfCheetah-v4` | P2 | 6 |
| Re-executing against copying | `run_main.py --items B1 C1 --tasks walker-walk` | P3, gains | 5 |
| Gain under stress | `run_main.py --items B1 B3 --tasks HalfCheetah-v4` | P4 | 4 |
| Stronger learner (CrossQ) | `run_main.py --item A2 --tasks walker-walk` | P6, CrossQ gain | 5 |
| Fixed repeat count | `run_main.py --items A1 D1 --tasks walker-walk` | P7 | 6 |
| Random-episode replay (E1) | `run_main.py --item E1`, then `python scripts/e1_analysis.py --raw check/raw --out check/e1.json` | `e1.json` | 1 |

Each task can also be checked with fewer seeds, for example `--seeds 0 1 2 3 4`, at a quarter of the cost and with wider intervals. Rows that need an item you did not retrain are missing from the check folder. The rows that compare CrossQ with SAC (the second P6 row and P11) need A1 on the same task as well. P5 is a correlation across all 12 stress cells, so it needs every task.

**Replay check.** It stores every best episode of short IER3 runs with its full simulator state, then replays each one from the identical start and from 20 fresh starts (about 35 minutes per task):

```bash
python scripts/replay_check.py collect --tasks walker-walk --out check
python scripts/replay_check.py replay  --tasks walker-walk --out check
python scripts/replay_check.py merge --out check          # check/aggregated/replay_check.csv
```

**Implementation check.** It trains our SAC and CrossQ beside stable-baselines3 SAC and sb3-contrib CrossQ on HalfCheetah, 3 seeds, 300k steps (about 4 hours):

```bash
python scripts/gate_g1.py run
```

It writes `results/aggregated/gate_g1_*.json`, which `git diff` compares with the saved versions.

## 3. Rebuild every number, figure and table from the saved runs

No GPU, a few minutes:

```bash
python scripts/rebuild.py
python scripts/e1_analysis.py
```

This writes `results/aggregated/`, `paper_assets/figures/`, `paper_assets/tables/` and `RESULTS.md` (one verdict per question, by the fixed rules of [`iers/verdicts.py`](../iers/verdicts.py)). It reads only the experiments named in [`logs/UNBLIND_OK.md`](../logs/UNBLIND_OK.md).

## 4. Rerun every experiment

```bash
python scripts/run_main.py --list
python scripts/run_main.py --item A1 --out rerun/raw
```

| Item | What | Groups of 20 seeds | Hours per group |
|---|---|---|---|
| A1 | SAC with and without IER, 4 tasks, 1M steps | 8 | 5 |
| B1 | SAC, IER3 under `default`, `noise5`, `dyn20`, 3 tasks, 300k | 18 | 1.4 |
| C1 | Copy control DUP, 3 tasks, `default` and `noise5`, 300k | 6 | 1.4 |
| R1 | Released-code schedule, 4 tasks, 1M | 4 | 5 |
| A2 | CrossQ with and without IER, 4 tasks, 300k | 8 | 5 |
| D1 | Fixed RN = 3, 4 tasks, 1M | 4 | 5 |
| B2 | CrossQ, IER3 under `noise5`, 3 tasks, 300k | 9 | 5 |
| B3 | SAC, IER3 under `noise10`, 3 tasks, 300k | 6 | 1.4 |
| E1 | Random-episode replay, walker-walk, 300k (exploratory) | 1 | 1.4 |

Hours are wall-clock time for one group on an RTX 3090 that runs two (CrossQ) or three (SAC) groups at once. The whole study took about 93 GPU-hours. `--tasks`, `--share ITEMS:TASKS` and `--seeds` split the work between GPUs. Finished runs are skipped, and an interrupted group resumes from its last checkpoint (every 50k steps). Each run writes `record.json` (configuration, code hash, GPU, wall time), `eval.csv`, `train.csv`, `triggers.csv`, `repeats.csv`, `sources.csv` and `episodes.npz`. Main seeds are 0 to 19.

## 5. What to expect

- Rebuilding from the saved runs (Section 3) gives the same numbers, figures and tables exactly.
- Retraining gives the same results up to GPU arithmetic. The seeds fix every random stream, but different GPUs and drivers round differently, so single curves differ while the intervals and conclusions should hold. CrossQ was the most sensitive: its first 50k steps differed between the pilot machine and the RTX 3090s used here. For the closest match, use an RTX 3090 and the pinned versions.
- Since the main runs, the training code changed only by additions that leave the main arms as they were (the E1 arm, the `--seeds` and `--out` options, job file names) and by comments. Every run records the hash of its code, and [`results/code_manifests/`](../results/code_manifests) maps each hash to the hash of every file.
- The replay check and E1 are exploratory.
- [`third_party/ier/`](../third_party/ier) is the authors' IER code (MIT license), unmodified, at commit `fe757afaab1a066b266c5055817e99ad194290cd` of https://github.com/UoA-CARES/instant-episode-repetition (zip SHA256 `D1B4A179620C6E19ADF567FC1AE32E2DAF5DD983100CFE79FE989A7DFAE9D4C5`). This study's SAC uses its networks and settings.
