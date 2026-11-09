# Reproducing the results

Every command runs from the repository root. Training writes one folder per run to `results/raw/`, and the analysis writes `results/aggregated/`, `paper_assets/figures/` and `paper_assets/tables/`.

## 1. Setup

You need an NVIDIA GPU whose driver supports CUDA 12.8, and conda. Linux and Windows both work. One group of 20 seeds needs at most about 5 GB of GPU memory.

```bash
conda env create -f environment.yml
conda activate ierstress
python scripts/check_env.py
```

The environment installs the exact versions behind every result, pinned in [`requirements-lock.txt`](../requirements-lock.txt). On a machine without a display, set `MUJOCO_GL=egl` (or `osmesa`) so that the DeepMind Control tasks load.

## 2. Check one reported result

**Without training.** [`results/`](../results) holds the records of every run. Each run folder in `results/raw/` has `record.json` (configuration, code hash, GPU, wall time), `eval.csv`, `train.csv`, `triggers.csv`, `repeats.csv` and `sources.csv`. `results/aggregated/` holds the aggregated results, and `results/replay_check/` and `results/precision_check/` the stored episodes of the two checks. Only the per-episode files `episodes.npz` (about 490 MB in total) are left out. `python scripts/make_paper_assets.py` rebuilds every figure and table of the paper from these files (section 5). `scripts/aggregate.py` also reads the per-episode files, so it needs retrained runs.

**With training.** Train the runs behind the result into a fresh folder, analyze them, and compare with the numbers in the [main README](../README.md#main-results) and the paper. Training into `results/raw/` would skip every run, since the shipped records mark them finished.

```bash
python scripts/run_main.py --item A1 --tasks walker-walk --processes 2 --out check/raw
python scripts/aggregate.py --raw check/raw --out check/aggregated
```

`check/aggregated/contrasts.csv` then holds every contrast these runs allow, one row per contrast (`id`, `task`, `condition`), with the estimate, its 95% interval and the win rate over seeds.

| Result | Training command (add `--processes 2 --out check/raw`) | Rows | GPU hours, one RTX 3090 |
|---|---|---|---|
| IER against SAC alone | `run_main.py --item A1 --tasks walker-walk` | P1 | 4 |
| Paper schedule against released code | `run_main.py --items A1 R1 --tasks HalfCheetah-v4` | P2 | 6 |
| Re-executing against copying | `run_main.py --items B1 C1 --tasks walker-walk` | P3 | 5 |
| Gain under stress | `run_main.py --items B1 B3 --tasks HalfCheetah-v4` | P4 | 4 |
| Stronger learner (CrossQ) | `run_main.py --item A2 --tasks walker-walk` | P6, CrossQ gain | 5 |
| Fixed repeat count | `run_main.py --items A1 D1 --tasks walker-walk` | P7 | 6 |
| Random-episode replay | `run_main.py --items B1 C1 E1 --tasks walker-walk`, then `python scripts/e1_analysis.py --raw check/raw --out check/e1.json` | `e1.json` | 6 |

Fewer seeds, for example `--seeds 0 1 2 3 4`, cost a quarter as much and give wider intervals. The rows that compare CrossQ with SAC (the second P6 row and P11) also need A1 on the same task. P5 is a correlation across all 12 stress cells, so it needs B1 and B3 on every task.

**Replay check.** It stores every best episode of short IER3 runs with its full simulator state, then replays each one from the identical start and from 20 fresh starts (about 35 minutes per task):

```bash
python scripts/replay_check.py collect --tasks walker-walk --out check
python scripts/replay_check.py replay  --tasks walker-walk --out check
python scripts/replay_check.py merge --out check          # check/aggregated/replay_check.csv
```

**Implementation check.** It trains our SAC and CrossQ beside stable-baselines3 SAC and sb3-contrib CrossQ on HalfCheetah, 3 seeds, 300k steps, about 4 hours. The outcome goes to `results/aggregated/gate_g1_*.json` and replaces the shipped files there, so copy them first to compare. The paper reports it in its implementation-check table:

```bash
python scripts/gate_g1.py run
```

## 3. Rerun everything

Move the shipped `results/` folder aside first (for example, rename it to `results_shipped/`), so that nothing is skipped and every script writes to its usual place.

```bash
python scripts/run_main.py --list            # every experiment with its runs and steps
python scripts/run_main.py --item A1         # and so on for each item below
python scripts/rebuild.py                    # numbers, figures and tables of the fixed analysis, and RESULTS.md
python scripts/e1_analysis.py
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

Hours are wall-clock time for one group on an RTX 3090 that runs two (CrossQ) or three (SAC) groups at once. The 1260 main runs took about 200 hours of training time, or about 90 GPU-hours with two or three groups sharing each GPU. `--tasks`, `--share ITEMS:TASKS` and `--seeds` split the work between GPUs. Finished runs are skipped, and an interrupted group resumes from its last checkpoint (every 50k steps). Each run writes `record.json` (configuration, code hash, GPU, wall time), `eval.csv`, `train.csv`, `triggers.csv`, `repeats.csv`, `sources.csv` and `episodes.npz`. Main seeds are 0 to 19.

## 4. What to expect

- Retraining gives the same results up to GPU arithmetic. The seeds fix every random stream, but different GPUs and drivers round differently, so single curves differ while the intervals and conclusions should hold. CrossQ was the most sensitive. For the closest match, use an RTX 3090 and the pinned versions.
- The replay check, the precision check and the random-episode replay (E1) are exploratory.
- [`third_party/ier/`](../third_party/ier) is the authors' IER code (MIT license), unmodified, at commit `fe757afaab1a066b266c5055817e99ad194290cd` of https://github.com/UoA-CARES/instant-episode-repetition (zip SHA256 `D1B4A179620C6E19ADF567FC1AE32E2DAF5DD983100CFE79FE989A7DFAE9D4C5`). This study's SAC uses its networks and settings.

## 5. Paper figures and tables

The figures and tables of the paper need `results/aggregated/`, the E1 runs in `results/raw/E1/`, the replay check of section 2 and the precision check below. All of them are shipped, so the command at the end of this section runs as it is.

**Precision check.** It asks how much a change at the level of rounding moves a replay. For each task it trains one short IER3 run on the CPU (seed 0, 100k steps, or 60k on finger-turn_hard, where repetition stops early) and keeps every stored best episode with its full start state. Up to 12 of these episodes are then replayed from the identical start with the actions cast to float64 before scaling, and from 20 fresh resets (about 20 minutes per task on one CPU core):

```bash
python scripts/precision_check.py --tasks HalfCheetah-v4 Hopper-v4 walker-walk finger-turn_hard --summary --out check/precision_check
```

Without `--out`, it overwrites the shipped `results/precision_check/`. `summary.json` gives, per task, the median relative change of the return under the cast and the first step at which the cast replay leaves the original one by more than one unit of spread.

**Figures and tables.** One command builds them all:

```bash
python scripts/make_paper_assets.py
```

It writes the figures to `paper_assets/figures/main/` (PDF and 300 dpi PNG) and the tables to `paper_assets/tables/main/`. It also writes `results/aggregated/multiple_comparisons.csv`, the exploratory multiple-comparison check of the paper: an exact sign-flip p-value over all 2^20 sign patterns for each of the 38 contrasts of the analysis plan, with Holm and Benjamini-Hochberg adjustments. Before drawing, it recomputes every per-seed score from `results/aggregated/curves.csv` and stops if one differs from `cells.csv`. The filmstrip figure renders MuJoCo frames, which needs `MUJOCO_GL=egl` on a machine without a display (the script sets it on Linux). Rendered frames can differ slightly between graphics drivers.
