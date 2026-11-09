"""The training loop: a group of S seeds in lockstep, with S = 1 for a separate run.

For each seed the order of operations matches the upstream IER loop
(third_party/ier/train_loops/ier/episode_reward_loop.py):

    choose action, step, store, update, flag evaluation, then at episode end: log, trigger,
    evaluate if flagged, reset.

Each seed keeps its own environment, replay rows, random streams and repetition state, so a
seed's results do not depend on which other seeds share its group (test T10).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .batched import SOURCE_CODE, SOURCES, StackedReplay
from .codehash import code_hash, write_manifest
from .envs import denormalize, env_dims, make_env, normalize
from .learners.sac import SACConfig, StackedSAC
from .repetition import ARM_TO_MODE, EXPLORE, POLICY, REPLAY, Mode, RepetitionController, arm_pick
from .specs import (  # noqa: F401  (re-exported, so that everything training needs comes from one place)
    RESULTS_CKPT,
    RESULTS_RAW,
    Hardware,
    RunSpec,
    TrainConfig,
    atomic_savez,
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_text,
    is_finished,
    make_spec,
    replace_file,
)


@dataclass
class Streams:
    """One seed's independent random streams, all derived from the run seed."""

    env_seed: int
    action_seed: int
    init_seed: int
    replay: np.random.Generator
    noise: np.random.Generator
    cond: np.random.Generator

    @classmethod
    def from_seed(cls, seed: int) -> "Streams":
        children = np.random.SeedSequence(seed).spawn(6)
        as_int = lambda child: int(child.generate_state(1)[0] & 0x7FFFFFFF)  # noqa: E731
        return cls(
            env_seed=as_int(children[0]),
            action_seed=as_int(children[1]),
            init_seed=as_int(children[2]),
            replay=np.random.default_rng(children[3]),
            noise=np.random.default_rng(children[4]),
            cond=np.random.default_rng(children[5]),
        )


def default_env_factory(spec: RunSpec, streams: Streams):
    return make_env(spec.task, spec.condition, streams.env_seed, streams.action_seed, streams.cond)


def default_learner_factory(specs: list[RunSpec], init_seeds, obs_dim, act_dim, hw: Hardware):
    learner = specs[0].learner
    if learner == "sac":
        return StackedSAC(init_seeds, obs_dim, act_dim, SACConfig(), hw.device, hw.dtype)
    if learner == "crossq":
        from .learners.crossq import CrossQConfig, StackedCrossQ

        return StackedCrossQ(init_seeds, obs_dim, act_dim, CrossQConfig(), hw.device, hw.dtype)
    raise ValueError(f"Unknown learner: {learner}")


# ------------------------------------------------------------------ small helpers


def package_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    versions = {"python": platform.python_version()}
    for name in ("torch", "mujoco", "gymnasium", "dm_control", "numpy", "scipy", "pandas", "pydantic",
                 "opencv-python-headless"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def machine_info(device) -> dict:
    """Which machine ran a group. IERS_MACHINE and IERS_MACHINE_TYPE name it, for example 'cloud:3090:1' and
    'RTX 3090'. Arms compared within a task should share the machine type."""
    info = {"name": os.environ.get("IERS_MACHINE", "local"),
            "machine_type": os.environ.get("IERS_MACHINE_TYPE", "local"),
            "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
            "processes_per_gpu": int(os.environ.get("IERS_PROCESSES", "1")), "gpu": None}
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(torch.device(device))
    return info


def stack_or_empty(items: list, width: int) -> np.ndarray:
    return np.stack(items) if items else np.empty((0, width))


# ------------------------------------------------------------------ per-seed state


@dataclass
class SeedRun:
    spec: RunSpec
    env: object
    streams: Streams
    ctrl: RepetitionController
    obs: np.ndarray = None
    ep_t: int = 0
    ep_return: float = 0.0
    ep_index: int = 0
    ep_explore_steps: int = 0
    ep_replay_steps: int = 0
    ep_first_source: str = ""
    src: str = ""
    stored: object = None
    eval_flag: bool = False
    failed: bool = False
    reason: str = ""
    # Current episode. Upstream reads the stored actions back out of its memory. The same values
    # are kept here, in the same types.
    tr_actions: list = field(default_factory=list)
    tr_next_obs: list = field(default_factory=list)
    tr_rows: list = field(default_factory=list)
    # The stored best episode.
    best_path: np.ndarray = None
    best_rows: np.ndarray = None
    best_return: float = 0.0
    best_start_obs: np.ndarray = None
    best_trigger: int = -1
    # Logs.
    ep_log: list = field(default_factory=list)
    eval_log: list = field(default_factory=list)
    trigger_log: list = field(default_factory=list)
    trigger_starts: list = field(default_factory=list)
    repeat_log: list = field(default_factory=list)
    repeat_starts: list = field(default_factory=list)
    repeat_stored_starts: list = field(default_factory=list)
    repeat_deviation: list = field(default_factory=list)
    start_obs: list = field(default_factory=list)
    start_qpos: list = field(default_factory=list)
    start_qvel: list = field(default_factory=list)
    start_factors: list = field(default_factory=list)
    eval_start_obs: list = field(default_factory=list)
    eval_start_qpos: list = field(default_factory=list)
    eval_start_qvel: list = field(default_factory=list)
    eval_start_factors: list = field(default_factory=list)
    source_log: list = field(default_factory=list)

    def note_start(self) -> None:
        start = self.env.last_start
        self.start_obs.append(start.obs)
        self.start_qpos.append(start.qpos)
        self.start_qvel.append(start.qvel)
        if start.factors is not None:
            self.start_factors.append(np.concatenate([start.factors["mass"], start.factors["friction"]]))

    def note_eval_start(self) -> None:
        start = self.env.last_start
        self.eval_start_obs.append(start.obs)
        self.eval_start_qpos.append(start.qpos)
        self.eval_start_qvel.append(start.qvel)
        if start.factors is not None:
            self.eval_start_factors.append(np.concatenate([start.factors["mass"], start.factors["friction"]]))

    def episode_kind(self) -> str:
        if self.ep_first_source == REPLAY:
            return "replay"
        if self.ep_replay_steps > 0:
            return "mixed"  # began under exploration and went on into a replay
        if self.ep_explore_steps > 0:
            return "explore"
        return "policy"

    def clear_episode(self) -> None:
        self.ep_t = 0
        self.ep_return = 0.0
        self.ep_index += 1
        self.ep_explore_steps = 0
        self.ep_replay_steps = 0
        self.ep_first_source = ""
        self.tr_actions = []
        self.tr_next_obs = []
        self.tr_rows = []


_SEED_LOG_FIELDS = (
    "ep_t", "ep_return", "ep_index", "ep_explore_steps", "ep_replay_steps", "ep_first_source",
    "eval_flag", "failed", "reason", "tr_actions", "tr_next_obs", "tr_rows", "obs",
    "best_path", "best_rows", "best_return", "best_start_obs", "best_trigger",
    "ep_log", "eval_log", "trigger_log", "trigger_starts", "repeat_log", "repeat_starts",
    "repeat_stored_starts", "repeat_deviation", "start_obs", "start_qpos", "start_qvel",
    "start_factors", "eval_start_obs", "eval_start_qpos", "eval_start_qvel", "eval_start_factors", "source_log",
)
# Optional in a checkpoint, because checkpoints written before the source log existed lack it.
_OPTIONAL_SEED_FIELDS = {"source_log"}
SOURCE_WINDOW = 1000  # steps per row of sources.csv
SOURCE_COLUMNS = (["total_steps", "updates"] + [f"sampled_{s}" for s in SOURCES]
                  + ["max_share_replay", "max_share_copy"] + [f"buffer_{s}" for s in SOURCES]
                  + [f"evicted_{s}" for s in SOURCES])


# ------------------------------------------------------------------ the group


class GroupRunner:
    def __init__(self, specs: list[RunSpec], train: TrainConfig | None = None, hw: Hardware | None = None,
                 env_factory=default_env_factory, learner_factory=default_learner_factory,
                 results_root: Path = RESULTS_RAW, ckpt_root: Path = RESULTS_CKPT, logger=None,
                 record_losses: bool = False, write_outputs: bool = True, touch_replay: bool = False):
        budgets = {s.budget for s in specs}
        if len(budgets) != 1:
            raise ValueError("All seeds in a group must share one step budget.")
        self.specs = specs
        self.budget = budgets.pop()
        self.train = train or TrainConfig()
        self.hw = hw or Hardware()
        self.results_root = Path(results_root)
        self.log = logger.info if logger else print
        self.record_losses = record_losses
        self.write_outputs = write_outputs
        self.S = len(specs)
        self.t = 0
        # The code this session runs. Taken at start, so editing a file during a run cannot change what the
        # record says the run executed. A resumed run lists the hash of every session.
        self.code_hash_start = write_manifest()
        self.code_hashes = [self.code_hash_start]
        self.resume_count = 0
        self.wall_before = 0.0
        self.paused_s = 0.0  # time the resource guard held training back, part of wall_before
        self.paused_s_before = 0.0  # the same, from earlier sessions of this group
        self.checkpoint_on_stop = False  # set by a tick that asks for a clean stop
        self.train_env_steps = 0
        self.eval_env_steps = 0
        self.loss_log: list[np.ndarray] = []

        group_key = "|".join(sorted(s.run_id for s in specs))
        self.group_id = hashlib.sha256(group_key.encode()).hexdigest()[:16]
        self.ckpt_dir = Path(ckpt_root) / specs[0].experiment / self.group_id

        self.seeds: list[SeedRun] = []
        init_seeds = []
        dims = set()
        for spec in specs:
            streams = Streams.from_seed(spec.seed)
            env = env_factory(spec, streams)
            dims.add(env_dims(env))
            init_seeds.append(streams.init_seed)
            ctrl = RepetitionController(ARM_TO_MODE[spec.arm], spec.rn, self.train.exploration_steps, spec.schedule,
                                        pick=arm_pick(spec.arm), pick_seed=spec.seed)
            self.seeds.append(SeedRun(spec=spec, env=env, streams=streams, ctrl=ctrl))
        if len(dims) != 1:
            raise ValueError("All seeds in a group must share observation and action sizes.")
        self.O, self.A = dims.pop()

        self.learner = learner_factory(specs, init_seeds, self.O, self.A, self.hw)
        # A smaller allocation behaves the same as the 1e6 ring until it would overflow, which
        # only DUP can reach. DUP keeps the full capacity.
        needs_full = any(ARM_TO_MODE[s.arm] == Mode.DUPLICATE for s in specs)
        capacity = self.train.buffer_capacity if needs_full else min(self.train.buffer_capacity, self.budget)
        self.replay = StackedReplay(self.S, capacity, self.O, self.A, self.hw.replay_device, self.hw.dtype,
                                    touch=touch_replay)
        self.replay_saved = np.zeros(self.S, dtype=np.int64)

        np_dtype = np.float64 if self.hw.float64 else np.float32
        self.np_dtype = np_dtype
        self.batch = self.learner.cfg.batch_size
        self.obs_stack = np.zeros((self.S, self.O), dtype=np.float64)
        self.eps = np.zeros((self.S, self.A), dtype=np_dtype)
        self.noise = np.zeros((self.S, 2, self.batch, self.A), dtype=np_dtype)
        self.rows = np.zeros((self.S, self.replay.D), dtype=np_dtype)
        self.codes = np.full(self.S, -1, dtype=np.int8)
        self._reset_source_window()
        self.replay_rngs = [s.streams.replay for s in self.seeds]
        self.log(
            f"group {self.group_id} seeds={[s.seed for s in specs]} task={specs[0].task} "
            f"device_networks={self.learner.device} device_replay={self.replay.device} "
            f"dtype={self.hw.dtype} capacity={capacity} hardware={self.hw.label}"
        )

    # -------------------------------------------------------------- main loop

    def begin(self) -> None:
        """Resume from a checkpoint if one exists, otherwise start each episode."""
        if not self._load_checkpoint():
            for i, seed in enumerate(self.seeds):
                seed.obs = self._safe_reset(i)
                if not seed.failed:
                    seed.note_start()
                    self.obs_stack[i] = seed.obs

    def run(self, stop_at: int | None = None, tick=None) -> None:
        """Train up to the budget. stop_at simulates an interruption. tick(t) may return True to stop."""
        self.begin()
        start = time.time()
        end = self.budget if stop_at is None else min(stop_at, self.budget)
        while self.t < end:
            self.step()
            if (self.t % self.train.checkpoint_every == 0) and self.t < self.budget and self.write_outputs:
                self.wall_before += time.time() - start
                start = time.time()
                self.save_checkpoint()
            if tick is not None and tick(self.t):
                break
            if all(s.failed for s in self.seeds):
                self.log("all seeds failed, stopping the group")
                break
        self.wall_before += time.time() - start
        if self.t >= self.budget or all(s.failed for s in self.seeds):
            if self.write_outputs:
                self.finish()
        elif self.checkpoint_on_stop and self.write_outputs:
            self.save_checkpoint()

    def step(self) -> None:
        t = self.t
        seeds = self.seeds
        O, A = self.O, self.A
        rows = self.rows

        # Choose actions. Only seeds on the policy draw noise, and all policy seeds share one
        # forward pass.
        any_policy = False
        self.codes.fill(-1)
        for i, seed in enumerate(seeds):
            if seed.failed:
                continue
            seed.ep_t += 1
            seed.src, seed.stored = seed.ctrl.source(t, seed.ep_t)
            if seed.src == POLICY:
                any_policy = True
                seed.streams.noise.standard_normal(dtype=self.np_dtype, out=self.eps[i])
        policy_actions = self.learner.act(self.obs_stack, self.eps) if any_policy else None

        # Step each environment.
        for i, seed in enumerate(seeds):
            if seed.failed:
                continue
            env = seed.env
            src = seed.src
            if src == EXPLORE:
                action_env = env.sample_action()
                action = normalize(action_env, env.max_action_value, env.min_action_value)
                seed.ep_explore_steps += 1
            else:
                if src == REPLAY:
                    action = seed.stored
                    seed.ep_replay_steps += 1
                else:
                    action = policy_actions[i]
                    if not np.isfinite(action).all():
                        self._fail(i, f"non-finite policy action at step {t}")
                        continue
                action_env = denormalize(action, env.max_action_value, env.min_action_value)
            if seed.ep_t == 1:
                seed.ep_first_source = src
            try:
                next_obs, reward, done, truncated = env.step(action_env)
            except Exception as exc:  # A simulator failure is kept with its reason.
                self._fail(i, f"{type(exc).__name__} in training step {t}: {exc}")
                continue
            if not np.isfinite(next_obs).all():
                self._fail(i, f"non-finite observation at step {t}")
                continue
            self.train_env_steps += 1

            row = rows[i]
            row[:O] = seed.obs
            row[O:O + A] = action
            row[O + A] = reward
            row[O + A + 1:O + A + 1 + O] = next_obs
            row[-1] = done
            self.codes[i] = SOURCE_CODE[src]

            mode = seed.ctrl.mode
            if mode != Mode.NONE:
                seed.tr_actions.append(action)
                if mode == Mode.DUPLICATE:
                    seed.tr_rows.append(row.copy())
                else:
                    seed.tr_next_obs.append(next_obs)

            seed.obs = next_obs
            self.obs_stack[i] = next_obs
            seed.ep_return += reward
            seed.done_now = bool(done or truncated)

        self.replay.add_step(rows, self.codes)

        # Update.
        if t >= self.train.exploration_steps and t % self.train.updates_per_step == 0:
            batch = self.replay.sample(self.replay_rngs, self.batch)
            self._count_sources()
            for i, seed in enumerate(seeds):
                seed.streams.noise.standard_normal(dtype=self.np_dtype, out=self.noise[i])
            losses = self.learner.update(batch, self.noise, record=self.record_losses)
            if losses is not None:
                self.loss_log.append(losses)

        if (t + 1) % self.train.eval_every == 0:
            for seed in seeds:
                seed.eval_flag = True

        # Episode ends.
        ended = [i for i, s in enumerate(seeds) if not s.failed and getattr(s, "done_now", False)]
        if ended:
            due = []
            for i in ended:
                seed = seeds[i]
                seed.done_now = False
                self._end_episode(i, t)
                if seed.eval_flag:
                    due.append(i)
            if due:
                self._evaluate(due, t)
            for i in ended:
                seed = seeds[i]
                if seed.failed:
                    continue
                obs = self._safe_reset(i)
                if seed.failed:
                    continue
                seed.obs = obs
                self.obs_stack[i] = obs
                seed.clear_episode()
                seed.note_start()

        if (t + 1) % 1000 == 0:
            finite = self.learner.seed_is_finite()
            for i, seed in enumerate(seeds):
                if not seed.failed and not finite[i]:
                    self._fail(i, f"non-finite network parameters at step {t}")
        if (t + 1) % SOURCE_WINDOW == 0:
            self._flush_sources(t + 1)
        self.t = t + 1

    # -------------------------------------------------------------- transition sources

    def _reset_source_window(self) -> None:
        self.src_updates = 0
        self.src_counts = np.zeros((self.S, len(SOURCES)), dtype=np.int64)
        self.src_max = np.zeros((self.S, 2), dtype=np.float64)  # largest per-update share of replay and of copy

    def _count_sources(self) -> None:
        """Share of the minibatch just sampled from each source. Reads the indices already drawn, so it uses no
        random numbers and changes nothing the learner sees."""
        counts = self.replay.sampled_source_counts()
        self.src_counts += counts
        self.src_updates += 1
        shares = counts[:, [SOURCE_CODE["replay"], SOURCE_CODE["copy"]]] / self.batch
        np.maximum(self.src_max, shares, out=self.src_max)

    def _source_row(self, i: int, total_steps: int) -> tuple:
        return (total_steps, self.src_updates, *self.src_counts[i].tolist(),
                round(float(self.src_max[i, 0]), 6), round(float(self.src_max[i, 1]), 6),
                *self.replay.composition[i].tolist(), *self.replay.evicted[i].tolist())

    def _flush_sources(self, total_steps: int) -> None:
        for i, seed in enumerate(self.seeds):
            if not seed.failed:
                seed.source_log.append(self._source_row(i, total_steps))
        self._reset_source_window()

    def _end_episode(self, i: int, t: int) -> None:
        seed = self.seeds[i]
        kind = seed.episode_kind()
        steps = seed.ep_t
        seed.ep_log.append((seed.ep_index, t + 1, steps, float(seed.ep_return), kind, seed.ep_replay_steps))

        if kind == "replay":
            n = min(len(seed.tr_next_obs), len(seed.best_path))
            deviation = (np.linalg.norm(np.asarray(seed.tr_next_obs[:n]) - seed.best_path[:n], axis=1) if n
                         else np.empty(0))
            # The stored length comes from the stored actions. For every arm but IER3_RANDOM it equals the stored
            # path's length. IER3_RANDOM keeps no path of the drawn episode, so it has no deviation.
            seed.repeat_log.append(
                (t + 1, steps, float(seed.ep_return), float(seed.best_return), seed.best_trigger,
                 seed.ep_replay_steps, len(seed.ctrl.stored_actions))
            )
            seed.repeat_starts.append(seed.env.last_start.obs)
            seed.repeat_stored_starts.append(seed.best_start_obs)
            seed.repeat_deviation.append(deviation.astype(np.float32))

        result = seed.ctrl.on_episode_end(float(seed.ep_return), seed.tr_actions, t + 1)
        if result.triggered:
            seed.best_return = float(seed.ep_return)
            seed.best_start_obs = seed.env.last_start.obs
            seed.best_trigger = len(seed.trigger_log)
            seed.trigger_log.append((t + 1, float(seed.ep_return), steps, kind))
            seed.trigger_starts.append(seed.env.last_start.obs)
            if seed.ctrl.pick == "random":
                # IER3_RANDOM replays a drawn past episode: retention is measured against its own return and start.
                seed.best_return = seed.ctrl.stored_return
                seed.best_start_obs = seed.start_obs[seed.ctrl.stored_index]
                seed.best_path = np.empty((0, self.O))
            elif seed.ctrl.mode == Mode.REPEAT:
                seed.best_path = np.asarray(seed.tr_next_obs)
            else:
                seed.best_rows = np.stack(seed.tr_rows)
        if result.dup_copies:
            self.replay.add_many(i, np.tile(seed.best_rows, (result.dup_copies, 1)), code=SOURCE_CODE["copy"])

    def _safe_reset(self, i: int):
        seed = self.seeds[i]
        try:
            return seed.env.reset()
        except Exception as exc:
            self._fail(i, f"{type(exc).__name__} in reset: {exc}")
            return None

    def _fail(self, i: int, reason: str) -> None:
        seed = self.seeds[i]
        seed.failed = True
        seed.reason = reason
        self.log(f"run {seed.spec.run_id} failed: {reason}")

    def _evaluate(self, due: list[int], t: int) -> None:
        """Ten deterministic episodes for each due seed. These steps do not count toward the budget."""
        seeds = self.seeds
        obs_stack = self.obs_stack.copy()
        n_eval = self.train.eval_episodes
        returns = {}
        counts = {}
        active = []
        for i in due:
            obs = self._safe_reset(i)
            if seeds[i].failed:
                continue
            seeds[i].note_eval_start()
            obs_stack[i] = obs
            returns[i] = 0.0
            counts[i] = 0
            active.append(i)
        while active:
            actions = self.learner.act_deterministic(obs_stack)
            still_active = []
            for i in active:
                seed = seeds[i]
                env = seed.env
                action_env = denormalize(actions[i], env.max_action_value, env.min_action_value)
                try:
                    obs, reward, done, truncated = env.step(action_env)
                except Exception as exc:
                    self._fail(i, f"{type(exc).__name__} in evaluation at step {t}: {exc}")
                    continue
                self.eval_env_steps += 1
                returns[i] += reward
                if done or truncated:
                    counts[i] += 1
                    seed.eval_log.append((t + 1, counts[i], float(returns[i])))
                    returns[i] = 0.0
                    obs = self._safe_reset(i)
                    if seed.failed:
                        continue
                    if counts[i] < n_eval:
                        seed.note_eval_start()
                if counts[i] < n_eval:
                    obs_stack[i] = obs
                    still_active.append(i)
            active = still_active
        for i in due:
            seeds[i].eval_flag = False

    # -------------------------------------------------------------- checkpoints

    def _replay_path(self, i: int) -> Path:
        return self.ckpt_dir / f"replay_{i}.bin"

    def _tags_path(self, i: int) -> Path:
        return self.ckpt_dir / f"tags_{i}.bin"

    def save_checkpoint(self) -> None:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        row_bytes = self.replay.D * np.dtype(self.np_dtype).itemsize
        for i in range(self.S):
            # Drop anything written after the last good checkpoint, then append the new rows.
            path = self._replay_path(i)
            start_total = int(self.replay_saved[i])
            mode = "r+b" if path.exists() else "wb"
            with open(path, mode) as handle:
                handle.seek(start_total * row_bytes)
                handle.truncate()
                handle.write(self.replay.rows_since(i, start_total).tobytes())
                handle.flush()
                os.fsync(handle.fileno())
            tag_path = self._tags_path(i)
            with open(tag_path, "r+b" if tag_path.exists() else "wb") as handle:
                handle.seek(start_total)  # one byte per row
                handle.truncate()
                handle.write(self.replay.tags_since(i, start_total).tobytes())
                handle.flush()
                os.fsync(handle.fileno())
            self.replay_saved[i] = self.replay.total[i]

        per_seed = []
        for seed in self.seeds:
            entry = {name: getattr(seed, name) for name in _SEED_LOG_FIELDS}
            entry["ctrl"] = seed.ctrl.state_dict()
            entry["replay_rng"] = seed.streams.replay.bit_generator.state
            entry["noise_rng"] = seed.streams.noise.bit_generator.state
            entry["env"] = seed.env.get_state() if not seed.failed else None
            per_seed.append(entry)
        payload = {
            "t": self.t,
            "learner": self.learner.state_dict(),
            "replay_total": self.replay.total.copy(),
            "seeds": per_seed,
            "wall_before": self.wall_before,
            "paused_s": self.paused_s,
            "resume_count": self.resume_count,
            "train_env_steps": self.train_env_steps,
            "eval_env_steps": self.eval_env_steps,
            "loss_log": self.loss_log,
            "code_hashes": self.code_hashes,
            "source_window": {"updates": self.src_updates, "counts": self.src_counts.copy(),
                              "max": self.src_max.copy()},
            "replay_evicted": self.replay.evicted.copy(),
        }
        tmp = self.ckpt_dir / "ckpt.pt.tmp"
        torch.save(payload, tmp)
        replace_file(tmp, self.ckpt_dir / "ckpt.pt")
        self.log(f"group {self.group_id} checkpoint at step {self.t}")

    def _load_checkpoint(self) -> bool:
        path = self.ckpt_dir / "ckpt.pt"
        if not path.exists():
            return False
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.t = payload["t"]
        self.learner.load_state_dict(payload["learner"])
        self.wall_before = payload["wall_before"]
        self.paused_s = self.paused_s_before = payload.get("paused_s", 0.0)
        self.resume_count = payload["resume_count"] + 1
        earlier = payload.get("code_hashes", ["not recorded: checkpoint written before hashes were taken at start"])
        self.code_hashes = list(earlier) + [self.code_hash_start]
        self.train_env_steps = payload["train_env_steps"]
        self.eval_env_steps = payload["eval_env_steps"]
        self.loss_log = payload["loss_log"]
        window = payload.get("source_window")
        if window is not None:
            self.src_updates, self.src_counts, self.src_max = window["updates"], window["counts"], window["max"]
        for i, (seed, entry) in enumerate(zip(self.seeds, payload["seeds"])):
            for name in _SEED_LOG_FIELDS:
                if name in entry or name not in _OPTIONAL_SEED_FIELDS:
                    setattr(seed, name, entry[name])
            seed.ctrl.load_state_dict(entry["ctrl"])
            seed.streams.replay.bit_generator.state = entry["replay_rng"]
            seed.streams.noise.bit_generator.state = entry["noise_rng"]
            if entry["env"] is not None:
                seed.env.set_state(entry["env"])
            total = int(payload["replay_total"][i])
            keep = min(total, self.replay.capacity)
            row_bytes = self.replay.D * np.dtype(self.np_dtype).itemsize
            rows = np.fromfile(
                self._replay_path(i), dtype=self.np_dtype, count=keep * self.replay.D,
                offset=(total - keep) * row_bytes,
            ).reshape(keep, self.replay.D)
            tag_path = self._tags_path(i)
            tags = (np.fromfile(tag_path, dtype=np.int8, count=keep, offset=total - keep)
                    if tag_path.exists() and tag_path.stat().st_size >= total else None)
            self.replay.load_rows(i, rows, total, tags)
            self.replay_saved[i] = total
            if seed.obs is not None:
                self.obs_stack[i] = seed.obs
        if "replay_evicted" in payload:
            self.replay.evicted[:] = payload["replay_evicted"]
        self.log(f"group {self.group_id} resumed at step {self.t} (resume {self.resume_count})")
        return True

    # -------------------------------------------------------------- outputs

    def run_dir(self, spec: RunSpec) -> Path:
        return self.results_root / spec.experiment / spec.run_id

    def finish(self) -> None:
        wall = self.wall_before
        chash = self.code_hash_start
        hash_at_finish = code_hash()
        versions = package_versions()
        partial = bool(self.src_updates or self.t % SOURCE_WINDOW)  # a budget or stop between two windows
        machine = machine_info(self.learner.device)
        for i, seed in enumerate(self.seeds):
            spec = seed.spec
            out = self.run_dir(spec)
            tail = [self._source_row(i, self.t)] if partial and not seed.failed else []
            atomic_write_csv(out / "sources.csv", SOURCE_COLUMNS, seed.source_log + tail)
            atomic_write_csv(out / "eval.csv", ["total_steps", "episode", "return"], seed.eval_log)
            atomic_write_csv(out / "train.csv", ["episode", "total_steps", "length", "return", "kind", "replay_steps"],
                             seed.ep_log)
            atomic_write_csv(out / "triggers.csv", ["total_steps", "new_best", "length", "kind"], seed.trigger_log)
            atomic_write_csv(
                out / "repeats.csv",
                ["total_steps", "length", "return", "stored_return", "trigger", "replay_steps", "stored_length"],
                seed.repeat_log,
            )
            deviation = seed.repeat_deviation
            atomic_savez(
                out / "episodes.npz",
                start_obs=stack_or_empty(seed.start_obs, self.O),
                start_qpos=np.stack(seed.start_qpos) if seed.start_qpos else np.empty((0, 0)),
                start_qvel=np.stack(seed.start_qvel) if seed.start_qvel else np.empty((0, 0)),
                start_factors=np.stack(seed.start_factors) if seed.start_factors else np.empty((0, 0)),
                eval_start_obs=stack_or_empty(seed.eval_start_obs, self.O),
                eval_start_qpos=np.stack(seed.eval_start_qpos) if seed.eval_start_qpos else np.empty((0, 0)),
                eval_start_qvel=np.stack(seed.eval_start_qvel) if seed.eval_start_qvel else np.empty((0, 0)),
                eval_start_factors=np.stack(seed.eval_start_factors) if seed.eval_start_factors else np.empty((0, 0)),
                trigger_start_obs=stack_or_empty(seed.trigger_starts, self.O),
                repeat_start_obs=stack_or_empty(seed.repeat_starts, self.O),
                repeat_stored_start_obs=stack_or_empty(seed.repeat_stored_starts, self.O),
                repeat_deviation=np.concatenate(deviation) if deviation else np.empty(0, dtype=np.float32),
                repeat_deviation_lengths=np.array([len(d) for d in deviation], dtype=np.int64),
            )
            if self.record_losses:
                np.save(out / "losses.npy", np.stack([row[i] for row in self.loss_log]))
            record = {
                "run_id": spec.run_id,
                "status": "failed" if seed.failed else "finished",
                "reason": seed.reason,
                "spec": asdict(spec),
                "train_config": asdict(self.train),
                "learner_config": asdict(self.learner.cfg),
                "hardware": asdict(self.hw),
                "device_networks": str(self.learner.device),
                "device_replay": str(self.replay.device),
                "group_id": self.group_id,
                "group_seeds": [s.seed for s in self.specs],
                "code_hash": chash,
                "code_hashes_by_session": self.code_hashes,
                "code_hash_at_finish": hash_at_finish,
                "package_versions": versions,
                "platform": platform.platform(),
                "machine": machine,
                "seed": spec.seed,
                "steps_reached": self.t,
                "wall_time_s": wall,
                "guard_paused_s": self.paused_s,
                "resume_count": self.resume_count,
                "train_env_steps_group": self.train_env_steps,
                "eval_env_steps_group": self.eval_env_steps,
                "python_argv": sys.argv,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            atomic_write_text(out / "record.json", json.dumps(record, indent=1, default=str))
        if self.ckpt_dir.exists():
            shutil.rmtree(self.ckpt_dir, ignore_errors=True)
        self.log(f"group {self.group_id} finished, wall {wall:.1f} s")


