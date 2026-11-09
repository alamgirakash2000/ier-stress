"""The repetition controller for the arms BASE, IER, IER3 and DUP.

Two schedules are available, because the paper and the released code differ.

`algorithm1` follows Algorithm 1 of the paper. This is the default for the IER arms:
    R_max starts at 0. When an episode ends with a return above R_max, the actions are
    stored and the counter is set to RN. Otherwise a positive counter drops by one. While
    the counter is positive, episodes replay the stored actions. A replay that beats
    R_max therefore re-triggers.

`released_code` follows train_loops/ier/episode_reward_loop.py in the released repository:
    A trigger sets the counter to RN but does not start repeating, so the next episode is a
    policy episode. RN - 1 replays follow. No episode can trigger until the counter is back
    at 0, which includes replays and the policy episode after the trigger.

Both schedules hand over to the policy when a replayed episode outlasts the stored actions
(the paper does not say what happens here, this is what the released code does). Exploration
actions take priority over replay in both.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

EXPLORE, REPLAY, POLICY = "explore", "replay", "policy"


class Mode(IntEnum):
    NONE = 0
    REPEAT = 1
    DUPLICATE = 2


@dataclass
class EndResult:
    triggered: bool = False
    dup_copies: int = 0  # how many copies of the stored episode the runner must insert


class RepetitionController:
    def __init__(self, mode: Mode, rn: int, exploration_steps: int, schedule: str = "algorithm1",
                 pick: str = "best", pick_seed: int | None = None):
        if schedule not in ("algorithm1", "released_code"):
            raise ValueError(f"Unknown schedule: {schedule}")
        if pick not in ("best", "random") or (pick == "random" and (schedule != "algorithm1" or pick_seed is None)):
            raise ValueError(f"pick={pick!r} needs the algorithm1 schedule and a pick_seed")
        self.mode = mode
        self.rn = rn
        self.exploration_steps = exploration_steps
        self.schedule = schedule
        self.r_max = 0.0
        self.counter = 0
        self.repeat_episode = False  # used only by the released_code schedule
        self.stored_actions: list | None = None
        # E1, IER3_RANDOM (exploratory): a trigger replays a uniformly drawn past episode, the triggering one
        # included, instead of the new best. The draw has its own stream, a seventh child of the run seed, so the six
        # streams of iers/runner.py are untouched.
        self.pick = pick
        self.past: list[tuple[np.ndarray, float]] = []
        self.stored_index = -1
        self.stored_return = 0.0
        self.pick_rng = (np.random.default_rng(np.random.SeedSequence(pick_seed).spawn(7)[6])
                         if pick == "random" else None)

    def source(self, t: int, ep_t: int):
        """Return (source, stored action or None) for global step t and 1-based episode step ep_t."""
        if t < self.exploration_steps:
            return EXPLORE, None
        if self.mode != Mode.REPEAT:
            return POLICY, None
        if self.schedule == "algorithm1":
            if self.counter > 0 and ep_t <= len(self.stored_actions):
                return REPLAY, self.stored_actions[ep_t - 1]
            return POLICY, None
        if self.counter > 0 and self.repeat_episode:
            action = self.stored_actions[ep_t - 1]
            if ep_t >= len(self.stored_actions):
                self.repeat_episode = False
            return REPLAY, action
        return POLICY, None

    def on_episode_end(self, episode_return: float, actions: list, steps_done: int) -> EndResult:
        """Update after an episode. steps_done is the number of environment steps taken so far."""
        if self.mode == Mode.NONE:
            return EndResult()
        if self.schedule == "released_code":
            return self._end_released(episode_return, actions, steps_done)
        if self.pick == "random":
            # One array per episode keeps checkpoints small. Stacking keeps every value exactly.
            self.past.append((np.stack(actions) if len(actions) else np.empty((0, 0)), float(episode_return)))
        if episode_return > self.r_max:
            self.r_max = episode_return
            self.stored_actions = actions
            self.stored_index, self.stored_return = -1, float(episode_return)
            if self.pick == "random":
                self.stored_index = int(self.pick_rng.integers(len(self.past)))
                drawn, self.stored_return = self.past[self.stored_index]
                self.stored_actions = list(drawn)
            if self.mode == Mode.REPEAT:
                self.counter = self.rn
                return EndResult(triggered=True)
            # DUP inserts when IER would start repeating. During exploration IER would not
            # repeat, because exploration actions take priority.
            copies = self.rn if steps_done >= self.exploration_steps else 0
            return EndResult(triggered=True, dup_copies=copies)
        if self.counter > 0:
            self.counter -= 1
        return EndResult()

    def _end_released(self, episode_return: float, actions: list, steps_done: int) -> EndResult:
        if self.counter == 1:
            self.repeat_episode = False
            self.counter -= 1
        elif self.counter > 1:
            self.counter -= 1
            self.repeat_episode = True
        elif episode_return > 0 and episode_return > self.r_max:
            self.r_max = episode_return
            self.stored_actions = actions
            if self.mode == Mode.REPEAT:
                self.counter = self.rn
                return EndResult(triggered=True)
            copies = self.rn if steps_done >= self.exploration_steps else 0
            return EndResult(triggered=True, dup_copies=copies)
        return EndResult()

    def state_dict(self) -> dict:
        state = {
            "r_max": self.r_max,
            "counter": self.counter,
            "repeat_episode": self.repeat_episode,
            "stored_actions": self.stored_actions,
        }
        if self.pick == "random":
            state.update(past=self.past, stored_index=self.stored_index, stored_return=self.stored_return,
                         pick_rng=self.pick_rng.bit_generator.state)
        return state

    def load_state_dict(self, state: dict) -> None:
        self.r_max = state["r_max"]
        self.counter = state["counter"]
        self.repeat_episode = state["repeat_episode"]
        self.stored_actions = state["stored_actions"]
        if self.pick == "random":
            self.past, self.stored_index = state["past"], state["stored_index"]
            self.stored_return = state["stored_return"]
            self.pick_rng.bit_generator.state = state["pick_rng"]


ARM_TO_MODE = {
    "BASE": Mode.NONE,
    "IER": Mode.REPEAT,
    "IER3": Mode.REPEAT,
    "IER_REL": Mode.REPEAT,
    "IER3_REL": Mode.REPEAT,
    "DUP": Mode.DUPLICATE,
    "IER3_RANDOM": Mode.REPEAT,  # E1, exploratory
}


def arm_pick(arm: str) -> str:
    """Which stored episode a trigger replays: the new best, or for IER3_RANDOM a uniformly drawn past one."""
    return "random" if arm == "IER3_RANDOM" else "best"


def arm_schedule(arm: str) -> str:
    """The sensitivity arms use the released-code schedule. Every other arm uses Algorithm 1."""
    return "released_code" if arm in ("IER_REL", "IER3_REL") else "algorithm1"


def arm_rn(arm: str, task_rn: int) -> int:
    """RN for an arm: the per-task paper value, fixed at 3 for IER3 and IER3_REL, and 0 for BASE.

    DUP takes the RN of the arm it is compared with. That is the per-task RN by default. Comparisons
    with IER3 pass rn=3 to make_spec, so that the only difference between the two arms is the arm's own
    change.
    """
    return {"BASE": 0, "IER": task_rn, "IER3": 3, "IER_REL": task_rn, "IER3_REL": 3, "DUP": task_rn,
            "IER3_RANDOM": 3}[arm]
