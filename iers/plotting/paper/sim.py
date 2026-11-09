"""Simulation-backed inputs for fig_filmstrip and the float64 line of fig_replay (panel b). Needs mujoco and dm_control.

1. float64 line. The precision check (scripts/precision_check.py) logs its float64 deviation as the RMS of
   spread-scaled observation differences, a different unit from the replay check (plain Euclidean distance between
   raw observations). This replays the precision check's captured episodes the same way (identical start state,
   actions as stored, then the same actions cast to float64), records the raw Euclidean distance per step, and
   checks that the spread-scaled distance reproduces the precision check's own curves.
2. Filmstrip. One late, high-return stored episode per task from the official replay check, replayed from its
   identical start (this reproduces the stored episode bit for bit) and from the first fresh default reset of the
   official replay check (fresh_01), rendered at fixed steps.
"""

from __future__ import annotations

import copy
import json
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from .common import AGG, CACHE, PRECISION, REPLAY_STORE, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT))

F64_TASKS = ["HalfCheetah-v4", "Hopper-v4", "walker-walk", "finger-turn_hard"]
N_TEST = 12  # as in scripts/precision_check.py


def _run(env, actions, record_obs=True):
    from iers.envs import denormalize

    obs, rewards = [], []
    for a in actions:
        nxt, r, done, trunc = env.step(denormalize(a, env.max_action_value, env.min_action_value))
        obs.append(np.array(nxt, dtype=np.float64))
        rewards.append(float(r))
        if done or trunc:
            break
    return np.array(obs), np.array(rewards)


def float64_raw(force: bool = False) -> dict:
    """Raw Euclidean distance between the float32 and float64 replays of each tested precision-check episode."""
    out = CACHE / "f64_raw.npz"
    meta_path = CACHE / "f64_raw.json"
    if out.exists() and meta_path.exists() and not force:
        return json.loads(meta_path.read_text())
    from iers.envs import make_env
    from iers.metrics import obs_scale

    CACHE.mkdir(parents=True, exist_ok=True)
    arrays, meta = {}, {"episodes": [], "check": {}}
    for task in F64_TASKS:
        seed = 0
        captured = pickle.load(open(PRECISION / f"{task}_s{seed}_captured.pkl", "rb"))
        curves = json.load(open(PRECISION / f"{task}_s{seed}_curves.json"))
        ref_curves = {c["trigger_step"]: np.array(c["dev"]) for c in curves if c["fresh"] == -1}
        scale = obs_scale(task)
        used = scale > 0
        picks = sorted(set(np.linspace(0, len(captured) - 1, min(N_TEST, len(captured))).round().astype(int).tolist()))
        env = make_env(task, "default", 10_000 + seed, 20_000 + seed, np.random.default_rng(30_000 + seed))
        worst = 0.0
        for k in picks:
            c = captured[k]
            env.reset()
            env.set_state(copy.deepcopy(c["start"]))
            ref_obs, ref_r = _run(env, c["actions"])
            env.reset()
            env.set_state(copy.deepcopy(c["start"]))
            f64_obs, f64_r = _run(env, [np.asarray(a, dtype=np.float64) for a in c["actions"]])
            n = min(len(ref_obs), len(f64_obs))
            raw = np.linalg.norm(f64_obs[:n] - ref_obs[:n], axis=1)
            scaled = np.sqrt(np.mean(((f64_obs[:n, used] - ref_obs[:n, used]) / scale[used]) ** 2, axis=1))
            ref = ref_curves[c["t"]]
            m = min(len(ref), len(scaled))
            diff = float(np.max(np.abs(scaled[:m] - ref[:m]))) if m else 0.0
            worst = max(worst, diff, abs(len(ref) - len(scaled)))
            dtype = str(np.asarray(c["actions"][0]).dtype)
            key = f"{task}|{c['t']}"
            arrays[key + "|raw"] = raw
            arrays[key + "|scaled"] = scaled
            first_spread = int(np.argmax(scaled > 1.0) + 1) if (scaled > 1.0).any() else None
            meta["episodes"].append({"task": task, "trigger_step": int(c["t"]), "action_dtype": dtype,
                                     "stored_return": float(c["ret"]), "f32_return": float(ref_r.sum()),
                                     "f64_return": float(f64_r.sum()), "length": int(n),
                                     "first_step_above_1_spread": first_spread,
                                     "raw_at": {s: (float(raw[s - 1]) if n >= s else None) for s in (1, 10, 100, 1000)}})
        meta["check"][task] = {"max_abs_diff_scaled_vs_precision_check": worst, "episodes": len(picks)}
        print(f"float64 recompute {task}: {len(picks)} episodes, max |scaled - precision check| = {worst:.3g}",
              flush=True)
    np.savez_compressed(out, **arrays)
    meta_path.write_text(json.dumps(meta, indent=1))
    return meta


def load_f64_arrays() -> dict:
    with np.load(CACHE / "f64_raw.npz") as d:
        return {k: d[k] for k in d.files}


# ---------------------------------------------------------------- filmstrip

FILM_STEPS = (0, 50, 100, 200, 400, 800)
FILM_PX = (287, 135)  # width, height of one frame at 300 dpi (0.955 x 0.45 in)
RENDER_SCALE = 3


def pick_episode(task: str) -> dict:
    """The latest stored episode among those with the highest stored return of the task."""
    eps = pickle.load(open(REPLAY_STORE / f"episodes_{task}.pkl", "rb"))
    best = max(eps, key=lambda e: (e["stored_return"], e["total_steps"]))
    return best


def _camera(task: str, data, model, view=None):
    """Side view that follows the torso along x. `view` = (lookat height, distance), fixed per task."""
    import mujoco

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    torso = data.body("torso").xpos
    z, dist = view
    cam.lookat[:] = [torso[0], 0.0, z]
    cam.distance = dist
    cam.azimuth = 90.0
    cam.elevation = -3.0
    return cam


def _body_z_extent(model, data, floor: int) -> tuple[float, float]:
    """Lowest and highest point of the body's capsules, spheres and boxes in the current state."""
    lo, hi = np.inf, -np.inf
    for g in range(model.ngeom):
        if g == floor or model.geom_rgba[g, 3] == 0:
            continue
        z = data.geom_xpos[g, 2]
        mat = data.geom_xmat[g].reshape(3, 3)
        size = model.geom_size[g]
        kind = model.geom_type[g]
        if kind == 3:  # capsule: radius, half length along local z
            ext = abs(mat[2, 2]) * size[1] + size[0]
        elif kind == 2:  # sphere
            ext = size[0]
        elif kind == 6:  # box
            ext = float(np.abs(mat[2]) @ size)
        else:
            ext = model.geom_rbound[g]
        lo, hi = min(lo, z - ext), max(hi, z + ext)
    return lo, hi


def _render(renderer, task, model, data, view=None):
    import mujoco

    renderer.update_scene(data, camera=_camera(task, data, model, view))
    # No shadows: a plane drawn without end gives shadow-map artifacts near the origin.
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    return renderer.render().copy()


SKY = np.array([246, 246, 246], dtype=np.float64)
FLOOR_EMISSION = 0.35


def _print_friendly(model) -> None:
    """Render-only model changes for print: a plain light sky, a light low-contrast floor checker that is drawn
    without end, and no mirror reflections. Used only on a second model that never runs the physics."""
    import mujoco

    for t in range(model.ntex):
        adr = int(model.tex_adr[t])
        size = int(model.tex_width[t] * model.tex_height[t] * model.tex_nchannel[t])
        data = model.tex_data[adr:adr + size].astype(np.float64)
        if model.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            data[:] = np.resize(SKY, size)
        else:
            nch = int(model.tex_nchannel[t])
            px = data.reshape(-1, nch)
            lum = px[:, :3].mean(axis=1, keepdims=True)
            lum = (lum - lum.min()) / max(lum.max() - lum.min(), 1e-9)
            px[:, :3] = 214 + 30 * lum  # checker between two light grays
            data = px.reshape(-1)
        model.tex_data[adr:adr + size] = np.clip(data, 0, 255).astype(np.uint8)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor >= 0:
        model.geom_rgba[floor] = [1, 1, 1, 1]
        mat = int(model.geom_matid[floor])
        if mat >= 0:
            model.mat_reflectance[mat] = 0.0
            model.mat_rgba[mat] = [1, 1, 1, 1]
            model.mat_specular[mat] = 0.0
            model.mat_shininess[mat] = 0.0
            model.mat_emission[mat] = FLOOR_EMISSION
            if not model.mat_texuniform[mat]:
                # Keep the checker size of the original finite floor, in repeats per meter.
                old = model.geom_size[floor, :2].copy()
                model.mat_texrepeat[mat] = model.mat_texrepeat[mat] / (2 * old)
                model.mat_texuniform[mat] = 1
        model.geom_size[floor, :2] = 0.0  # a plane of size 0 is drawn without end
    for m in range(model.nmat):
        model.mat_reflectance[m] = 0.0
    model.vis.rgba.haze[:] = [*(SKY / 255), 1.0]
    model.vis.map.fogstart = 100.0
    model.vis.map.fogend = 200.0


def _render_states(task: str, states: dict) -> dict:
    """Second pass: draw the recorded simulator states with a render-only copy of the task model."""
    import mujoco
    from PIL import Image

    from iers.envs import make_env

    env = make_env(task, "default", 1, 0, np.random.default_rng(0))
    env.reset()
    model, data = env._mj()
    _print_friendly(model)
    w, h = FILM_PX
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, w * RENDER_SCALE)
    model.vis.global_.offheight = max(model.vis.global_.offheight, h * RENDER_SCALE)
    # One fixed camera height and distance for every frame of the task, both rows: the vertical view covers the
    # floor and the highest point the body reaches in any shown frame, with a margin.
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    lo, hi = 0.0, -np.inf
    for state in states.values():
        mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)
        a, b = _body_z_extent(model, data, floor)
        lo, hi = min(lo, a), max(hi, b)
    span = (hi - lo) * 1.12 + 0.1
    fovy = float(model.vis.global_.fovy)
    view = ((lo + hi) / 2, span / (2 * np.tan(np.radians(fovy) / 2)))
    renderer = mujoco.Renderer(model, h * RENDER_SCALE, w * RENDER_SCALE)
    out = {}
    shadow_off = True
    for key, state in states.items():
        mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)
        img = _render(renderer, task, model, data, view)
        out[key] = np.asarray(Image.fromarray(img).resize((w, h), Image.LANCZOS))
    renderer.close()
    return out


def filmstrip(force: bool = False) -> dict:
    out = CACHE / "filmstrip.npz"
    meta_path = CACHE / "filmstrip.json"
    if out.exists() and meta_path.exists() and not force:
        return json.loads(meta_path.read_text())
    import mujoco
    import pandas as pd
    from PIL import Image

    from iers.envs import denormalize, make_env
    from iers.runner import Streams

    rc_table = pd.read_csv(AGG / "replay_check.csv")
    CACHE.mkdir(parents=True, exist_ok=True)
    frames, meta = {}, {"tasks": {}}
    w, h = FILM_PX
    for task in ("HalfCheetah-v4", "walker-walk"):
        ep = pick_episode(task)
        streams = Streams.from_seed(ep["seed"])
        same = make_env(task, "default", streams.env_seed, streams.action_seed, streams.cond)
        fresh = make_env(task, "default", 900_000 + 1000 * ep["seed"] + ep["trigger"], 0, np.random.default_rng(0))
        info = {"seed": int(ep["seed"]), "trigger": int(ep["trigger"]), "stored_at_step": int(ep["total_steps"]),
                "stored_return": float(ep["stored_return"]), "rows": {}}
        states = {}
        for row, env in (("stored", same), ("fresh", fresh)):
            # First pass: the physics only, exactly as the replay check plays an episode.
            env.reset()
            if row == "stored":
                env.set_state(copy.deepcopy(ep["start_state"]))
            model, data = env._mj()
            size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)

            def grab():
                buf = np.empty(size)
                mujoco.mj_getState(model, data, buf, mujoco.mjtState.mjSTATE_INTEGRATION)
                return buf

            total, running = 0.0, {}
            if 0 in FILM_STEPS:
                states[f"{task}|{row}|0"] = grab()
                running[0] = 0.0
            ended_at = None
            for t, action in enumerate(ep["actions"], start=1):
                obs, reward, done, trunc = env.step(denormalize(action, env.max_action_value, env.min_action_value))
                total += reward
                if t in FILM_STEPS:
                    states[f"{task}|{row}|{t}"] = grab()
                    running[t] = total
                if done or trunc:
                    ended_at = t
                    break
            info["rows"][row] = {"running_return": {str(k): float(v) for k, v in running.items()},
                                 "final_return": float(total), "ended_at": ended_at}
        # Second pass: draw the recorded states with a render-only model.
        frames.update(_render_states(task, states))
        # Checks against the official replay check: the stored row must give back the stored return exactly, and
        # the fresh row must give the official fresh_01 replay return.
        off = rc_table[(rc_table.task == task) & (rc_table.seed == ep["seed"]) & (rc_table.trigger == ep["trigger"])]
        info["check"] = {
            "stored_row_equals_stored_return": info["rows"]["stored"]["final_return"] == ep["stored_return"],
            "official_fresh_01_return": float(off[off.start == "fresh_01"].replay_return.iloc[0]),
            "official_identical_return": float(off[off.start == "identical"].replay_return.iloc[0]),
            "official_fresh_01_retention": float(off[off.start == "fresh_01"].retention_ratio.iloc[0]),
            "official_fresh_retention_median_this_episode": float(off[off.start != "identical"].retention_ratio.median()),
            "fresh_01_start_distance": float(off[off.start == "fresh_01"].start_distance.iloc[0]),
        }
        meta["tasks"][task] = info
        print(task, json.dumps(info["check"]), "stored", info["rows"]["stored"]["final_return"], "fresh",
              info["rows"]["fresh"]["final_return"], flush=True)
    np.savez_compressed(out, **frames)
    meta_path.write_text(json.dumps(meta, indent=1))
    return meta


def load_frames() -> dict:
    with np.load(CACHE / "filmstrip.npz") as d:
        return {k: d[k] for k in d.files}


if __name__ == "__main__":
    force = "--force" in sys.argv
    float64_raw(force)
    filmstrip(force)
