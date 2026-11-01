"""振付から合成の 1 エピソード（npz）を作る。値は契約と振付のファイルから読む。

注意: 合成データなので、observation_state は実機・シミュレータの実測ではない。作り方は 2 通り。
- variation=None（既定）: 「1 フレーム前の目標に追従できた」と仮定した値
  （observation_state[k] = action[k-1]、k=0 は action[0]）。
- variation あり（write_varied_set）: action の一次遅れ（s[0]=a[0]、s[k]=s[k-1]+α(a[k-1]-s[k-1])、
  α=1-exp(-(1/fps)/τ)、τ はエピソードごとに乱数）に観測ノイズ（正規分布を ±NOISE_CLIP_RAD で切り詰め・
  joint1 には足さない）を足し、joint_limits_rad で切り詰めた値。使った値は manifest.json に記録する。
どちらのモードでも action は振付どおりで変えない。
"""
import hashlib
import json
import math
from bisect import bisect_right
from pathlib import Path

import numpy as np

DATA_KEYS = ("observation_state", "observation_environment_state", "action", "timestamp")
META_KEYS = ("t0_s", "duration_s", "fps", "beat_times_s", "contract_version", "choreography_sha256")


def load_contract(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_choreography(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def target_at(t, keyframes):
    """時刻 t の目標姿勢。キーフレームの time_s を到達時刻とし、間は線形補間。前後は端の姿勢。float64。"""
    times = [float(k["time_s"]) for k in keyframes]
    poses = [np.asarray(k["joints_rad"], dtype=np.float64) for k in keyframes]
    if t <= times[0]:
        return poses[0].copy()
    if t >= times[-1]:
        return poses[-1].copy()
    i = bisect_right(times, t) - 1
    w = (t - times[i]) / (times[i + 1] - times[i])
    return poses[i] + (poses[i + 1] - poses[i]) * w


def phase(beats, tau):
    """契約 beat_phase の式。拍位相 φ[rad]。表の前後は端の拍間隔で延ばす。"""
    i = bisect_right(beats, tau) - 1
    if i < 0:
        edge, interval = beats[0], beats[1] - beats[0]
    elif i >= len(beats) - 1:
        edge, interval = beats[-1], beats[-1] - beats[-2]
    else:
        return 2 * math.pi * (tau - beats[i]) / (beats[i + 1] - beats[i])
    x = (tau - edge) / interval
    return 2 * math.pi * (x - math.floor(x))


def progress(tau, duration_s):
    return min(1.0, max(0.0, tau / duration_s))


# --- 観測側のばらつき。action は変えない。以下は全部仮の値（実測で決め直す）---
# ばらつきは observation_state 側だけ。関節値 ±2° の範囲内に置いた仮の値。
TAU_RANGE_S = (0.02, 0.10)  # 追従遅れの時定数 τ[s]（エピソードごとに一様乱数）
NOISE_SIGMA_RAD = 0.0175  # 1°
NOISE_CLIP_RAD = 0.035  # 2°（決定済みの ±2°）
NOISE_SKIP_COLUMNS = (0,)  # joint1 にはノイズを足さない（std の下限で守る）


def sample_variation(rng, tau_range_s=None, sigma_rad=None, clip_rad=None):
    """1 エピソード分のばらつきの値を決める（τ を一様乱数）。実際に使う値の辞書を返す。
    引数を省くと既定（TAU_RANGE_S・NOISE_SIGMA_RAD・NOISE_CLIP_RAD）。乱数の引き方は引数に依らず同じ。"""
    lo, hi = TAU_RANGE_S if tau_range_s is None else tau_range_s
    return {
        "tau_s": float(rng.uniform(lo, hi)),
        "noise_sigma_rad": NOISE_SIGMA_RAD if sigma_rad is None else float(sigma_rad),
        "noise_clip_rad": NOISE_CLIP_RAD if clip_rad is None else float(clip_rad),
    }


def lagged_state(action64, fps, tau_s):
    """action の一次遅れ。s[0]=a[0]、s[k]=s[k-1]+α(a[k-1]-s[k-1])、α=1-exp(-(1/fps)/τ)。float64。"""
    alpha = 1.0 - math.exp(-(1.0 / fps) / tau_s)
    s = np.empty_like(action64)
    s[0] = action64[0]
    for k in range(1, len(action64)):
        s[k] = s[k - 1] + alpha * (action64[k - 1] - s[k - 1])
    return s


def observation_noise(rng, shape, sigma, clip):
    """正規分布 σ を ±clip で切り詰めたノイズ。NOISE_SKIP_COLUMNS の列は 0。float64。"""
    noise = np.clip(rng.normal(0.0, sigma, size=shape), -clip, clip)
    for c in NOISE_SKIP_COLUMNS:
        noise[:, c] = 0.0
    return noise


def build_episode(contract, choreo, choreo_sha256, variation=None, rng=None):
    fps = int(contract["recording"]["fps"])
    duration = float(choreo["song"]["duration_s"])
    beats = [float(b) for b in choreo["song"]["beat_times_s"]]
    n = int(math.floor(duration * fps + 1e-9))
    timestamp = np.arange(n, dtype=np.float64) / fps
    action64 = np.stack([target_at(float(t), choreo["keyframes"]) for t in timestamp])
    action = action64.astype(np.float32)
    if variation is None:
        state = np.empty_like(action)
        state[0] = action[0]
        state[1:] = action[:-1]
    else:
        if rng is None:
            raise ValueError("variation を指定するときは rng も必要")
        state64 = lagged_state(action64, fps, float(variation["tau_s"]))
        state64 = state64 + observation_noise(
            rng, state64.shape, float(variation["noise_sigma_rad"]), float(variation["noise_clip_rad"])
        )
        lim = np.array([contract["joint_limits_rad"][nm] for nm in contract["joint_names"]], dtype=np.float64)
        state = np.clip(state64, lim[:, 0], lim[:, 1]).astype(np.float32)
    env = np.array(
        [[math.sin(phase(beats, float(t))), math.cos(phase(beats, float(t))), progress(float(t), duration)]
         for t in timestamp],
        dtype=np.float64,
    ).astype(np.float32)
    return {
        "observation_state": state,
        "observation_environment_state": env,
        "action": action,
        "timestamp": timestamp,
        "t0_s": np.float64(0.0),
        "duration_s": np.float64(duration),
        "fps": np.int64(fps),
        "beat_times_s": np.asarray(beats, dtype=np.float64),
        "contract_version": np.str_(contract["contract_version"]),
        "choreography_sha256": np.str_(choreo_sha256),
    }


def check_episode(ep, contract):
    """契約 npz.keys / npz.meta と照合し、不一致のメッセージのリストを返す（空なら一致）。"""
    errs = []
    spec = contract["npz"]
    n = None
    if "timestamp" in ep:
        n = int(np.asarray(ep["timestamp"]).shape[0])
    for key, s in spec["keys"].items():
        if key not in ep:
            errs.append(f"{key}: 鍵が無い")
            continue
        a = np.asarray(ep[key])
        want = tuple(n if d == "N" else d for d in s["shape"])
        if a.shape != want:
            errs.append(f"{key}: shape {a.shape} != {want}")
        if a.dtype != np.dtype(s["dtype"]):
            errs.append(f"{key}: dtype {a.dtype} != {s['dtype']}")
    for key, s in spec["meta"].items():
        if key not in ep:
            errs.append(f"{key}: メタの鍵が無い")
            continue
        a = np.asarray(ep[key])
        if s["dtype"] == "str":
            if a.dtype.kind != "U" or a.ndim != 0:
                errs.append(f"{key}: 0 次元の文字列でない ({a.dtype}, ndim={a.ndim})")
        else:
            if a.dtype != np.dtype(s["dtype"]):
                errs.append(f"{key}: dtype {a.dtype} != {s['dtype']}")
            if key != "beat_times_s" and a.ndim != 0:
                errs.append(f"{key}: 0 次元でない")
    return errs


def save_npz(path, ep, contract):
    errs = check_episode(ep, contract)
    if errs:
        raise ValueError("契約と不一致: " + "; ".join(errs))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.savez(f, **ep)


def write_varied_set(contract, choreo, choreo_sha256, episodes, seed, out_dir, tau_range_s=None, sigma_rad=None, clip_rad=None, state_mode=None):
    """ばらつきありの npz を episodes 本と manifest.json を out_dir に書く。out_dir が既にあれば止まる。
    ばらつきの範囲は引数で渡せる（省くと既定）。state_mode が constant なら observation_state を全フレーム定数にする
    （manifest には必ず state_mode を書く。省くと lagged）。"""
    from crane_dance_act import state_mode as sm_mod

    sm = sm_mod.lagged() if state_mode is None else sm_mod.validate(state_mode)
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"出力先が既にある: {out_dir.name}")
    children = np.random.SeedSequence(seed).spawn(episodes)
    out_dir.mkdir(parents=True)
    items = []
    for i, child in enumerate(children):
        rng = np.random.default_rng(child)
        var = sample_variation(rng, tau_range_s, sigma_rad, clip_rad)
        ep = build_episode(contract, choreo, choreo_sha256, variation=var, rng=rng)
        if sm["kind"] == "constant":
            ep["observation_state"] = np.tile(sm_mod.constant_vector(sm).astype(np.float32), (len(ep["timestamp"]), 1))
        save_npz(out_dir / f"ep{i:03d}.npz", ep, contract)
        items.append({"file": f"ep{i:03d}.npz", "spawn_index": i, "tau_s": var["tau_s"]})
    manifest = {
        "contract_version": contract["contract_version"],
        "choreography_sha256": choreo_sha256,
        "seed": seed,
        "episodes": episodes,
        "variation": {
            "kinds": ["observation_state の一次遅れ（τ はエピソードごとに一様乱数）", "observation_state への切り詰め正規ノイズ"],
            "action": "振付どおり（ばらつきなし）",
            "tau_range_s": list(TAU_RANGE_S if tau_range_s is None else tau_range_s),
            "noise_sigma_rad": NOISE_SIGMA_RAD if sigma_rad is None else float(sigma_rad),
            "noise_clip_rad": NOISE_CLIP_RAD if clip_rad is None else float(clip_rad),
            "noise_skip_columns": list(NOISE_SKIP_COLUMNS),
            "status": "仮の値。実測で決め直す",
        },
        "state_mode": sm,
        "items": items,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
