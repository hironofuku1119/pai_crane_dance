"""ACT のチェックポイントの評価と診断の道具。シミュレータなし。

- 検証データ: validation_episodes（synth.write_varied_set と同じ生成手順で ep{start}〜 を作る）
- 方策の読み込み: load_policy（正規化はチェックポイントに保存された stats を使う）
- 開ループ: 記録済みの observation_state を入力に行動を作る
- 閉ループ（合成）: synth.lagged_state と同じ一次遅れの「仮の機械」に方策の行動を入れて state を回す。
  Gazebo の代わりではない（仮の機械）。
- 指標: キーフレーム誤差・タイミングのずれ（拍単位）・合否（仮）
- 感度: 実際の軌道の上で拍位相・progress を少しずらしたときの行動の差

方策は「(state[5], env[3]) -> 行動の塊 (L, 5) float64」の呼び出し可能なもの（policy_fns）として扱う。
"""
import os

# lerobot・huggingface_hub を import する前に固定する（値は import 時に 1 回だけ読まれる）。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from bisect import bisect_right  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from crane_dance_act import state_mode as state_mode_mod  # noqa: E402
from crane_dance_act import synth  # noqa: E402
from crane_dance_act import variants  # noqa: E402
from crane_dance_act.to_lerobot import STD_FLOOR_RAD, check_episode_contract  # noqa: E402

# 仮の値（Gazebo での評価で見直す）。どちらも精度の根拠ではなく借りた数。
KEYFRAME_TOL_RAD = 0.035  # (a) joint3・hand のキーフレーム誤差の上限（決定済みの関節値 ±2° と同じ幅）
TIMING_TOL_BEATS = 0.1  # (b) 区間ごとのタイミングのずれ（r=0.5）の上限（決定済みの時刻 ±0.1 拍と同じ幅）
MIN_AMPLITUDE_RAD = 0.1  # これ未満の振幅の区間は、タイミングの測定から除く
LEVELS = (0.5, 0.9, 0.1)  # r(t) が最初に越える水準。0.5 が判定用、0.9 は着いたか、0.1 は動き出し
JUDGE_COLUMNS = (2, 4)  # joint3, hand（joint_names の並びでの列）
OTHER_COLUMNS = (0, 1, 3)  # 振付で動かない joint1・2・4
SENS_FRACTION = 1.0 / 16.0  # 感度試験でずらす量（拍の 1/16）


# ---------------------------------------------------------------- 検証データ
def validation_episodes(contract, choreo, sha, seed, start, count):
    """検証データ。synth.write_varied_set と全く同じ手順で ep{start}〜ep{start+count-1} を作る。
    要素は {"index", "variation", "ep", "noise_seq"}。noise_seq は閉ループのノイズ用の SeedSequence
    （np.random.default_rng(noise_seq) で何度でも同じ乱数列を得られる）。"""
    if start < 0 or count < 1:
        raise ValueError("start は 0 以上・count は 1 以上")
    children = np.random.SeedSequence(seed).spawn(start + count)[start:]
    out = []
    for i, child in enumerate(children, start=start):
        rng = np.random.default_rng(child)
        var = synth.sample_variation(rng)
        ep = synth.build_episode(contract, choreo, sha, variation=var, rng=rng)
        out.append({"index": i, "variation": var, "ep": ep, "noise_seq": np.random.SeedSequence(child.entropy, spawn_key=child.spawn_key + (10**6,))})
    return out


# ---------------------------------------------------------------- 方策の読み込み
class LoadedPolicy:
    """チェックポイントから読んだ方策と前処理器・後処理器。呼ぶと行動の塊 (L, 5) float64 を返す。"""

    def __init__(self, policy, pre, post, device, state_mode=None, allow_raw_state=False):
        self.policy = policy
        self.pre = pre
        self.post = post
        self.device = device
        self.state_mode = state_mode_mod.lagged() if state_mode is None else state_mode_mod.validate(state_mode)
        self.allow_raw_state = bool(allow_raw_state)
        self._const = (
            state_mode_mod.constant_vector(self.state_mode) if self.state_mode["kind"] == "constant" else None
        )

    def __call__(self, state, env, *, raw_state=False):
        """constant のモデルは、呼び出しの state を無視して定数を入れる（自動で包む）。
        raw_state=True は「わざと本物の state を渡す」呼び方で、constant のモデルでは allow_raw_state=True で
        読み込んだときだけ通り、それ以外は StateModeError（黙って通さない）。lagged のモデルでは何も変わらない。"""
        import torch

        if self._const is not None:
            if raw_state and not self.allow_raw_state:
                raise state_mode_mod.StateModeError(
                    "state_mode が constant のモデルに本物の state を渡した（allow_raw_state=True で読み込んだときだけ許す）"
                )
            state = state if raw_state else self._const
        batch = {
            "observation.state": torch.as_tensor(np.asarray(state, dtype=np.float32)),
            "observation.environment_state": torch.as_tensor(np.asarray(env, dtype=np.float32)),
        }
        chunk = self.post(self.policy.predict_action_chunk(self.pre(batch)))
        return chunk[0].detach().cpu().numpy().astype(np.float64)


def _normalizer_state_stats(pre):
    """前処理器の正規化の部品が持つ observation.state の (mean, std)。見つからなければ None。"""
    for step in getattr(pre, "steps", []):
        ts = getattr(step, "_tensor_stats", None)
        if ts and "observation.state" in ts and "mean" in ts["observation.state"] and "std" in ts["observation.state"]:
            st = ts["observation.state"]
            return st["mean"].detach().cpu().numpy(), st["std"].detach().cpu().numpy()
    return None


def load_policy(pretrained_dir, device, *, allow_raw_state=False, assume_legacy_lagged=False):
    """ACTPolicy と前処理器・後処理器をチェックポイント（pretrained_model のフォルダ）から読む。
    正規化はチェックポイントに保存された stats（検証データの stats は使わない）。

    state_mode（pretrained_model の横の state_mode.json）を読み、constant なら呼び出しの state を無視して定数を入れる。
    state_mode.json が無い・形が壊れている・run.json の state_mode と食い違う・stats と合わないときは StateModeError で止める。
    assume_legacy_lagged=True は「state_mode.json が無い旧形式のチェックポイントを lagged として読む」と明示するときだけ
    （stats の照合は省かない）。"""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    pretrained_dir = Path(pretrained_dir)
    try:
        sm = state_mode_mod.read_checkpoint(pretrained_dir)
    except state_mode_mod.StateModeError:
        if not (assume_legacy_lagged and not state_mode_mod.checkpoint_file(pretrained_dir).exists()):
            raise
        sm = state_mode_mod.lagged()
    run_json = pretrained_dir.parents[2] / "run.json" if len(pretrained_dir.parents) > 2 else None
    if run_json is not None and run_json.is_file():
        rec = json.loads(run_json.read_text(encoding="utf-8"))
        if "state_mode" in rec and rec["state_mode"] != sm:
            raise state_mode_mod.StateModeError("チェックポイントの state_mode.json が run.json の state_mode と違う")
    cfg = PreTrainedConfig.from_pretrained(pretrained_dir)
    cfg.device = device
    policy = ACTPolicy.from_pretrained(pretrained_dir, config=cfg)
    policy.eval()
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={"device_processor": {"device": device}, "normalizer_processor": {"device": device}},
    )
    stats = _normalizer_state_stats(pre)
    if stats is None:
        raise state_mode_mod.StateModeError("チェックポイントの stats から observation.state を読めない（state_mode を照合できない）")
    errs = state_mode_mod.check_against_stats(sm, stats[0], stats[1], STD_FLOOR_RAD)
    if errs:
        raise state_mode_mod.StateModeError("; ".join(errs))
    return LoadedPolicy(policy, pre, post, device, state_mode=sm, allow_raw_state=allow_raw_state)


# ---------------------------------------------------------------- 塊のバッファ
class ChunkBuffer:
    """k をキーに行動を引く。m フレームごと（k が m の倍数）に方策を呼んで塊を受け、塊の先頭を k に対応づけて保持する。
    保持している範囲は [k0, k0 + min(m, 塊の長さ))。空・範囲外（k が戻った・飛んで外れた）のときも方策を呼ぶ。
    k が飛んでも保持範囲の中なら、その分の行動は捨てて引く（B9）。reset() で空にする（B10）。"""

    def __init__(self, m):
        if int(m) < 1:
            raise ValueError("m は 1 以上")
        self.m = int(m)
        self.calls = 0
        self.reset()

    def reset(self):
        self._chunk = None
        self._k0 = None
        self._span = 0

    def is_empty(self):
        return self._chunk is None

    def get(self, k, predict):
        """predict() は呼ばれたときだけ評価される（行動の塊 (L, 5) を返す）。"""
        k = int(k)
        need = (
            self._chunk is None
            or not (self._k0 <= k < self._k0 + self._span)
            or (k % self.m == 0 and k != self._k0)
        )
        if need:
            chunk = np.asarray(predict(), dtype=np.float64)
            if chunk.ndim != 2 or len(chunk) < self.m:
                raise ValueError(f"塊の長さ {len(chunk)} が m={self.m} より短い")
            self.calls += 1
            self._chunk, self._k0, self._span = chunk, k, self.m
        return self._chunk[k - self._k0].copy()


# ---------------------------------------------------------------- 開ループ・閉ループ
def open_loop(policy_fns, ep, m):
    """記録済みの observation_state と環境状態（変種なら変えたもの）を入力に、行動の列 (N, 5) を作る。"""
    state = np.asarray(ep["observation_state"], dtype=np.float64)
    env = np.asarray(ep["observation_environment_state"], dtype=np.float64)
    buf = ChunkBuffer(m)
    pred = np.empty_like(state)
    for k in range(len(state)):
        pred[k] = buf.get(k, lambda k=k: policy_fns(state[k], env[k]))
    return pred


def closed_loop(policy_fns, ep, m, tau_s, noise_rng, contract):
    """仮の機械での閉ループ。内部の state はノイズなし（s[0]=action_ref[0]、s[k]=s[k-1]+α(a[k-1]-s[k-1])、
    α=1-exp(-(1/fps)/τ)。a は方策の行動）。方策に渡す観測にだけ synth.observation_noise と同じ作りのノイズを足し、
    joint_limits_rad で切り詰める（joint1 には足さない）。環境状態は ep のもの（変種なら変えたもの）。
    {"action": (N,5), "state": (N,5)（ノイズなしの内部 state）} を返す。"""
    fps = int(ep["fps"])
    ref = np.asarray(ep["action"], dtype=np.float64)
    env = np.asarray(ep["observation_environment_state"], dtype=np.float64)
    n = len(ref)
    alpha = 1.0 - math.exp(-(1.0 / fps) / float(tau_s))
    lim = np.array([contract["joint_limits_rad"][nm] for nm in contract["joint_names"]], dtype=np.float64)
    noise = synth.observation_noise(noise_rng, (n, ref.shape[1]), synth.NOISE_SIGMA_RAD, synth.NOISE_CLIP_RAD)
    s = np.empty_like(ref)
    a = np.empty_like(ref)
    s[0] = ref[0]
    buf = ChunkBuffer(m)
    for k in range(n):
        obs = np.clip(s[k] + noise[k], lim[:, 0], lim[:, 1])
        a[k] = buf.get(k, lambda k=k, obs=obs: policy_fns(obs, env[k]))
        if k + 1 < n:
            s[k + 1] = s[k] + alpha * (a[k] - s[k])
    return {"action": a, "state": s}


# ---------------------------------------------------------------- 指標
def beat_interval(beats, t):
    """時刻 t を含む拍の間隔 [s]。表の外は端の間隔。"""
    i = min(max(bisect_right(beats, t) - 1, 0), len(beats) - 2)
    return float(beats[i + 1] - beats[i])


def keyframe_errors(pred, ref, choreo, fps):
    """k_i = round(time_s × fps) での |pred[k_i] − ref[k_i]|（ref はお手本の action）。
    振付の time_s は 50 Hz の格子に乗らないので、joints_rad ではなく格子点でのお手本と比べる（完全な方策で 0）。
    {"per_keyframe": (K,5), "max_judge": joint3 と hand の最大, "max_other": joint1・2・4 の最大, "per_joint_max": (5,)}"""
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError(f"pred {pred.shape} と ref {ref.shape} の形が違う")
    rows = []
    for kf in choreo["keyframes"]:
        k = min(max(int(round(float(kf["time_s"]) * fps)), 0), len(pred) - 1)
        rows.append(np.abs(pred[k] - ref[k]))
    err = np.stack(rows)
    return {
        "per_keyframe": err,
        "per_joint_max": err.max(axis=0),
        "max_judge": float(err[:, list(JUDGE_COLUMNS)].max()),
        "max_other": float(err[:, list(OTHER_COLUMNS)].max()),
    }


def _first_up_crossing(r, level, fps, k_lo):
    """r[k-1] < level <= r[k] となる最初の k を隣のフレームとの線形補間で時刻[s]にする。無ければ None。"""
    for k in range(max(k_lo, 0) + 1, len(r)):
        if r[k - 1] < level <= r[k]:
            return (k - 1 + (level - r[k - 1]) / (r[k] - r[k - 1])) / fps
    return None


def timing_offsets(pred, choreo, beats, fps):
    """姿勢が変わる区間（キーフレーム i→i+1）ごとに、振幅最大の関節の r(t)=(pred−a)/(b−a) が
    0.5・0.9・0.1 を最初に越える時刻を、お手本（区間の直線補間なので t_i + 水準×区間長）と比べる。
    探す範囲は [t_i − 区間長/2, t_{i+1} + 区間長/2]。振幅 MIN_AMPLITUDE_RAD 未満の区間は除外。
    Δt は t_ref を含む拍の間隔で割る。越えなければ reached=False・NaN。
    区間ごとの辞書のリスト: {"segment", "joint", "amplitude_rad", "t_start_s", "t_end_s", "levels": {水準: {...}}}"""
    pred = np.asarray(pred, dtype=np.float64)
    kfs = choreo["keyframes"]
    beats = [float(b) for b in beats]
    out = []
    for i in range(len(kfs) - 1):
        a_vec = np.asarray(kfs[i]["joints_rad"], dtype=np.float64)
        b_vec = np.asarray(kfs[i + 1]["joints_rad"], dtype=np.float64)
        amp = np.abs(b_vec - a_vec)
        j = int(np.argmax(amp))
        if amp[j] < MIN_AMPLITUDE_RAD:
            continue
        t0, t1 = float(kfs[i]["time_s"]), float(kfs[i + 1]["time_s"])
        length = t1 - t0
        k_lo = max(int(math.floor((t0 - 0.5 * length) * fps)), 0)
        k_hi = min(int(math.ceil((t1 + 0.5 * length) * fps)), len(pred) - 1)
        r = (pred[k_lo : k_hi + 1, j] - a_vec[j]) / (b_vec[j] - a_vec[j])
        levels = {}
        for level in LEVELS:
            t_ref = t0 + level * length
            t_pred = _first_up_crossing(r, level, fps, 0)
            if t_pred is not None:
                t_pred += k_lo / fps
            interval = beat_interval(beats, t_ref)
            if t_pred is None:
                levels[level] = {
                    "reached": False, "t_ref_s": t_ref, "t_pred_s": math.nan,
                    "dt_s": math.nan, "beat_interval_s": interval, "dt_beats": math.nan,
                }
            else:
                levels[level] = {
                    "reached": True, "t_ref_s": t_ref, "t_pred_s": t_pred,
                    "dt_s": t_pred - t_ref, "beat_interval_s": interval, "dt_beats": (t_pred - t_ref) / interval,
                }
        out.append({
            "segment": i, "joint": j, "amplitude_rad": float(amp[j]),
            "t_start_s": t0, "t_end_s": t1, "levels": levels,
        })
    return out


def judge(kf_err, offsets, keyframe_tol=KEYFRAME_TOL_RAD, timing_tol=TIMING_TOL_BEATS):
    """合否の目安（仮）。(a) joint3・hand のキーフレーム誤差 ≤ keyframe_tol、
    (b) 全区間で |Δ(r=0.5)| ≤ timing_tol 拍（r=0.5 に届かない区間があれば不合格）。"""
    dts = [seg["levels"][0.5]["dt_beats"] for seg in offsets]
    unreached = sum(1 for seg in offsets if not seg["levels"][0.5]["reached"])
    finite = [abs(x) for x in dts if not math.isnan(x)]
    max_dt = max(finite) if finite else math.nan
    a_ok = bool(kf_err["max_judge"] <= keyframe_tol)
    b_ok = bool(unreached == 0 and len(offsets) > 0 and max_dt <= timing_tol)
    return {
        "keyframe_max_judge_rad": kf_err["max_judge"],
        "keyframe_tol_rad": keyframe_tol,
        "timing_max_abs_dt_beats": max_dt,
        "timing_tol_beats": timing_tol,
        "segments": len(offsets),
        "unreached_at_0.5": unreached,
        "pass_a": a_ok,
        "pass_b": b_ok,
        "pass": bool(a_ok and b_ok),
    }


# ---------------------------------------------------------------- 重ね合わせ（閉ループ m=1 + temporal ensemble）
ENSEMBLE_COEFF = 0.01  # ACT の論文・lerobot の説明の既定（configuration_act.py の temporal_ensemble_coeff の説明）


def ensemble_closed_loop(policy_fns, ep, tau_s, noise_rng, contract, coeff=ENSEMBLE_COEFF):
    """仮の機械での閉ループ（m=1・毎フレーム方策を呼ぶ）＋ lerobot の ACTTemporalEnsembler による重ね合わせ。
    重み w_i = exp(-coeff·i)（i=0 が最も古い塊）。仮の機械・観測ノイズ・環境状態の扱いは closed_loop と同じ。
    {"action": (N,5), "state": (N,5)} を返す。"""
    import torch
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

    fps = int(ep["fps"])
    ref = np.asarray(ep["action"], dtype=np.float64)
    env = np.asarray(ep["observation_environment_state"], dtype=np.float64)
    n = len(ref)
    alpha = 1.0 - math.exp(-(1.0 / fps) / float(tau_s))
    lim = np.array([contract["joint_limits_rad"][nm] for nm in contract["joint_names"]], dtype=np.float64)
    noise = synth.observation_noise(noise_rng, (n, ref.shape[1]), synth.NOISE_SIGMA_RAD, synth.NOISE_CLIP_RAD)
    s = np.empty_like(ref)
    a = np.empty_like(ref)
    s[0] = ref[0]
    ens = None
    for k in range(n):
        obs = np.clip(s[k] + noise[k], lim[:, 0], lim[:, 1])
        chunk = np.asarray(policy_fns(obs, env[k]), dtype=np.float64)
        if ens is None:
            ens = ACTTemporalEnsembler(coeff, len(chunk))
        a[k] = ens.update(torch.as_tensor(chunk)[None])[0].numpy()
        if k + 1 < n:
            s[k + 1] = s[k] + alpha * (a[k] - s[k])
    return {"action": a, "state": s}


# ---------------------------------------------------------------- 評価の中核
def column_name(col):
    """評価の列の名前。open: open_m10 / closed: closed_m10_tau0.06 / ensemble: closed_m1_ens_tau0.06。"""
    kind = col["kind"]
    if kind == "open":
        return f"open_m{int(col['m'])}"
    if kind == "closed":
        return f"closed_m{int(col['m'])}_tau{float(col['tau'])}"
    if kind == "ensemble":
        return f"closed_m{int(col['m'])}_ens_tau{float(col['tau'])}"
    raise ValueError(f"未知の評価の列: {col!r}")


def check_columns(columns):
    """評価の列の形を確かめて（コピーして）返す。外れたら ValueError。ensemble は m=1 のみ。"""
    out, seen = [], set()
    if not columns:
        raise ValueError("評価の列が 0 個")
    for col in columns:
        kind = col.get("kind") if isinstance(col, dict) else None
        allowed = {"open": {"kind", "m"}, "closed": {"kind", "m", "tau"}, "ensemble": {"kind", "m", "tau", "coeff"}}
        if kind not in allowed:
            raise ValueError(f"評価の列の kind は open / closed / ensemble: {col!r}")
        extra = set(col) - allowed[kind]
        if extra:
            raise ValueError(f"評価の列 {kind} に未知の鍵: {sorted(extra)}")
        m = col.get("m")
        if isinstance(m, bool) or not isinstance(m, int) or m < 1:
            raise ValueError(f"評価の列の m は 1 以上の整数: {col!r}")
        c = {"kind": kind, "m": m}
        if kind != "open":
            tau = col.get("tau")
            if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not math.isfinite(tau) or tau <= 0:
                raise ValueError(f"評価の列の tau は正の数: {col!r}")
            c["tau"] = float(tau)
        if kind == "ensemble":
            if m != 1:
                raise ValueError("ensemble の列は m=1（毎フレーム方策を呼ぶ）")
            coeff = col.get("coeff", ENSEMBLE_COEFF)
            if isinstance(coeff, bool) or not isinstance(coeff, (int, float)) or not math.isfinite(coeff):
                raise ValueError(f"ensemble の coeff は有限な数: {col!r}")
            c["coeff"] = float(coeff)
        name = column_name(c)
        if name in seen:
            raise ValueError(f"評価の列が重複: {name}")
        seen.add(name)
        out.append(c)
    return out


def predict_column(policy_fns, ep, col, noise_seq, contract):
    """1 本の検証データでの 1 列ぶんの予測。{"action": (N,5), "state": (N,5)}。
    state は open なら記録済みの観測、closed / ensemble なら仮の機械の内部 state（ノイズなし）。"""
    kind = col["kind"]
    if kind == "open":
        return {"action": open_loop(policy_fns, ep, col["m"]), "state": np.asarray(ep["observation_state"], dtype=np.float64)}
    rng = np.random.default_rng(noise_seq)
    if kind == "closed":
        return closed_loop(policy_fns, ep, col["m"], col["tau"], rng, contract)
    return ensemble_closed_loop(policy_fns, ep, col["tau"], rng, contract, col.get("coeff", ENSEMBLE_COEFF))


def score_prediction(pred, ref, choreo, beats, fps):
    kf = keyframe_errors(pred, ref, choreo, fps)
    off = timing_offsets(pred, choreo, beats, fps)
    return {
        "keyframe": {"per_joint_max_rad": kf["per_joint_max"], "max_judge_rad": kf["max_judge"], "max_other_rad": kf["max_other"]},
        "timing": off,
        "judge": judge(kf, off),
    }


def summarize_column(eps):
    """列の要約（検証データの最大・合計）。eps は score_prediction の結果（index つき）のリスト。"""
    tb = [e["judge"]["timing_max_abs_dt_beats"] for e in eps]
    tb = [x for x in tb if x is not None and not math.isnan(x)]
    return {
        "keyframe_max_rad": max(e["judge"]["keyframe_max_judge_rad"] for e in eps),
        "timing_max_beats": max(tb) if tb else None,
        "unreached": int(sum(e["judge"]["unreached_at_0.5"] for e in eps)),
        "all_pass": all(e["judge"]["pass"] for e in eps),
    }


class EvalError(RuntimeError):
    """評価を始められない（契約と合わない・装置が使えない 等）。"""


@dataclass
class EvalSpec:
    checkpoint: Path  # pretrained_model のフォルダ
    columns: list = field(default_factory=lambda: [{"kind": "open", "m": 10}])
    device: str = "mps"
    count: int = 4
    start: int = 20  # 検証データの先頭の番号（学習は 0〜19）
    seed: int = 0
    env_variant: str = "full"
    contract_path: Path = Path(__file__).resolve().parents[2] / "contract" / "contract.json"
    choreography_path: Path = Path(__file__).resolve().parents[2] / "contract" / "choreography.json"
    assume_legacy_lagged: bool = False
    on_progress: object = None  # (message:str) を受ける関数


def evaluate_checkpoint(spec):
    """チェックポイントを検証データで評価する（scripts/eval_act.py・バッチから同じものを呼ぶ）。
    結果の辞書を返す（ファイルには書かない）。始められないときは EvalError。"""
    columns = check_columns(spec.columns)
    if spec.env_variant not in variants.KINDS:
        raise EvalError(f"env_variant は {variants.KINDS} のどれか: {spec.env_variant}")
    if spec.device == "mps":
        import torch

        if not torch.backends.mps.is_available():
            raise EvalError("MPS が使えない。--device cpu を明示すれば CPU で回せる")
    elif spec.device != "cpu":
        raise EvalError(f"device は mps か cpu: {spec.device}")
    contract = synth.load_contract(spec.contract_path)
    choreo = synth.load_choreography(spec.choreography_path)
    sha = synth.file_sha256(spec.choreography_path)
    fps = int(contract["recording"]["fps"])
    beats = choreo["song"]["beat_times_s"]

    vals = validation_episodes(contract, choreo, sha, spec.seed, spec.start, spec.count)
    for v in vals:
        errs = check_episode_contract(v["ep"], contract, choreo)
        if errs:
            raise EvalError(f"ep{v['index']:03d} が契約と合わない: " + "; ".join(errs))
        v["ep"] = dict(v["ep"])
        v["ep"]["observation_environment_state"] = variants.apply_env_variant(v["ep"]["observation_environment_state"], spec.env_variant)

    policy = load_policy(spec.checkpoint, spec.device, assume_legacy_lagged=spec.assume_legacy_lagged)
    t_all = time.perf_counter()
    cols = {}
    for col in columns:
        name = column_name(col)
        t0 = time.perf_counter()
        eps = []
        for v in vals:
            pred = predict_column(policy, v["ep"], col, v["noise_seq"], contract)["action"]
            eps.append({"index": v["index"], **score_prediction(pred, v["ep"]["action"], choreo, beats, fps)})
        cols[name] = {"spec": col, "episodes": eps, "all_pass": all(e["judge"]["pass"] for e in eps),
                      "summary": summarize_column(eps), "eval_sec": round(time.perf_counter() - t0, 1)}
        if spec.on_progress is not None:
            spec.on_progress(f"列 {name} 完了")
    ck = Path(spec.checkpoint)
    return {
        "checkpoint": "/".join(ck.parts[-3:]),
        "checkpoint_dir_name": ck.parent.name,
        "choreography_sha256": sha,
        "contract_version": contract["contract_version"],
        "seed": spec.seed,
        "episodes": [v["index"] for v in vals],
        "env_variant": spec.env_variant,
        "device": spec.device,
        "state_mode": policy.state_mode,
        "eval_sec": round(time.perf_counter() - t_all, 1),
        "note": "閉ループは synth.py の一次遅れの追従モデル（仮の機械）。Gazebo の代わりではない。合否の目安は仮",
        "columns": cols,
    }


# ---------------------------------------------------------------- 感度
def sensitivity(policy_fns, ep, k_list, contract):
    """実際の軌道（ep の記録済み state・環境状態）の上の k で、拍位相を ±1/16 拍分（φ に ±2π/16）、
    progress を同じ時間幅（1/16 拍の秒数 ÷ duration）ずらした入力で、行動の差（絶対値）を測る。
    差は塊の全体で見て、k ごとの値の平均と最大を返す。ずらす量は小さい（外挿させない）。"""
    state = np.asarray(ep["observation_state"], dtype=np.float64)
    env = np.asarray(ep["observation_environment_state"], dtype=np.float64)
    beats = [float(b) for b in ep["beat_times_s"]]
    fps = int(ep["fps"])
    duration = float(ep["duration_s"])
    shifts = {
        "phase_plus": ("phase", +1.0), "phase_minus": ("phase", -1.0),
        "progress_plus": ("progress", +1.0), "progress_minus": ("progress", -1.0),
    }
    acc = {name: [] for name in shifts}
    for k in k_list:
        base = np.asarray(policy_fns(state[k], env[k]), dtype=np.float64)
        sec = SENS_FRACTION * beat_interval(beats, k / fps)
        for name, (what, sign) in shifts.items():
            e = env[k].copy()
            if what == "phase":
                phi = math.atan2(e[0], e[1]) + sign * 2.0 * math.pi * SENS_FRACTION
                e[0], e[1] = math.sin(phi), math.cos(phi)
            else:
                e[2] = min(1.0, max(0.0, e[2] + sign * sec / duration))
            alt = np.asarray(policy_fns(state[k], e), dtype=np.float64)
            n = min(len(base), len(alt))
            acc[name].append(np.abs(alt[:n] - base[:n]))
    out = {"k_count": len(list(k_list)), "shift_fraction_of_beat": SENS_FRACTION}
    for name, diffs in acc.items():
        per_k_mean = np.array([d.mean() for d in diffs])
        per_joint = np.stack([d.mean(axis=0) for d in diffs])
        out[name] = {
            "mean_abs_rad": float(per_k_mean.mean()),
            "max_over_k_mean_abs_rad": float(per_k_mean.max()),
            "per_joint_mean_abs_rad": per_joint.mean(axis=0).tolist(),
        }
    return out
