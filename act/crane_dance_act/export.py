"""方策の出力の書き出し。主は契約の npz、従は JointTrajectory の 1 行 YAML。

- 軌道の時刻の割り当ては trajectory_times_ns の 1 つだけ（t_k = k/fps + offset・既定 offset 0）。YAML も npz もこれを使う。
  以前の試作は (k+1)/fps で、お手本に対して 1 フレーム遅れていた。
- 書いた npz は「方策の出力」であって記録（bag から作ったもの）ではない。同じフォルダの export.json にその印を書く。
- 「閉ループ」の出力は仮の機械の中で回した指令の記録であり、Gazebo の閉ループではない。
"""
import hashlib
import json
import math
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from crane_dance_act import evaluate, synth, variants
from crane_dance_act.to_lerobot import check_episode_contract

YAML_RATE_HZ = 25
YAML_BYTE_LIMIT = 131072  # ros2 の引数 1 個の上限（試作で確かめた値）
KIND_LABEL = "policy_output（記録ではない）"
CLOSED_NOTE = "仮の機械（一次遅れ τ）の中で回した指令の記録であり、Gazebo の閉ループではない"


class ExportError(RuntimeError):
    """書かない（NaN・関節範囲外・契約と合わない・出力先が既にある 等）。"""


# ---------------------------------------------------------------- 時刻の割り当て（唯一の定義）
def trajectory_times_ns(n, fps, offset_s=0.0):
    """t_k = offset + k/fps を整数ナノ秒で返す（長さ n の list[int]）。fps は 1e9 を割り切ること（契約と同じ）。"""
    fps = int(fps)
    if fps < 1 or 10**9 % fps != 0:
        raise ValueError(f"fps {fps} は 1e9 を割り切る正の整数でない")
    if not math.isfinite(offset_s):
        raise ValueError("offset_s が有限でない")
    step = 10**9 // fps
    off = int(round(float(offset_s) * 1e9))
    return [off + k * step for k in range(int(n))]


def trajectory_times_s(n, fps, offset_s=0.0):
    """同じ割り当ての秒（float64）。"""
    return np.asarray(trajectory_times_ns(n, fps, offset_s), dtype=np.float64) / 1e9


# ---------------------------------------------------------------- 検査
def check_action(action, contract):
    """NaN・inf・関節範囲外のメッセージのリスト（空なら書いてよい）。"""
    a = np.asarray(action)
    if a.ndim != 2 or a.shape[1] != len(contract["joint_names"]):
        return [f"action の形が (N, {len(contract['joint_names'])}) でない: {a.shape}"]
    if not np.all(np.isfinite(a)):
        return ["action に有限でない値（NaN・inf）がある"]
    lim = np.array([contract["joint_limits_rad"][nm] for nm in contract["joint_names"]], dtype=np.float64)
    if not (np.all(a >= lim[:, 0]) and np.all(a <= lim[:, 1])):
        return ["action に関節範囲の外の値がある"]
    return []


def _atomic_write_bytes(path, data):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


# ---------------------------------------------------------------- npz（主）
def build_npz_episode(action, observation_state, environment_state, contract, choreo, choreography_sha256, offset_s=0.0):
    """契約の平らな 10 鍵の辞書を作る。action は方策の出力、observation_state は記録済み（開ループ）か仮の機械の state（閉ループ）。
    timestamp は trajectory_times_s（契約は t0 からの相対秒＝k/fps なので offset は 0 のみ）。"""
    if offset_s != 0.0:
        raise ExportError("npz の timestamp は契約で k/fps（t0 からの相対秒）なので offset は 0 のみ")
    errs = check_action(action, contract)
    if errs:
        raise ExportError("; ".join(errs))
    fps = int(contract["recording"]["fps"])
    act32 = np.asarray(action, dtype=np.float32)
    errs = check_action(act32.astype(np.float64), contract)
    if errs:
        raise ExportError("float32 に直した後: " + "; ".join(errs))
    n = len(act32)
    ep = {
        "observation_state": np.asarray(observation_state, dtype=np.float32),
        "observation_environment_state": np.asarray(environment_state, dtype=np.float32),
        "action": act32,
        "timestamp": trajectory_times_s(n, fps, offset_s),
        "t0_s": np.float64(0.0),
        "duration_s": np.float64(float(choreo["song"]["duration_s"])),
        "fps": np.int64(fps),
        "beat_times_s": np.asarray([float(b) for b in choreo["song"]["beat_times_s"]], dtype=np.float64),
        "contract_version": np.str_(contract["contract_version"]),
        "choreography_sha256": np.str_(choreography_sha256),
    }
    errs = check_episode_contract(ep, contract, choreo)
    if errs:
        raise ExportError("契約と合わない: " + "; ".join(errs))
    return ep


def write_npz(path, ep, contract):
    """契約どおりの npz を原子的に書く（一時ファイル → os.replace）。既にあれば止まる。"""
    path = Path(path)
    if path.exists():
        raise ExportError(f"既にある: {path.name}")
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    try:
        synth.save_npz(tmp, ep, contract)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


# ---------------------------------------------------------------- YAML（従）
def _num(x):
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def trajectory_yaml(joint_names, columns, action, fps, offset_s=0.0, rate_hz=YAML_RATE_HZ):
    """JointTrajectory の 1 行 YAML。rate_hz に間引き（k = 0, step, 2·step, …）し、最後のフレームは必ず残す。
    time_from_start は trajectory_times_ns（npz と同じ割り当て）。"""
    fps = int(fps)
    if fps % rate_hz != 0:
        raise ExportError(f"fps {fps} が {rate_hz} Hz で割り切れない")
    action = np.asarray(action, dtype=np.float64)
    n = len(action)
    step = fps // rate_hz
    ks = list(range(0, n, step))
    if ks[-1] != n - 1:
        ks.append(n - 1)
    t_ns = trajectory_times_ns(n, fps, offset_s)
    pts = []
    for k in ks:
        pos = ", ".join(_num(float(action[k, c])) for c in columns)
        pts.append(f"{{positions: [{pos}], time_from_start: {{sec: {t_ns[k] // 10**9}, nanosec: {t_ns[k] % 10**9}}}}}")
    return "{joint_names: [" + ", ".join(joint_names) + "], points: [" + ", ".join(pts) + "]}\n"


def yaml_parts(contract):
    names = list(contract["joint_names"])
    return (("arm", names[:4], list(range(4))), ("gripper", names[4:], [4]))


# ---------------------------------------------------------------- 書き出しの中核
@dataclass
class ExportSpec:
    checkpoint: Path  # pretrained_model のフォルダ
    out_dir: Path  # 新しく作る（既にあれば止まる）
    mode: str = "open"  # open / closed
    m: int = 10
    tau: float = 0.06  # closed のときの仮の機械の時定数
    formats: tuple = ("npz", "yaml")
    device: str = "mps"
    index: int = 20  # 検証データの番号（学習は 0〜19）
    seed: int = 0
    env_variant: str = "full"
    contract_path: Path = Path(__file__).resolve().parents[2] / "contract" / "contract.json"
    choreography_path: Path = Path(__file__).resolve().parents[2] / "contract" / "choreography.json"
    assume_legacy_lagged: bool = False


def write_export(spec):
    """検証データ 1 本について方策の出力を書き出す。out_dir 全体を一時フォルダ → rename で原子的に作る。
    export.json の内容を返す。書けないときは ExportError。"""
    if spec.mode not in ("open", "closed"):
        raise ExportError(f"mode は open か closed: {spec.mode}")
    bad = sorted(set(spec.formats) - {"npz", "yaml"})
    if bad or not spec.formats:
        raise ExportError(f"formats は npz / yaml: {list(spec.formats)}")
    out_dir = Path(spec.out_dir)
    if out_dir.exists():
        raise ExportError(f"出力先が既にある: {out_dir.name}")
    if spec.env_variant not in variants.KINDS:
        raise ExportError(f"env_variant は {variants.KINDS}: {spec.env_variant}")
    contract = synth.load_contract(spec.contract_path)
    choreo = synth.load_choreography(spec.choreography_path)
    sha = synth.file_sha256(spec.choreography_path)
    fps = int(contract["recording"]["fps"])

    v = evaluate.validation_episodes(contract, choreo, sha, spec.seed, spec.index, 1)[0]
    ep = dict(v["ep"])
    ep["observation_environment_state"] = variants.apply_env_variant(ep["observation_environment_state"], spec.env_variant)
    policy = evaluate.load_policy(spec.checkpoint, spec.device, assume_legacy_lagged=spec.assume_legacy_lagged)
    col = {"kind": "open", "m": spec.m} if spec.mode == "open" else {"kind": "closed", "m": spec.m, "tau": spec.tau}
    pred = evaluate.predict_column(policy, ep, col, v["noise_seq"], contract)
    action, state = np.asarray(pred["action"], dtype=np.float64), pred["state"]
    errs = check_action(action, contract)
    if errs:
        raise ExportError("書かない: " + "; ".join(errs))

    tmp = out_dir.parent / f".{out_dir.name}.tmp-{uuid.uuid4().hex[:8]}"
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp.mkdir()
        files = {}
        if "npz" in spec.formats:
            e = build_npz_episode(action, state, ep["observation_environment_state"], contract, choreo, sha)
            write_npz(tmp / "trajectory.npz", e, contract)
            files["npz"] = ["trajectory.npz"]
        if "yaml" in spec.formats:
            files["yaml"] = []
            for part, names, cols in yaml_parts(contract):
                text = trajectory_yaml(names, cols, action, fps)
                if len(text.encode()) > YAML_BYTE_LIMIT:
                    raise ExportError(f"{part} の YAML が引数 1 個の上限（{YAML_BYTE_LIMIT} バイト）を超える")
                _atomic_write_bytes(tmp / f"{part}_{YAML_RATE_HZ}hz.yaml", text.encode("utf-8"))
                files["yaml"].append(f"{part}_{YAML_RATE_HZ}hz.yaml")
        meta = {
            "kind": KIND_LABEL,
            "checkpoint": Path(spec.checkpoint).parent.name,
            "mode": spec.mode,
            "m": spec.m,
            "tau_s": spec.tau if spec.mode == "closed" else None,
            "state_mode": policy.state_mode,
            "choreography_sha256": sha,
            "contract_version": contract["contract_version"],
            "validation_episode": v["index"],
            "env_variant": spec.env_variant,
            "frames": len(action),
            "time_assignment": "t_k = k/fps（offset 0）",
            "files": files,
            "file_sha256": {f: hashlib.sha256((tmp / f).read_bytes()).hexdigest() for fs in files.values() for f in fs},
            "note": CLOSED_NOTE if spec.mode == "closed" else "開ループ：記録済みの観測を方策に入れた出力",
        }
        _atomic_write_bytes(tmp / "export.json", (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        os.rename(tmp, out_dir)
    except BaseException:
        if tmp.exists():
            shutil.rmtree(tmp)
        raise
    return meta
