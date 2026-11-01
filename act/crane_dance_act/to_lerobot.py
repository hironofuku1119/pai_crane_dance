"""合成 npz（契約の形）を LeRobotDataset（v3・画像なし）に変換する。

LeRobot の API は act/.venv の lerobot 0.6.x のソースに合わせてある
（LeRobotDataset.create / add_frame / save_episode / finalize、io_utils.load_stats / write_stats）。
"""
import os

# lerobot・huggingface_hub を import する前に固定する（値は import 時に 1 回だけ読まれる）。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json  # noqa: E402
import shutil  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from crane_dance_act.synth import DATA_KEYS, check_episode, file_sha256  # noqa: E402

# 仮の値（実測で決め直す）。
# 振付で全部 0 の列（joint1・2・4）は std≈0 になり、推論で 0.01 rad ずれると正規化後の入力が約 10^6 倍になる。
# 0.035 rad（約 2°・決定済みの ±2° 相当）なら 0.01 rad のずれは約 0.29 に収まる。MEAN_STD 正規化のときだけ効く。
STD_FLOOR_RAD = 0.035
# 仮の値。
DEFAULT_REPO_ID = "local/crane_dance_synth_v1"
ENV_NAMES = ["sin_phase", "cos_phase", "progress"]
FLOOR_KEYS = ("observation.state", "action")


def build_features(contract):
    names = list(contract["joint_names"])
    return {
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "observation.environment_state": {"dtype": "float32", "shape": (len(ENV_NAMES),), "names": list(ENV_NAMES)},
        "action": {"dtype": "float32", "shape": (len(names),), "names": names},
    }


def _load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def check_admission(eps, choreography_path, manifest_path=None):
    """取り込み可否。外れたら ValueError。"""
    shas = {str(e["choreography_sha256"]) for e in eps}
    if len(shas) != 1:
        raise ValueError(f"振付の sha256 が混在している: {len(shas)} 種類")
    sha = shas.pop()
    now = file_sha256(choreography_path)
    if sha != now:
        raise ValueError("npz の振付 sha256 が、いまの choreography.json と一致しない")
    if manifest_path is not None:
        m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if m.get("choreography_sha256") != sha:
            raise ValueError("manifest の振付 sha256 が npz と一致しない")
    return sha


def check_episode_contract(ep, contract, choreo):
    """1 本の npz が今の契約・振付と合うかを見て、不一致のメッセージのリストを返す（空なら一致）。
    形と dtype（synth.check_episode）に加えて、契約版・fps・曲長・拍の時刻表・フレーム数 N・timestamp = k/fps。"""
    errs = list(check_episode(ep, contract))
    if errs:
        return errs
    tol = float(contract["tolerance"]["time_abs_s"])
    fps = int(contract["recording"]["fps"])
    song = choreo["song"]
    if str(ep["contract_version"]) != str(contract["contract_version"]):
        errs.append(f"contract_version {ep['contract_version']} が今の契約 {contract['contract_version']} と違う")
    if int(ep["fps"]) != fps:
        errs.append(f"fps {int(ep['fps'])} が契約の {fps} と違う")
    # 時刻の値と配列は、比べる前に全部有限か確かめる（NaN は「> tol」が偽になり、比較をすり抜けるため）
    for key in dict.fromkeys(("duration_s", "beat_times_s", "timestamp", *DATA_KEYS)):
        if not np.all(np.isfinite(np.asarray(ep[key], dtype=np.float64))):
            errs.append(f"{key} に有限でない値（NaN・inf）がある")
    if errs:
        return errs
    # 比較は「全部が許容内」を真の条件にする（NaN が紛れても合格にならない向き）
    duration = float(song["duration_s"])
    if not abs(float(ep["duration_s"]) - duration) <= tol:
        errs.append(f"duration_s {float(ep['duration_s'])} が振付の {duration} と違う")
    beats = np.asarray(song["beat_times_s"], dtype=np.float64)
    eb = np.asarray(ep["beat_times_s"], dtype=np.float64)
    if eb.shape != beats.shape or not np.all(np.abs(eb - beats) <= tol):
        errs.append("beat_times_s が振付の拍の時刻表と違う")
    n = int(np.floor(duration * fps + 1e-9))
    ts = np.asarray(ep["timestamp"], dtype=np.float64)
    if len(ts) != n:
        errs.append(f"フレーム数 {len(ts)} が契約の式の N = {n} と違う")
    elif not np.all(np.abs(ts - np.arange(n, dtype=np.float64) / fps) <= tol):
        errs.append("timestamp が k/fps と許容誤差を超えて違う")
    return errs


def apply_std_floor(root, floor=STD_FLOOR_RAD):
    """meta/stats.json の observation.state と action の std に下限をかける。何度呼んでも同じ結果。"""
    from lerobot.datasets.io_utils import load_stats, write_stats

    root = Path(root)
    stats = load_stats(root)
    if stats is None:
        raise FileNotFoundError("meta/stats.json が無い")
    for key in FLOOR_KEYS:
        stats[key]["std"] = np.maximum(np.asarray(stats[key]["std"], dtype=np.float64), floor)
    write_stats(stats, root)


def convert(npz_paths, contract, choreography_path, out_root, repo_id=DEFAULT_REPO_ID, manifest_path=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    npz_paths = [Path(p) for p in npz_paths]
    if not npz_paths:
        raise ValueError("npz が 0 本")
    out_root = Path(out_root)
    eps = [_load_npz(p) for p in npz_paths]
    check_admission(eps, choreography_path, manifest_path)
    choreo = json.loads(Path(choreography_path).read_text(encoding="utf-8"))
    for p, e in zip(npz_paths, eps):
        errs = check_episode_contract(e, contract, choreo)
        if errs:
            raise ValueError(f"{p.name} が契約と合わない: " + "; ".join(errs))
    if out_root.exists():
        raise FileExistsError(f"出力先が既にある: {out_root.name}")
    fps = int(contract["recording"]["fps"])
    task = contract["lerobot"]["task"]

    out_root.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_root.parent / f".{out_root.name}.tmp-{uuid.uuid4().hex[:8]}"
    try:
        ds = LeRobotDataset.create(
            repo_id, fps=fps, features=build_features(contract), root=tmp, robot_type="crane_plus", use_videos=False
        )
        try:
            for e in eps:
                st = np.asarray(e["observation_state"], dtype=np.float32)
                en = np.asarray(e["observation_environment_state"], dtype=np.float32)
                ac = np.asarray(e["action"], dtype=np.float32)
                for k in range(len(st)):
                    ds.add_frame(
                        {
                            "observation.state": st[k].copy(),
                            "observation.environment_state": en[k].copy(),
                            "action": ac[k].copy(),
                            "task": task,
                        }
                    )
                ds.save_episode()
        finally:
            ds.finalize()
        apply_std_floor(tmp, STD_FLOOR_RAD)
        if manifest_path is not None:
            shutil.copyfile(manifest_path, tmp / "manifest.json")
        os.rename(tmp, out_root)
    except BaseException:
        if tmp.exists():
            shutil.rmtree(tmp)
        raise
    return out_root
