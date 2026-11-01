"""state_mode（観測 state の持ち方）を 1 つの定義にする。

- {"kind": "lagged"}: 仮の機械の一次遅れ＋ノイズの state（今までどおり）
- {"kind": "constant", "values": [5 個], "source": "keyframes[0]"}: 全フレーム同じ定数（方策は「時計 → 姿勢」の生成器になる）

データセットの manifest.json → run.json・チェックポイントの state_mode.json → load_policy が読んで自動で包む。
定数の入れ忘れ（定数モデルに本物の state を渡す、など）はここの検査で止める。
"""
import json
import math
import os
from pathlib import Path

import numpy as np

KINDS = ("lagged", "constant")
FILE_NAME = "state_mode.json"
JOINT_DIM = 5
# stats との照合の許容（定数の mean は float32 で保存されるため）
MEAN_ATOL = 1e-4
STD_ATOL = 1e-6


class StateModeError(ValueError):
    """state_mode が無い・食い違う・stats と合わない、または定数モデルに本物の state を渡した。"""


def lagged():
    return {"kind": "lagged"}


def constant_from_choreography(choreo):
    """振付の keyframes[0].joints_rad を定数にした state_mode（定数を作る唯一の関数）。"""
    values = [float(v) for v in choreo["keyframes"][0]["joints_rad"]]
    return validate({"kind": "constant", "values": values, "source": "keyframes[0]"})


def validate(sm):
    """形を確かめる。外れたら StateModeError。正しければ（コピーした）辞書を返す。"""
    if not isinstance(sm, dict) or sm.get("kind") not in KINDS:
        raise StateModeError(f"state_mode の kind は {KINDS} のどれか: {sm!r}")
    if sm["kind"] == "lagged":
        if set(sm) != {"kind"}:
            raise StateModeError(f"lagged に余分な鍵がある: {sorted(set(sm) - {'kind'})}")
        return {"kind": "lagged"}
    extra = set(sm) - {"kind", "values", "source"}
    if extra:
        raise StateModeError(f"constant に余分な鍵がある: {sorted(extra)}")
    vals = sm.get("values")
    if not isinstance(vals, list) or len(vals) != JOINT_DIM:
        raise StateModeError(f"constant の values は {JOINT_DIM} 個のリスト")
    for v in vals:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise StateModeError("constant の values に有限でない値・数でない値がある")
    if not isinstance(sm.get("source"), str) or not sm["source"]:
        raise StateModeError("constant の source（定数の出どころ）が無い")
    return {"kind": "constant", "values": [float(v) for v in vals], "source": sm["source"]}


def constant_vector(sm):
    return np.asarray(sm["values"], dtype=np.float64)


def read_manifest_state_mode(dataset_root):
    """データセットの manifest.json の state_mode。無ければ StateModeError（旧形式は黙って lagged 扱いにしない）。"""
    p = Path(dataset_root) / "manifest.json"
    if not p.is_file():
        raise StateModeError("manifest.json が無い")
    m = json.loads(p.read_text(encoding="utf-8"))
    if "state_mode" not in m:
        raise StateModeError("manifest に state_mode が無い（旧形式。state の持ち方が分からないので止める）")
    return validate(m["state_mode"])


def write_atomic(path, sm):
    """state_mode.json を原子的に書く（一時ファイル → os.replace）。"""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(validate(sm), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def checkpoint_file(pretrained_dir):
    """pretrained_model の横（チェックポイントのフォルダ）の state_mode.json。"""
    return Path(pretrained_dir).parent / FILE_NAME


def read_checkpoint(pretrained_dir):
    p = checkpoint_file(pretrained_dir)
    if not p.is_file():
        raise StateModeError(f"{p.parent.name}/{FILE_NAME} が無い（定数の入れ忘れを止めるため、読めないときは止める）")
    return validate(json.loads(p.read_text(encoding="utf-8")))


def check_against_stats(sm, mean, std, floor):
    """正規化の stats（observation.state の mean・std）と state_mode が合うか。不一致のメッセージのリスト。
    constant: std が全部下限・mean が定数。lagged: std が全部下限ではない（定数で学習した stats を lagged と言い張れない）。"""
    errs = []
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    std = np.asarray(std, dtype=np.float64).reshape(-1)
    if mean.shape != (JOINT_DIM,) or std.shape != (JOINT_DIM,):
        return [f"stats の state の形が {JOINT_DIM} 次元でない"]
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std))):
        return ["stats の state に有限でない値（NaN・inf）がある"]
    all_floor = bool(np.all(np.abs(std - floor) <= STD_ATOL))
    if sm["kind"] == "constant":
        if not all_floor:
            errs.append("state_mode は constant なのに、stats の state の std が全部下限ではない")
        if not np.all(np.abs(mean - constant_vector(sm)) <= MEAN_ATOL):
            errs.append("state_mode は constant なのに、stats の state の mean が定数と違う")
    elif all_floor:
        errs.append("state_mode は lagged なのに、stats の state の std が全部下限（定数で学習した可能性）")
    return errs
