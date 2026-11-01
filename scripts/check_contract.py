#!/usr/bin/env python3
"""契約表の検査器（標準ライブラリのみ・決定論・Python 3.10 以上）。

contract/contract.json・contract/choreography.json を確かめる。
「形が合うか」と「式の再計算が golden と合うか」を見る。値が正しいか（曲・振付）は見ない。

    python3 scripts/check_contract.py [--root DIR] [--require-measured]
    python3 scripts/check_contract.py [--root DIR] --npz FILE

最終行に `CONTRACT: PASS` か `CONTRACT: FAIL (n)` を出す。
`--npz FILE` は npz の鍵・型・形を標準ライブラリだけで確かめ、最終行に `NPZ: PASS` か
`NPZ: FAIL (n)` を出す（`CONTRACT:` の行は出さない）。
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import struct
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ros" / "src" / "crane_dance"))
from crane_dance import contract_math  # noqa: E402

DESCRIPTION = "契約表（contract/）の検査器"
RE_SHA256 = re.compile(r"[0-9a-f]{64}")
# npz の文字列の型表記：<U と 1〜6 桁の文字数（桁数の上限は、巨大な数字列の int() 変換で例外を出さないため）
RE_U_DESCR = re.compile(r"<U([1-9][0-9]{0,5})")
VECTOR_KEYS = (
    ("observation_state", 5),
    ("observation_environment_state", 3),
    ("action", 5),
)
NPZ_KEYS = ("observation_state", "observation_environment_state", "action", "timestamp")
# npz のキーごとの保存の型（timestamp は float64、ほかは float32）
NPZ_DTYPE = {
    "observation_state": "float32",
    "observation_environment_state": "float32",
    "action": "float32",
    "timestamp": "float64",
}
# LeRobot 側の特徴量名（ドット区切り。observation.environment_state は名前の完全一致で判定される）
NPZ_LEROBOT_NAME = {
    "observation_state": "observation.state",
    "observation_environment_state": "observation.environment_state",
    "action": "action",
    "timestamp": "timestamp",
}
META_KEYS = (
    "t0_s", "duration_s", "fps", "beat_times_s", "contract_version", "choreography_sha256",
)
META_FLOAT64 = ("beat_times_s",)
# npz.meta の形（契約の shape 欄が、この値と一致していること）
META_SHAPE = {
    "t0_s": [], "duration_s": [], "fps": [], "beat_times_s": ["M"],
    "contract_version": [], "choreography_sha256": [],
}
SHA_LENGTH = 64
TOPIC_KEYS = {"joint_target", "joint_states", "debug"}
JOINT_TARGET_QOS = {
    "reliability": "reliable", "durability": "volatile", "history": "keep_last", "depth": 10,
}
RECORDING_STR_KEYS = (
    "grid", "n_samples", "hold_rule", "time_source", "t0_rule", "grid_index", "episode_rule",
    "clock_requirement", "record_command", "record_note",
)
DESCR = {"float32": "<f4", "float64": "<f8", "int64": "<i8"}
HEADER_MAX = 10000
# npz_schema ごとに確定した npz 欄の sha256（25-A）。npz 欄を変えたら npz_schema を上げ、
# ここに新しい行を足す（既存の行は書き換えない）。変更後の値だけでは成立しない照合にするため、
# 対応表は contract.json の外（検査器）に置く。
NPZ_SCHEMA_SHA256 = {
    1: "326c9caeec7a29c01aa7d1b3c99472f1dc012c1a7dd94261aac0ba2127b06176",
}
REQUIRED_KEYS = (
    "contract_version", "joint_names", "units", "dtype_compute", "tolerance", "vectors",
    "recording", "beat_phase", "npz", "topics", "golden", "choreography_format",
    "beats_format", "measured_in_gazebo", "joint_limits_rad",
    "golden_grid", "golden_interp", "golden_t0",
    "npz_schema", "npz_schema_sha256",
)
STALE_KEYS = ("tolerance_abs", "dtype_storage")


def npz_schema_sha256(npz) -> str:
    """npz 欄の正規化した JSON（鍵を並べ替え・区切りに空白なし）の UTF-8 の sha256。"""
    text = json.dumps(npz, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load(path: Path, errs: list[str]):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        errs.append(f"{path.name}: 読めない ({e})")
        return None


def _is_num(x) -> bool:
    """数で、かつ有限（NaN・Infinity・文字列・真偽値は数としない）。"""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_pos(x) -> bool:
    return _is_num(x) and x > 0


def _is_num_list(xs) -> bool:
    return isinstance(xs, list) and all(_is_num(x) for x in xs)


def _increasing(xs, min_gap: float = 0.0) -> bool:
    return _is_num_list(xs) and all(b - a > min_gap for a, b in zip(xs, xs[1:]))


def _is_sha256(x) -> bool:
    return isinstance(x, str) and RE_SHA256.fullmatch(x) is not None


# ---------------------------------------------------------------- contract.json


def _tolerance(c: dict, errs: list[str]):
    t = c.get("tolerance")
    if not isinstance(t, dict):
        errs.append("contract.json: tolerance が辞書でない（env_abs・time_abs_s・target_abs_rad を持つ）")
        return None
    out = []
    for key in ("env_abs", "time_abs_s", "target_abs_rad"):
        if not _is_pos(t.get(key)):
            errs.append(f"contract.json: tolerance.{key} が正の有限な数でない")
        out.append(t.get(key))
    return tuple(out) if all(_is_pos(v) for v in out) else None


def _check_npz(c: dict, errs: list[str]) -> None:
    npz = c["npz"]
    keys = npz.get("keys")
    if not isinstance(keys, dict):
        errs.append("contract.json: npz.keys が辞書でない")
        return
    dims = dict(VECTOR_KEYS)
    for key in NPZ_KEYS:
        ent = keys.get(key)
        if not (isinstance(ent, dict) and isinstance(ent.get("shape"), list)
                and isinstance(ent.get("dtype"), str)):
            errs.append(f"contract.json: npz.keys.{key} が {{shape, dtype}} の形でない")
            continue
        want = ["N"] if key == "timestamp" else ["N", c["vectors"][key]["dim"]]
        if ent["shape"] != want:
            errs.append(f"contract.json: npz.keys.{key}.shape が {want} でない"
                        f"（2 つ目の次元は vectors.{key}.dim = {dims.get(key)} と一致させる）")
        if ent["dtype"] != NPZ_DTYPE[key]:
            errs.append(f"contract.json: npz.keys.{key}.dtype が {NPZ_DTYPE[key]} でない")
        if ent.get("lerobot_name") != NPZ_LEROBOT_NAME[key]:
            errs.append(f"contract.json: npz.keys.{key}.lerobot_name が"
                        f" {NPZ_LEROBOT_NAME[key]} でない")
    meta = npz.get("meta")
    if not isinstance(meta, dict):
        errs.append("contract.json: npz.meta が辞書でない")
        return
    for key in META_KEYS:
        ent = meta.get(key)
        if not (isinstance(ent, dict) and isinstance(ent.get("dtype"), str)):
            errs.append(f"contract.json: npz.meta.{key} が {{dtype, shape}} の形でない")
            continue
        if key in META_FLOAT64 and ent["dtype"] != "float64":
            errs.append(f"contract.json: npz.meta.{key}.dtype が float64 でない")
        if ent.get("shape") != META_SHAPE[key]:
            errs.append(f"contract.json: npz.meta.{key}.shape が {META_SHAPE[key]} でない")
        if key == "choreography_sha256" and ent.get("length") != SHA_LENGTH:
            errs.append(f"contract.json: npz.meta.choreography_sha256.length が {SHA_LENGTH} でない")
    if not isinstance(npz.get("layout"), str):
        errs.append("contract.json: npz.layout が文字列でない")


def _check_topics(c: dict, errs: list[str]) -> None:
    topics = c["topics"]
    if set(topics) != TOPIC_KEYS:
        errs.append(f"contract.json: topics の鍵が {sorted(TOPIC_KEYS)} ちょうどでない"
                    "（music・music_params は置かない）")
    for name, t in topics.items():
        if not isinstance(t, dict):
            errs.append(f"contract.json: topics.{name} が辞書でない")
        elif "type" in t and not (isinstance(t["type"], str)
                                  and t["type"].startswith("sensor_msgs/msg/")):
            errs.append(f"contract.json: topics.{name}.type が sensor_msgs/msg/ の形でない")
    jt = topics.get("joint_target")
    if isinstance(jt, dict) and jt.get("qos") != JOINT_TARGET_QOS:
        errs.append(f"contract.json: topics.joint_target.qos が {JOINT_TARGET_QOS} でない")


def _check_recording(rec: dict, errs: list[str]) -> None:
    for key in RECORDING_STR_KEYS:
        if not isinstance(rec.get(key), str):
            errs.append(f"contract.json: recording.{key} が文字列でない（無い）")
    topics = rec.get("record_topics")
    if not (isinstance(topics, list) and topics and all(isinstance(t, str) for t in topics)):
        errs.append("contract.json: recording.record_topics が文字列の並びでない")
    elif "/clock" in topics:
        errs.append("contract.json: recording.record_topics に /clock がある（記録しない）")


def check_structure(c: dict, errs: list[str]):
    """contract.json の形を確かめる。続きの検査に使える形なら (env_abs, time_abs_s) を返す。"""
    joints = c.get("joint_names")
    if not (isinstance(joints, list) and len(joints) == 5
            and all(isinstance(j, str) for j in joints)):
        errs.append("contract.json: joint_names が 5 つの文字列でない")
        return None
    n0 = len(errs)
    for key in REQUIRED_KEYS:
        if key not in c:
            errs.append(f"contract.json: キー {key} が無い")
    for key in STALE_KEYS:
        if key in c:
            errs.append(f"contract.json: 古い欄 {key} が残っている（tolerance・npz の dtype へ）")
    if len(errs) > n0:
        return None
    tol = _tolerance(c, errs)
    vec = c["vectors"]
    for key, dim in VECTOR_KEYS:
        if not (isinstance(vec, dict) and isinstance(vec.get(key), dict)
                and vec[key].get("dim") == dim):
            errs.append(f"contract.json: vectors.{key}.dim が {dim} でない")
    rec = c["recording"]
    fps = rec.get("fps") if isinstance(rec, dict) else None
    if not (isinstance(fps, int) and not isinstance(fps, bool) and fps > 0):
        errs.append("contract.json: recording.fps が正の整数でない")
    if not (isinstance(rec, dict) and isinstance(rec.get("n_samples"), str)):
        errs.append("contract.json: recording.n_samples が文字列の式でない")
    bp = c["beat_phase"]
    if not (isinstance(bp, dict) and _is_pos(bp.get("min_beat_interval_s"))):
        errs.append("contract.json: beat_phase.min_beat_interval_s が正の有限な数でない")
    if not (_is_int(c["npz_schema"]) and c["npz_schema"] > 0):
        errs.append("contract.json: npz_schema が正の整数でない")
    if not isinstance(c["npz"], dict):
        errs.append("contract.json: npz が辞書でない")
    elif len(errs) == n0:
        _check_npz(c, errs)
    if isinstance(c["npz"], dict):
        got = npz_schema_sha256(c["npz"])
        if got != c["npz_schema_sha256"]:
            errs.append("contract.json: npz 欄と npz_schema_sha256 が合わない"
                        "（npz を変えたら npz_schema と npz_schema_sha256 を同時に直す）")
        pinned = NPZ_SCHEMA_SHA256.get(c["npz_schema"]) if _is_int(c["npz_schema"]) else None
        if pinned is None:
            errs.append(f"contract.json: npz_schema {c['npz_schema']!r} は検査器の対応表"
                        "（NPZ_SCHEMA_SHA256）に無い（版を上げたら対応表に行を足す）")
        elif got != pinned:
            errs.append(f"contract.json: npz 欄が npz_schema {c['npz_schema']} の確定形と違う"
                        "（npz を変えたら npz_schema を上げ、NPZ_SCHEMA_SHA256 に行を足す）")
    if not isinstance(c["joint_limits_rad"], dict):
        errs.append("contract.json: joint_limits_rad が辞書でない")
    else:
        for j, lim in c["joint_limits_rad"].items():
            if j not in joints or not (
                _is_num_list(lim) and len(lim) == 2 and lim[0] < lim[1]
            ):
                errs.append(f"contract.json: joint_limits_rad.{j} が不正")
    if not isinstance(c["topics"], dict) or not isinstance(c["golden"], dict):
        errs.append("contract.json: topics か golden が辞書でない")
    else:
        _check_topics(c, errs)
    if isinstance(rec, dict):
        _check_recording(rec, errs)
    if not isinstance(c["beats_format"], dict):
        errs.append("contract.json: beats_format が辞書でない")
    lr = c.get("lerobot")
    if not (isinstance(lr, dict) and lr.get("task") == "dance_v1"
            and lr.get("episodes_per_npz") == 1 and lr.get("k_start") == 0):
        errs.append("contract.json: lerobot が task = dance_v1・1 npz = 1 エピソード・"
                    "k は 0 始まり、の形でない")
    return tol if len(errs) == n0 else None


def check_golden(c: dict, tol: float, errs: list[str]) -> int:
    g = c["golden"]
    min_gap = c["beat_phase"]["min_beat_interval_s"]
    tables = g.get("tables")
    if not isinstance(tables, dict):
        errs.append("golden: tables が辞書でない")
        return 0
    for name, tb in tables.items():
        if not (isinstance(tb, list) and len(tb) >= 2 and _increasing(tb, min_gap)):
            errs.append(f"golden: 表 {name} が有限な数の狭義単調増加（間隔 > {min_gap}）でない")
            return 0
    rows = g.get("rows")
    if not (isinstance(rows, list) and rows and _is_pos(g.get("duration_s"))):
        errs.append("golden: rows か duration_s（正の有限な数）が無い")
        return 0
    for k, r in enumerate(rows):
        if not isinstance(r, dict):
            errs.append(f"golden: rows[{k}] が辞書でない")
            continue
        if not all(_is_num(r.get(x)) for x in ("tau", "sin", "cos", "progress")):
            errs.append(f"golden: rows[{k}] の tau・sin・cos・progress が有限な数でない")
            continue
        tb = tables.get(r.get("table")) if isinstance(r.get("table"), str) else None
        if tb is None:
            errs.append(f"golden: rows[{k}] の表 {r.get('table')} が無い")
            continue
        got = contract_math.env_state(tb, r["tau"], g["duration_s"])
        want = (r["sin"], r["cos"], r["progress"])
        if any(abs(a - b) > tol for a, b in zip(got, want)):
            errs.append(f"golden: rows[{k}] (表 {r['table']}, τ={r['tau']}) が式の再計算と合わない")
    return len(rows)


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def check_golden_grid(c: dict, errs: list[str]) -> int:
    g = c["golden_grid"]
    if not (isinstance(g, dict) and _is_int(g.get("fps")) and g["fps"] > 0
            and _is_int(g.get("t0_ns")) and isinstance(g.get("cases"), list) and g["cases"]):
        errs.append("golden_grid: fps（正の整数）・t0_ns（整数）・cases（1 件以上）が無い")
        return 0
    for k, case in enumerate(g["cases"]):
        label = f"golden_grid: cases[{k}]"
        ok = (isinstance(case, dict) and isinstance(case.get("nows_ns"), list)
              and all(_is_int(x) for x in case["nows_ns"])
              and isinstance(case.get("emit"), list) and all(_is_int(x) for x in case["emit"])
              and _is_int(case.get("skipped"))
              and (case.get("invalid_at") is None or _is_int(case.get("invalid_at"))))
        if not ok:
            errs.append(f"{label} の形が不正（nows_ns・emit・skipped・invalid_at）")
            continue
        got = contract_math.grid_steps(case["nows_ns"], g["t0_ns"], g["fps"])
        want = {"emit": case["emit"], "skipped": case["skipped"],
                "invalid_at": case["invalid_at"]}
        if got != want:
            errs.append(f"{label} ({case.get('name')}) が grid_steps の結果と合わない")
    return len(g["cases"])


def _keyframes_ok(kfs, n_joints: int) -> bool:
    return (isinstance(kfs, list) and len(kfs) >= 2
            and all(isinstance(k, dict) and _is_num(k.get("time_s"))
                    and _is_num_list(k.get("joints_rad")) and len(k["joints_rad"]) == n_joints
                    for k in kfs)
            and _increasing([k["time_s"] for k in kfs]))


def check_golden_interp(c: dict, tol: float, errs: list[str]) -> int:
    g = c["golden_interp"]
    n = len(c["joint_names"])
    if not (isinstance(g, dict) and _keyframes_ok(g.get("keyframes"), n)
            and isinstance(g.get("rows"), list) and g["rows"]):
        errs.append("golden_interp: keyframes（単調増加・関節数ぶん）か rows が無い・不正")
        return 0
    for k, r in enumerate(g["rows"]):
        if not (isinstance(r, dict) and _is_num(r.get("t")) and _is_num_list(r.get("want"))
                and len(r["want"]) == n):
            errs.append(f"golden_interp: rows[{k}] の t・want が不正")
            continue
        got = contract_math.target_at(r["t"], g["keyframes"])
        if any(abs(a - b) > tol for a, b in zip(got, r["want"])):
            errs.append(f"golden_interp: rows[{k}] (t={r['t']}) が target_at と合わない")
    return len(g["rows"])


def t0_series(g: dict, kfs: list[dict], t0_ns: int, n: int) -> tuple[list[int], list[list[float]]]:
    """k = 0..n-1 の stamp（grid_time_ns）と関節目標（target_at(k / fps)）の列。"""
    fps = g["fps"]
    stamps = [contract_math.grid_time_ns(k, t0_ns, fps) for k in range(n)]
    positions = [contract_math.target_at(k / fps, kfs) for k in range(n)]
    return stamps, positions


def check_golden_t0(c: dict, tol: float, errs: list[str]) -> int:
    g = c["golden_t0"]
    gi = c["golden_interp"]
    if not (isinstance(g, dict) and _is_int(g.get("fps")) and g["fps"] > 0
            and _is_int(g.get("t0_ns")) and _is_int(g.get("n")) and g["n"] > 0
            and isinstance(g.get("cases"), list) and g["cases"]
            and isinstance(gi, dict) and _keyframes_ok(gi.get("keyframes"), len(c["joint_names"]))):
        errs.append("golden_t0: fps・t0_ns・n・cases が無い・不正（または golden_interp が不正）")
        return 0
    kf_sets = {"interp": gi["keyframes"], "same_ends": g.get("keyframes_same_ends")}
    if not _keyframes_ok(kf_sets["same_ends"], len(c["joint_names"])):
        errs.append("golden_t0: keyframes_same_ends が無い・不正")
        return 0
    for k, case in enumerate(g["cases"]):
        label = f"golden_t0: cases[{k}]"
        if not (isinstance(case, dict) and isinstance(case.get("consistent"), bool)
                and isinstance(case.get("complete"), bool)
                and case.get("keyframes", "interp") in kf_sets):
            errs.append(f"{label} の形が不正（consistent・complete が真偽でない、または keyframes が不正）")
            continue
        kfs = kf_sets[case.get("keyframes", "interp")]
        drop = case.get("drop_first", 0)
        keep = case.get("keep_last")
        second = case.get("append_second_t0_ns")
        n2 = case.get("append_second_n", g["n"])
        if not (_is_int(drop) and 0 <= drop < g["n"]
                and (keep is None or (_is_int(keep) and 0 < keep <= g["n"]))
                and (second is None or _is_int(second))
                and _is_int(n2) and 0 < n2 <= g["n"]):
            errs.append(f"{label} の drop_first・keep_last・append_second_t0_ns・append_second_n が不正")
            continue
        try:
            stamps, pos = t0_series(g, kfs, g["t0_ns"], g["n"])
            stamps, pos = stamps[drop:], pos[drop:]
            if keep is not None:
                stamps, pos = stamps[-keep:], pos[-keep:]
            if second is not None:
                s2, p2 = t0_series(g, kfs, second, n2)
                stamps, pos = stamps + s2, pos + p2
        except (TypeError, ValueError) as e:
            errs.append(f"{label}: 列を作れない ({e})")
            continue
        got = contract_math.t0_consistent(stamps, pos, kfs, tol)
        if got != case["consistent"]:
            errs.append(f"{label} ({case.get('name')}) の t0_consistent が {got}（期待 {case['consistent']}）")
        got_c = contract_math.episode_complete(stamps, g["n"], g["fps"])
        if got_c != case["complete"]:
            errs.append(f"{label} ({case.get('name')}) の episode_complete が {got_c}（期待 {case['complete']}）")
    return len(g["cases"])


# ---------------------------------------------------------------- choreography / beats


def check_sections(sections, duration_s, label: str, tol: float, errs: list[str]) -> None:
    if not isinstance(sections, list) or not sections:
        errs.append(f"{label}: sections が 1 つ以上無い")
        return
    pos = 0.0
    for k, s in enumerate(sections):
        if not isinstance(s, dict) or not all(
            key in s for key in ("start_s", "end_s", "bpm", "meter")
        ):
            errs.append(f"{label}: sections[{k}] のキーが足りない")
            return
        if not (_is_num(s["start_s"]) and _is_num(s["end_s"])):
            errs.append(f"{label}: sections[{k}] の start_s・end_s が有限な数でない")
            return
        if abs(s["start_s"] - pos) > tol:
            errs.append(f"{label}: sections[{k}] の手前に隙間か重なり")
        if not _is_pos(s["bpm"]):
            errs.append(f"{label}: sections[{k}].bpm が正でない")
        if not (isinstance(s["meter"], str) and re.fullmatch(r"\d+/\d+", s["meter"])):
            errs.append(f"{label}: sections[{k}].meter が「数/数」でない")
        pos = s["end_s"]
    if not _is_num(duration_s) or abs(pos - duration_s) > tol:
        errs.append(f"{label}: sections が duration_s まで覆っていない")


def check_choreography(c: dict, ch, require_measured: bool, tol: float,
                       errs: list[str]) -> None:
    if not isinstance(ch, dict):
        errs.append("choreography: 最上位が辞書でない")
        return
    joints = c["joint_names"]
    min_gap = c["beat_phase"]["min_beat_interval_s"]
    limits = c["joint_limits_rad"]
    if "joint_names" in ch and ch["joint_names"] != joints:
        errs.append("choreography: joint_names が contract.json と完全一致でない（名前か並び）")
    song = ch.get("song")
    if not isinstance(song, dict):
        errs.append("choreography: song が辞書でない")
        return
    dur = song.get("duration_s")
    if not _is_pos(dur):
        errs.append("choreography: song.duration_s が正の有限な数でない")
        return
    beats = song.get("beat_times_s")
    if not (isinstance(beats, list) and len(beats) >= 2 and _increasing(beats, min_gap)):
        errs.append(
            f"choreography: beat_times_s が有限な数の狭義単調増加（間隔 > {min_gap}）でない"
        )
        return
    downs = song.get("downbeat_times_s")
    if not _is_num_list(downs):
        errs.append("choreography: downbeat_times_s が有限な数の並びでない")
        return
    for d in downs:
        if not any(abs(d - b) <= tol for b in beats):
            errs.append(f"choreography: downbeat {d} が beat_times_s に無い")
    if "sections" in song:
        errs.append("choreography: song に sections を置かない（beats.json の書式だけ）")
    src = song.get("source")
    src = src if isinstance(src, dict) else {}
    if src.get("kind") not in ("provisional", "measured"):
        errs.append("choreography: song.source.kind が provisional / measured でない")
    if require_measured and src.get("kind") != "measured":
        errs.append("choreography: --require-measured なのに song.source.kind が measured でない")
    sha = src.get("audio_sha256")
    if sha is not None and not _is_sha256(sha):
        errs.append("choreography: song.source.audio_sha256 が null でも 64 桁の小文字 16 進でもない")
    elif require_measured and sha is None:
        errs.append("choreography: --require-measured なのに song.source.audio_sha256 が null")
    kfs = ch.get("keyframes")
    if not (isinstance(kfs, list) and len(kfs) >= 2 and all(isinstance(k, dict) for k in kfs)):
        errs.append("choreography: keyframes が 2 件以上の辞書の並びでない")
        return
    names = [k.get("name") for k in kfs]
    if len(set(map(str, names))) != len(names):
        errs.append("choreography: keyframes の name が重複している")
    idx = [k.get("beat_index") for k in kfs]
    if not all(isinstance(i, int) and not isinstance(i, bool) for i in idx) or not all(
        a < b for a, b in zip(idx, idx[1:])
    ):
        errs.append("choreography: beat_index が整数の単調増加でない")
        idx_ok = False
    else:
        idx_ok = True
    times = [k.get("time_s") for k in kfs]
    if not _increasing(times):
        errs.append("choreography: time_s が有限な数の単調増加でない")
    for k in kfs:
        n = k.get("name")
        t = k.get("time_s")
        i = k.get("beat_index")
        if idx_ok and _is_num(t) and 0 <= i < len(beats):
            if abs(t - beats[i]) > tol:
                errs.append(f"choreography: {n}: time_s が beat_times_s[beat_index] と一致しない")
        elif idx_ok:
            errs.append(f"choreography: {n}: beat_index が範囲外")
        if _is_num(t) and not any(abs(t - d) <= tol for d in downs):
            errs.append(f"choreography: {n}: time_s が小節頭（downbeat）のどれとも一致しない")
        jr = k.get("joints_rad")
        if not (_is_num_list(jr) and len(jr) == len(joints)):
            errs.append(
                f"choreography: {n}: joints_rad が joint_names と同じ数（{len(joints)}）の数でない"
            )
        else:
            if jr[0] != 0:
                errs.append(f"choreography: {n}: joint1 が 0 でない")
            for j, v in zip(joints, jr):
                lim = limits.get(j)
                if lim and not (lim[0] <= v <= lim[1]):
                    errs.append(f"choreography: {n}: {j}={v} が可動範囲外")
        pj = k.get("primary_joints")
        if not (isinstance(pj, list) and all(p in joints for p in pj)):
            errs.append(f"choreography: {n}: primary_joints が空か関節名のどれかでない")
        if not isinstance(k.get("measured_in_gazebo"), bool):
            errs.append(f"choreography: {n}: measured_in_gazebo が真偽でない")
        elif require_measured and not k["measured_in_gazebo"]:
            errs.append(f"choreography: {n}: --require-measured なのに measured_in_gazebo が false")
    if _is_num(times[-1]) and times[-1] > dur + tol:
        errs.append("choreography: 最後の time_s が duration_s を超える")


def check_beats(c: dict, ch, root: Path, require_measured: bool, tol: float,
                errs: list[str]) -> str:
    path = root / "act" / "measurements" / "beats.json"
    if not path.exists():
        if require_measured:
            errs.append("beats.json: --require-measured なのに act/measurements/beats.json が無い")
            return "beats: 無い"
        return "beats: 未照合"
    b = _load(path, errs)
    if b is None:
        return "beats: 読めない"
    if not isinstance(b, dict):
        errs.append("beats.json: 最上位が辞書でない")
        return "beats: 照合したが不一致あり"
    before = len(errs)
    fmt = c["beats_format"]
    for key in fmt.get("keys", []):
        if key not in b:
            errs.append(f"beats.json: キー {key} が無い")
    dur = b.get("duration_s")
    if not _is_pos(dur):
        errs.append("beats.json: duration_s が正の有限な数でない")
    check_sections(b.get("sections"), dur, "beats.json", tol, errs)
    bsrc = b.get("source")
    bsrc = bsrc if isinstance(bsrc, dict) else {}
    for key in fmt.get("source_keys", []):
        if key not in bsrc:
            errs.append(f"beats.json: source.{key} が無い")
    for key, subs in fmt.get("source_subkeys", {}).items():
        sub = bsrc.get(key)
        if isinstance(sub, dict):
            for name in subs:
                if name not in sub:
                    errs.append(f"beats.json: source.{key}.{name} が無い")
        elif key in bsrc:
            errs.append(f"beats.json: source.{key} が辞書でない")
    method = bsrc.get("downbeat_method")
    if not (isinstance(method, str) and method.strip()):
        errs.append("beats.json: source.downbeat_method（小節頭の決め方）が空")
    song = ch.get("song") if isinstance(ch, dict) else None
    song = song if isinstance(song, dict) else {}
    for key in ("beat_times_s", "downbeat_times_s"):
        a, m = song.get(key), b.get(key)
        if not (_is_num_list(a) and _is_num_list(m)):
            errs.append(f"beats.json: {key} が有限な数の並びでない（song と実測のどちらか）")
        elif len(a) != len(m) or any(abs(x - y) > tol for x, y in zip(a, m)):
            errs.append(f"beats.json: song.{key} が実測の写しと一致しない")
    if not (_is_num(dur) and abs(dur - song.get("duration_s", -1)) <= tol):
        errs.append("beats.json: song.duration_s が実測の写しと一致しない")
    sha = bsrc.get("audio_sha256")
    song_src = song.get("source")
    song_sha = song_src.get("audio_sha256") if isinstance(song_src, dict) else None
    if song_sha != sha:
        errs.append("beats.json: song.source.audio_sha256 が実測の source と一致しない")
    elif require_measured and not _is_sha256(sha):
        errs.append("beats.json: --require-measured なのに source.audio_sha256 が"
                    " 64 桁の小文字 16 進でない")
    return "beats: 照合した" if len(errs) == before else "beats: 照合したが不一致あり"


# ---------------------------------------------------------------- npz の検査（--npz）


def _read_header(zf: zipfile.ZipFile, name: str, errs: list[str], want_data: bool = False):
    """npy の先頭だけ読む。(shape, descr, data, header_total) を返す（読めなければ None）。

    data は want_data のときだけ本体の先頭（0 次元の値を読むのに足りる分）、他は b""。
    header_total は magic から改行までのバイト数（本体はその後ろから始まる）。
    """
    label = f"npz: {name}"
    try:
        with zf.open(name) as f:
            head = f.read(8)
            if len(head) < 8 or head[:6] != b"\x93NUMPY":
                errs.append(f"{label}: magic が \\x93NUMPY でない")
                return None
            ver = (head[6], head[7])
            if ver in ((1, 0),):
                lf, fmt, enc = 2, "<H", "latin1"
            elif ver == (2, 0):
                lf, fmt, enc = 4, "<I", "latin1"
            elif ver == (3, 0):
                lf, fmt, enc = 4, "<I", "utf-8"
            else:
                errs.append(f"{label}: npy の版 {ver} は (1,0)・(2,0)・(3,0) のどれでもない")
                return None
            raw = f.read(lf)
            if len(raw) < lf:
                errs.append(f"{label}: ヘッダ長が読めない")
                return None
            hlen = struct.unpack(fmt, raw)[0]
            if hlen > HEADER_MAX:
                errs.append(f"{label}: ヘッダ長 {hlen} が {HEADER_MAX} バイトを超える")
                return None
            hbytes = f.read(hlen)
            if len(hbytes) < hlen:
                errs.append(f"{label}: ヘッダが途中で切れている")
                return None
            try:
                hdr = ast.literal_eval(hbytes.decode(enc))
            except (ValueError, SyntaxError, MemoryError, RecursionError, UnicodeDecodeError):
                errs.append(f"{label}: ヘッダを読めない（辞書でない）")
                return None
            if not (isinstance(hdr, dict) and set(hdr) == {"descr", "fortran_order", "shape"}):
                errs.append(f"{label}: ヘッダの鍵が descr・fortran_order・shape ちょうどでない")
                return None
            shape = hdr["shape"]
            if not (isinstance(shape, tuple) and all(_is_int(x) for x in shape)):
                errs.append(f"{label}: shape が int の tuple でない")
                return None
            if any(x < 0 for x in shape):
                errs.append(f"{label}: shape {shape} に負の次元がある")
                return None
            if not isinstance(hdr["fortran_order"], bool):
                errs.append(f"{label}: fortran_order が bool でない")
                return None
            if not isinstance(hdr["descr"], str):
                errs.append(f"{label}: descr が文字列でない")
                return None
            total = 6 + 2 + lf + hlen
            return shape, hdr["descr"], (f.read(4 * 4096) if want_data else b""), total
    except (zipfile.BadZipFile, KeyError, OSError, RuntimeError, EOFError) as e:
        errs.append(f"{label}: 読めない ({e})")
        return None


def _itemsize(descr: str):
    """descr から 1 要素のバイト数。知らない descr は None。"""
    fixed = {"<f4": 4, "<f8": 8, "<i8": 8}
    if descr in fixed:
        return fixed[descr]
    m = RE_U_DESCR.fullmatch(descr)
    return 4 * int(m.group(1)) if m else None


def _shape_matches(want: list, shape: tuple, n_box: list) -> bool:
    """want は ["N", 5] のような宣言。"N" は npz の中で全部同じ値（n_box に覚える）。"""
    if len(want) != len(shape):
        return False
    for w, g in zip(want, shape):
        if w == "N":
            if n_box[0] is None:
                n_box[0] = g
            if g != n_box[0]:
                return False
        elif w == "M":
            if g < 0:
                return False
        elif w != g:
            return False
    return True


def check_npz(path: Path, c: dict, errs: list[str]) -> None:
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as e:
        errs.append(f"npz: zip として開けない ({e})")
        return
    with zf:
        names = zf.namelist()
        keys = list(NPZ_KEYS) + list(META_KEYS)
        expected = {k + ".npy" for k in keys}
        for name, cnt in Counter(names).items():
            if cnt > 1:
                errs.append(f"npz: 項目名 {name} が {cnt} 回ある（重複）")
        for name in sorted(set(names)):
            if "/" in name or name.endswith("/"):
                errs.append(f"npz: 項目名 {name} にディレクトリが含まれる")
            elif name not in expected:
                errs.append(f"npz: 余分な項目 {name}")
        for name in sorted(expected - set(names)):
            errs.append(f"npz: 項目 {name} が無い")
        npz, rec = c["npz"], c["recording"]
        n_box: list = [None]
        scalar: dict = {}
        for key in keys:
            name = key + ".npy"
            if name not in names:
                continue
            got = _read_header(zf, name, errs, key in ("duration_s", "fps", "contract_version"))
            if got is None:
                continue
            shape, descr, data, header_total = got
            if key in NPZ_KEYS:
                ent = npz["keys"][key]
                dtype, want_shape = ent["dtype"], ent["shape"]
            else:
                ent = npz["meta"][key]
                dtype, want_shape = ent["dtype"], ent["shape"]
            if dtype == "str":
                want_descr = "<U64" if key == "choreography_sha256" else None
                if not RE_U_DESCR.fullmatch(descr) or (want_descr and descr != want_descr):
                    errs.append(f"npz: {key} の descr {descr!r} が"
                                f" {want_descr or '<U と 1 以上の文字数の形'} でない")
                    continue
            elif descr != DESCR[dtype]:
                errs.append(f"npz: {key} の descr {descr!r} が {DESCR[dtype]!r} でない")
                continue
            if not _shape_matches(want_shape, shape, n_box):
                errs.append(f"npz: {key} の形 {shape} が宣言 {want_shape} と合わない")
                continue
            # 本体の大きさ：ヘッダの後ろに、形と型から求めたバイト数がちょうどあること（zip の非圧縮サイズで見る）
            size = _itemsize(descr)
            if size is None:  # 上の照合で弾いているはずだが、要素の大きさが決まらないまま計算しない
                errs.append(f"npz: {key} の descr {descr!r} から要素の大きさが決まらない")
                continue
            count = 1
            for d in shape:
                count *= d
            need = header_total + count * size
            have = zf.getinfo(name).file_size
            if have != need:
                errs.append(f"npz: {key} の本体の大きさ {have} バイトが、形と型から求めた {need} バイトと違う")
                continue
            if key in ("duration_s", "fps", "contract_version"):
                scalar[key] = (descr, data)
        _check_npz_values(c, rec, n_box[0], scalar, errs)


def _check_npz_values(c: dict, rec: dict, n, scalar: dict, errs: list[str]) -> None:
    vals: dict = {}
    for key, fmt, size in (("duration_s", "<d", 8), ("fps", "<q", 8)):
        if key in scalar:
            data = scalar[key][1]
            if data is None or len(data) < size:
                errs.append(f"npz: {key} の値が読めない")
            else:
                vals[key] = struct.unpack(fmt, data[:size])[0]
    if "contract_version" in scalar:
        descr, data = scalar["contract_version"]
        m = RE_U_DESCR.fullmatch(descr)
        if not m or data is None or len(data) < 4 * int(m.group(1)):
            errs.append("npz: contract_version の値が読めない")
        else:
            try:
                text = data[:4 * int(m.group(1))].decode("utf-32-le").rstrip("\x00")
            except UnicodeDecodeError:
                errs.append("npz: contract_version が UTF-32 として読めない")
            else:
                if text != c["contract_version"]:
                    errs.append(f"npz: contract_version {text!r} が契約の {c['contract_version']!r}"
                                " と違う")
    if "fps" in vals and vals["fps"] != rec["fps"]:
        errs.append(f"npz: fps {vals['fps']} が契約の {rec['fps']} と違う")
    if "duration_s" in vals:
        d = vals["duration_s"]
        if not (math.isfinite(d) and d > 0):
            errs.append(f"npz: duration_s {d} が正の有限な数でない")
            vals.pop("duration_s")
    if n is not None and n < 1:
        errs.append(f"npz: N={n} が 1 以上でない")
    if "duration_s" in vals and "fps" in vals and n is not None:
        if n != math.floor(vals["duration_s"] * vals["fps"] + 1e-9):
            errs.append(f"npz: N={n} が floor(duration_s * fps + 1e-9) と違う")


def run_npz(root: Path, npz_path: Path) -> int:
    errs: list[str] = []
    c = _load(root / "contract" / "contract.json", errs)
    if isinstance(c, dict):
        if check_structure(c, errs) is not None:
            check_npz(npz_path, c, errs)
    elif c is not None:
        errs.append("contract.json: 最上位が辞書でない")
    for e in errs:
        print(e)
    print(f"NPZ: {'FAIL (%d)' % len(errs) if errs else 'PASS'}")
    return 1 if errs else 0


# ---------------------------------------------------------------- 入口


def run(root: Path, require_measured: bool) -> int:
    errs: list[str] = []
    cdir = root / "contract"
    c = _load(cdir / "contract.json", errs)
    ch = _load(cdir / "choreography.json", errs)
    if isinstance(c, dict):
        tol = check_structure(c, errs)
        if tol is not None:
            n = check_golden(c, tol[0], errs)
            print(f"golden: {n} 行を再計算")
            print(f"golden_grid: {check_golden_grid(c, errs)} ケース")
            print(f"golden_interp: {check_golden_interp(c, tol[2], errs)} 行")
            print(f"golden_t0: {check_golden_t0(c, tol[2], errs)} ケース")
            if ch is not None:
                check_choreography(c, ch, require_measured, tol[1], errs)
                print(check_beats(c, ch, root, require_measured, tol[1], errs))
    elif c is not None:
        errs.append("contract.json: 最上位が辞書でない")
    for e in errs:
        print(e)
    print(f"CONTRACT: {'FAIL (%d)' % len(errs) if errs else 'PASS'}")
    return 1 if errs else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=DESCRIPTION)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--require-measured", action="store_true")
    ap.add_argument("--npz", type=Path, metavar="FILE",
                    help="npz の鍵・型・形を確かめる（NPZ: PASS / FAIL を出す）")
    a = ap.parse_args(argv)
    if a.npz:
        return run_npz(a.root, a.npz)
    return run(a.root, a.require_measured)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
