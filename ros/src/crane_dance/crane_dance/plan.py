"""再生の計画（ROS に依存しない純関数・標準ライブラリのみ・Python 3.10 以上）。

お手本（振付ファイル）か npz の action から、格子番号 k ごとの関節目標を返す。
"""
from __future__ import annotations

import math
import os
import re

from crane_dance import contract_math
from crane_dance.npz_io import rows


class Plan:
    def __init__(self, fps: int, n: int, kind: str, joint_names: list, targets,
                 heads: dict, warnings=None) -> None:
        self.fps = fps
        self.n = n
        self.kind = kind
        self.joint_names = list(joint_names)
        self._targets = targets
        self.bar_heads = heads
        self.warnings = list(warnings or [])

    def target(self, k: int) -> list:
        if not isinstance(k, int) or isinstance(k, bool) or not 0 <= k < self.n:
            raise IndexError(f"k={k} は 0 以上 {self.n} 未満でない")
        return self._targets(k)

    def initial(self) -> list:
        return self.target(0)


def bar_heads(choreo: dict, fps: int) -> dict:
    """downbeat_times_s の i 番目 d を k = round(d * fps) に写す。小節番号は i（0 は前奏）。"""
    return {round(d * fps): i for i, d in enumerate(choreo["song"]["downbeat_times_s"])}


def from_choreography(contract: dict, choreo: dict) -> Plan:
    fps = contract["recording"]["fps"]
    n = math.floor(choreo["song"]["duration_s"] * fps + 1e-9)
    keyframes = choreo["keyframes"]

    def targets(k: int) -> list:
        return [float(x) for x in contract_math.target_at(k / fps, keyframes)]

    return Plan(fps, n, "お手本（振付ファイル）", contract["joint_names"], targets,
                bar_heads(choreo, fps))


def _kind_from_name(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    if "closed" in stem:
        return "閉ループの出力"
    if "open" in stem:
        return "開ループの出力"
    return "種類不明"


_DESCR = {"float32": "<f4", "float64": "<f8", "int64": "<i8"}
_RE_U = re.compile(r"<U([1-9][0-9]{0,5})")


def check_npz(contract: dict, npz: dict, choreo_sha256: str, choreo: dict) -> list:
    """npz を契約と今の振付に照合する（scripts/check_contract.py --npz の規則＋時刻と曲長）。

    違反があれば ValueError（全部の理由をつなげて）。contract_version の違いだけは警告として返す
    （判断：npz の形が同じなら再生に支障がない。--npz の検査器は厳密なまま）。
    鍵・型・形・N・fps・振付の sha・数の有限性・関節の可動範囲に加え、
    timestamp[k] = k / fps（全ゼロ・ずれ・不連続を拒む）と、duration_s・N が今の振付の曲長と
    合うこと（許容は contract の tolerance.time_abs_s）を見る。
    """
    errs: list = []
    warns: list = []
    keys, meta = contract["npz"]["keys"], contract["npz"]["meta"]
    want = set(keys) | set(meta)
    have = set(npz)
    if have != want:
        raise ValueError(f"npz の鍵が契約の {len(want)} 個と違う（足りない {sorted(want - have)}・"
                         f"余分 {sorted(have - want)}）")
    n_seen = None
    for key, spec in list(keys.items()) + list(meta.items()):
        ent = npz[key]
        descr, shape = ent["descr"], tuple(ent["shape"])
        if spec["dtype"] == "str":
            m = _RE_U.fullmatch(descr)
            if not m or (key == "choreography_sha256" and descr != "<U64"):
                errs.append(f"{key} の型 {descr} が契約と違う")
                continue
        elif descr != _DESCR.get(spec["dtype"]):
            errs.append(f"{key} の型 {descr} が契約の {spec['dtype']} と違う")
            continue
        want_shape = spec["shape"]
        if len(want_shape) != len(shape):
            errs.append(f"{key} の形 {shape} が契約の {want_shape} と違う")
            continue
        for w, g in zip(want_shape, shape):
            if w == "N":
                if n_seen is None:
                    n_seen = g
                if g != n_seen:
                    errs.append(f"{key} の行数 {g} が他のデータの {n_seen} と違う")
            elif w != "M" and w != g:
                errs.append(f"{key} の形 {shape} が契約の {want_shape} と違う")
        if spec["dtype"] in ("float32", "float64"):
            if not all(isinstance(x, float) and math.isfinite(x) for x in ent["data"]):
                errs.append(f"{key} に有限でない数（NaN・inf）がある")
    if errs:
        raise ValueError("npz が契約と合わない：" + "／".join(errs))
    fps = contract["recording"]["fps"]
    got_fps = npz["fps"]["data"][0]
    if got_fps != fps:
        errs.append(f"fps {got_fps} が契約の {fps} と違う")
    dur = npz["duration_s"]["data"][0]
    if not dur > 0:
        errs.append(f"duration_s {dur} が正でない")
    elif n_seen is None or n_seen < 1 or n_seen != math.floor(dur * fps + 1e-9):
        errs.append(f"行数 N={n_seen} が floor(duration_s × fps + 1e-9) と違う（または 1 未満）")
    tol = contract["tolerance"]["time_abs_s"]
    song = choreo["song"]["duration_s"]
    if abs(dur - song) > tol:
        errs.append(f"duration_s {dur} が今の振付の曲長 {song} と違う")
    n_song = math.floor(song * fps + 1e-9)
    if n_seen != n_song:
        errs.append(f"行数 N={n_seen} が今の振付の曲長から決まる {n_song} と違う")
    bad_ts = [k for k, t in enumerate(npz["timestamp"]["data"]) if abs(t - k / fps) > tol]
    if bad_ts:
        k = bad_ts[0]
        errs.append(f"timestamp が k / fps と合わない（{len(bad_ts)} 行・最初は k={k} で "
                    f"{npz['timestamp']['data'][k]}、{k / fps} のはず）")
    sha = npz["choreography_sha256"]["data"][0]
    if sha != choreo_sha256:
        errs.append(f"振付の sha256 {sha[:12]}… が今の choreography.json の {choreo_sha256[:12]}… と違う")
    names = contract["joint_names"]
    limits = contract["joint_limits_rad"]
    for i, row in enumerate(rows(npz["action"])):
        for name, x in zip(names, row):
            lo, hi = limits[name]
            if not lo <= x <= hi:
                errs.append(f"action[{i}] の {name}={x:.4f} が可動範囲 [{lo}, {hi}] の外")
                break
        if len(errs) > 5:
            break
    if errs:
        raise ValueError("npz が契約と合わない：" + "／".join(errs))
    ver = npz["contract_version"]["data"][0]
    if ver != contract["contract_version"]:
        warns.append(f"npz の契約の版 {ver} が今の契約の {contract['contract_version']} と違う"
                     "（形は同じなので再生は続ける）")
    return warns


def from_npz(contract: dict, choreo: dict, npz: dict, filename: str, *,
             choreo_sha256: str) -> Plan:
    warns = check_npz(contract, npz, choreo_sha256, choreo)
    fps = contract["recording"]["fps"]
    table = rows(npz["action"])

    def targets(k: int) -> list:
        return [float(x) for x in table[k]]

    return Plan(fps, len(table), _kind_from_name(filename), contract["joint_names"], targets,
                bar_heads(choreo, fps), warns)


def full_trajectory(plan: Plan) -> list:
    """全体を 1 本の軌道にした点の並び [(time_from_start の ns, 5 関節の位置), ...]。

    k = 0..n-1 の点（時刻 k * (NS // fps)・位置 plan.target(k)）のあとに、戻りの 1 点
    （時刻 (n-1) * (NS // fps) + 2 * NS・位置 plan.initial()）を付ける。
    """
    step = contract_math.NS // plan.fps
    pts = [(k * step, [float(x) for x in plan.target(k)]) for k in range(plan.n)]
    pts.append(((plan.n - 1) * step + 2 * contract_math.NS, [float(x) for x in plan.initial()]))
    return pts


def controller_points(contract: dict, points) -> dict:
    """full_trajectory の点を、腕と手の制御器ごとの (関節名の list, [(t_ns, 位置), ...]) に分ける。

    関節名と順は contract["controller"] の宣言どおり。値は contract["joint_names"] の中の位置で引く。
    """
    names = contract["joint_names"]
    ctl = contract["controller"]
    out = {}
    for key, field in (("arm", "arm_controller_joints"), ("gripper", "gripper_controller_joints")):
        jn = list(ctl[field])
        idx = [names.index(j) for j in jn]
        out[key] = (jn, [(t, [pos[i] for i in idx]) for t, pos in points])
    return out
