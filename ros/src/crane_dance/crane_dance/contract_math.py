"""契約表の計算（ROS に依存しない純関数・標準ライブラリのみ・Python 3.10 以上）。

npz を書く側・再生ノード・検査器（scripts/check_contract.py）が、
この 1 本を使う。式の正本は contract/contract.json の beat_phase・recording・golden_*。
"""
from __future__ import annotations

import math
from bisect import bisect_right


def phase(beats: list[float], tau: float) -> float:
    """拍の時刻表 beats と曲の時刻 tau から拍位相 φ (rad) を返す。"""
    i = bisect_right(beats, tau) - 1
    if i < 0:
        edge, interval = beats[0], beats[1] - beats[0]
        x = (tau - edge) / interval
        return 2 * math.pi * (x - math.floor(x))
    if i >= len(beats) - 1:
        edge, interval = beats[-1], beats[-1] - beats[-2]
        x = (tau - edge) / interval
        return 2 * math.pi * (x - math.floor(x))
    return 2 * math.pi * (tau - beats[i]) / (beats[i + 1] - beats[i])


def progress(tau: float, duration_s: float) -> float:
    return min(1.0, max(0.0, tau / duration_s))


def env_state(beats: list[float], tau: float, duration_s: float) -> tuple[float, float, float]:
    """環境状態 (sin φ, cos φ, progress)。"""
    phi = phase(beats, tau)
    return (math.sin(phi), math.cos(phi), progress(tau, duration_s))


NS = 1_000_000_000
# 時刻は整数のナノ秒で扱う（header.stamp は sec と nanosec の整数。
# 秒の float を経由すると 1 ns ずれ、floor で k が 1 つ下がることがある）


def _need_int(**kw) -> None:
    for name, v in kw.items():
        if not isinstance(v, int) or isinstance(v, bool):
            raise TypeError(f"{name} は int でなければならない（{type(v).__name__} を受け取った）")


def grid_index(now_ns: int, t0_ns: int, fps: int) -> int:
    """格子番号 k = (now_ns - t0_ns) * fps // NS。t0 より前なら負のまま返す。"""
    _need_int(now_ns=now_ns, t0_ns=t0_ns, fps=fps)
    return (now_ns - t0_ns) * fps // NS


def grid_time_ns(k: int, t0_ns: int, fps: int) -> int:
    """格子点 k の時刻 t_k (ns) = t0_ns + k * (NS // fps)。fps が NS を割り切らなければ ValueError。"""
    _need_int(k=k, t0_ns=t0_ns, fps=fps)
    if fps <= 0 or NS % fps != 0:
        raise ValueError(f"fps={fps} は 1e9 を割り切る正の整数でなければならない")
    return t0_ns + k * (NS // fps)


def grid_steps(nows_ns: list[int], t0_ns: int, fps: int) -> dict:
    """順に来る now の列から、出すべき格子番号を決める。

    返り値 {"emit": [k...], "skipped": 整数, "invalid_at": None か int}。
    k < 0（t0 より前）は出さない。直前に出した k より大きいときだけ出す。
    飛んだ番号は skipped に数える（最初に出した k は 0 から数える）。
    now が直前の now より小さければ、その位置を invalid_at にして以後は見ない。
    """
    emit: list[int] = []
    skipped = 0
    invalid_at = None
    last_k = None
    prev = None
    for pos, now in enumerate(nows_ns):
        k = grid_index(now, t0_ns, fps)
        if prev is not None and now < prev:
            invalid_at = pos
            break
        prev = now
        if k < 0:
            continue
        if last_k is None:
            skipped += k
        elif k > last_k:
            skipped += k - last_k - 1
        else:
            continue
        emit.append(k)
        last_k = k
    return {"emit": emit, "skipped": skipped, "invalid_at": invalid_at}


def target_at(t: float, keyframes: list[dict]) -> list[float]:
    """振付の補間。最初より前は最初の値・最後より後は最後の値・間は関節ごとに直線補間。"""
    times = [kf["time_s"] for kf in keyframes]
    if t <= times[0]:
        return list(keyframes[0]["joints_rad"])
    if t >= times[-1]:
        return list(keyframes[-1]["joints_rad"])
    i = bisect_right(times, t) - 1
    a, b = keyframes[i], keyframes[i + 1]
    w = (t - a["time_s"]) / (b["time_s"] - a["time_s"])
    return [x + w * (y - x) for x, y in zip(a["joints_rad"], b["joints_rad"])]


def episode_complete(stamps_ns: list[int], n_expected: int, fps: int) -> bool:
    """1 エピソードとして欠けも混入も無いか。

    本数がちょうど n_expected で、stamp が stamps_ns[0] から 1/fps 刻みで隙間なく続く（k = 0..n-1 の連続）なら True。
    姿勢の照合（t0_consistent）だけでは、止まっている区間の取りこぼしや、同じ姿勢で始まる 2 本目の混入を
    見分けられないので、1 エピソードと認めるにはこの条件と t0_consistent の両方が要る。
    """
    _need_int(n_expected=n_expected, fps=fps)
    if n_expected <= 0 or len(stamps_ns) != n_expected:
        return False
    t0 = stamps_ns[0]
    if not isinstance(t0, int) or isinstance(t0, bool):
        return False
    return all(isinstance(s, int) and not isinstance(s, bool) and s == grid_time_ns(k, t0, fps)
               for k, s in enumerate(stamps_ns))


def episode_accepted(stamps_ns: list[int], positions: list[list[float]], keyframes, tol: float,
                     n_expected: int, fps: int) -> bool:
    """1 エピソードとして使ってよいか（完全性と姿勢の照合の両方）。"""
    return (episode_complete(stamps_ns, n_expected, fps)
            and t0_consistent(stamps_ns, positions, keyframes, tol))


def t0_consistent(stamps_ns: list[int], positions: list[list[float]], keyframes,
                  tol: float) -> bool:
    """t0 = stamps_ns[0] と仮定し、全 i で関節目標の全要素が有限な数で、振付の補間値と tol 以内なら True。

    NaN は abs(NaN - w) > tol が偽になって素通りするので、比較の前に有限性を確かめる（±inf も不可）。
    """
    if not stamps_ns or len(stamps_ns) != len(positions):
        return False
    t0 = stamps_ns[0]
    for stamp, pos in zip(stamps_ns, positions):
        want = target_at((stamp - t0) / NS, keyframes)
        if len(pos) != len(want):
            return False
        for p, w in zip(pos, want):
            if not (isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p)
                    and abs(p - w) <= tol):
                return False
    return True
