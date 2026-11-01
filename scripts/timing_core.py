"""踊りの時間のずれと動きのずれを、記録（ros2 bag を書き出した JSONL）から計算する純関数。

標準ライブラリのみ・ROS なし・Python 3.10 以上（Mac でもコンテナでも動く）。
使い方は scripts/analyze_timing.py。

時刻の名前
- wall：記録側（rosbag2）がメッセージを受け取った瞬間の PC の時計（ns）。配送の遅れが乗る
- sim ：シム時刻（Gazebo の時計・ns）。/clock の値と、各メッセージの header.stamp
- t   ：t0（最初の joint_target の stamp）からのシム時刻（秒）。格子 k の時刻は k / fps

ずれは 3 段に分けて出す
- learn：振付（お手本）と計画（npz の行）の差。学習の誤差。記録に依らない
- exec ：計画と実際の角度（/joint_states）の差。同じシム時刻で比べる。制御と物理の遅れ・誤差
- total：振付と実際の角度の差
シム時刻の上の遅れ（lag）は制御と物理の遅れで、シム時刻と実時間のずれ（実時間比）とは別物。
曲は実時間で鳴るので、「曲に合っていたか」は実時間比のほうで見る。
"""
from __future__ import annotations

import math
from bisect import bisect_right

NS = 1_000_000_000
ARM = 4  # 関節の並びの先頭 4 つが腕、5 つ目が手（contract.json の joint_names の順）

# 検査のしきい値（根拠つき）
# /joint_states の stamp と、同じ受信時刻の /clock の差。制御器は 100 Hz（contract.json controller.update_rate_hz）。
# bag の受信時刻にはまとめ届きの揺れが乗るので、中央値で 5 周期（50 ms）まで認める。
STAMP_SOURCE_MEDIAN_TOL_S = 0.05
# 壁時計のエポック秒（2001 年以降＝1e9 秒超）に見えるのにシム時刻が小さい場合は、時刻源が違う
EPOCH_LIKE_S = 1.0e9
# joint_target と計画の照合（同じ float64 を送っているので、ほぼ 0 で一致するはず）
TARGET_PLAN_TOL_RAD = 1e-6
# 遅れの探索範囲（0 以上に絞る。負の遅れは stamp の異常として扱う）
LAG_MAX_S = 0.30
LAG_STEP_S = 0.005
# 踊りの区間（t0〜最後の格子点）を記録が覆っているかの検査
# /clock はシム時刻 1 ms ごと・/joint_states は 10 ms ごとに出る（contract.json）。まとめ届きの揺れを見込んでも、
# 区間の中でこれより長い切れ目があれば、その区間の時計・動きは記録から決められない。
CLOCK_GAP_MAX_S = 0.1
JS_GAP_MAX_S = 0.2
# 振付で「動く関節」とみなす振れ幅（これ未満の関節は関節ごとの遅れを出さない）
MOVING_MIN_RAD = 0.01
# 計画の段差とみなす 1 コマあたりの変化（振付は 1 コマで最大 0.025 rad 程度しか動かないので、その 2 倍）
JUMP_RAD = 0.05


# ---------------------------------------------------------------- 読み分け
def split_records(records: list[dict], joint_names: list[str]) -> dict:
    """JSONL の行を、トピックごとの (wall_ns, stamp_ns, 位置) の並びに分ける。位置は joint_names の順。"""
    clock, js, jt, ctl = [], [], [], []
    js_zero = 0
    for r in records:
        topic = r.get("topic", "")
        if topic == "/clock":
            clock.append((int(r["wall_ns"]), int(r["stamp_ns"])))
        elif topic == "/joint_states":
            pos = _reorder(r.get("names", []), r.get("positions", []), joint_names)
            if pos is None:
                continue  # 関節がそろっていない行（別の発行元など）は使わない
            if int(r["stamp_ns"]) <= 0:
                js_zero += 1
            js.append((int(r["wall_ns"]), int(r["stamp_ns"]), pos))
        elif topic == "/crane_dance/joint_target":
            pos = _reorder(r.get("names", []), r.get("positions", []), joint_names)
            if pos is not None:
                jt.append((int(r["wall_ns"]), int(r["stamp_ns"]), pos))
        elif "reference" in r:
            ctl.append(r)
    clock.sort()
    js.sort()
    jt.sort()  # 受信順。stamp の後戻りは検査で見る
    return {"clock": clock, "js": js, "jt": jt, "ctl": ctl, "js_zero_stamp": js_zero}


def _reorder(names: list, positions: list, joint_names: list[str]):
    m = dict(zip(names, positions))
    if not all(j in m for j in joint_names):
        return None
    return [float(m[j]) for j in joint_names]


# ---------------------------------------------------------------- 時計の対応
class ClockMap:
    """/clock の (wall, sim) から、実時間 ⇄ シム時刻を直線補間で引く。"""

    def __init__(self, clock: list[tuple[int, int]]) -> None:
        # 同じ sim が続く（止まっている）区間は最初の wall だけ残す。sim の後戻りは捨てる
        pts = []
        for w, s in clock:
            if pts and s <= pts[-1][1]:
                continue
            pts.append((w, s))
        self.wall = [p[0] for p in pts]
        self.sim = [p[1] for p in pts]

    def __bool__(self) -> bool:
        return len(self.wall) >= 2

    def sim_at_wall(self, w: int) -> float:
        return _interp(self.wall, self.sim, w)

    def wall_at_sim(self, s: float) -> float:
        return _interp(self.sim, self.wall, s)


def _interp(xs: list, ys: list, x: float) -> float:
    """直線補間（範囲外は端の 2 点で外挿）。"""
    if len(xs) < 2:
        raise ValueError("点が 2 つ未満")
    i = bisect_right(xs, x) - 1
    i = min(max(i, 0), len(xs) - 2)
    x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


# ---------------------------------------------------------------- 目標
def grid_interp(targets: list[list[float]], fps: int, t: float) -> list[float]:
    """格子の目標（k / fps に置いた点）を直線補間する。範囲外は端の値。"""
    n = len(targets)
    u = t * fps
    if u <= 0:
        return list(targets[0])
    if u >= n - 1:
        return list(targets[-1])
    k = int(math.floor(u))
    w = u - k
    a, b = targets[k], targets[k + 1]
    return [x + w * (y - x) for x, y in zip(a, b)]


# ---------------------------------------------------------------- 検査
def check_targets(jt: list, plan: list[list[float]], fps: int) -> dict:
    """joint_target の本数・連続・後戻り・計画との一致を見る。t0 は最初の stamp。"""
    step = NS // fps
    out = {"count": len(jt), "expected": len(plan), "t0_ns": None, "gaps": [], "dups": 0,
           "backsteps": 0, "off_grid": 0, "plan_max_diff": None, "ok": False}
    if not jt:
        return out
    stamps = [s for _, s, _ in jt]
    t0 = stamps[0]
    out["t0_ns"] = t0
    seen = set()
    prev = None
    maxd = 0.0
    for _, s, pos in jt:
        if prev is not None and s < prev:
            out["backsteps"] += 1
        prev = s
        if (s - t0) % step != 0:
            out["off_grid"] += 1
            continue
        k = (s - t0) // step
        if k in seen:
            out["dups"] += 1
            continue
        seen.add(k)
        if 0 <= k < len(plan):
            maxd = max(maxd, max(abs(a - b) for a, b in zip(pos, plan[k])))
    ks = sorted(seen)
    out["gaps"] = [k for k in range(len(plan)) if k not in seen][:50]
    out["n_missing"] = sum(1 for k in range(len(plan)) if k not in seen)
    out["n_extra"] = sum(1 for k in ks if not 0 <= k < len(plan))
    out["plan_max_diff"] = maxd
    out["ok"] = (out["count"] == len(plan) and out["n_missing"] == 0 and out["dups"] == 0
                 and out["backsteps"] == 0 and out["off_grid"] == 0 and out["n_extra"] == 0
                 and maxd <= TARGET_PLAN_TOL_RAD)
    return out


def _in_window(js: list, window):
    if window is None:
        return js
    lo, hi = window
    return [x for x in js if lo <= x[1] <= hi]


def check_stamp_source(js: list, cmap: ClockMap, window=None) -> dict:
    """/joint_states の stamp がシム時刻かを、受信時刻が同じ /clock の値と比べて確かめる。
    window＝(t0_ns, 最後の格子点の ns) を渡すと、踊りの区間の中のサンプルだけで判定する。"""
    out = {"n": len(js), "median_abs_s": None, "max_abs_s": None, "epoch_like": False, "ok": False}
    if js and cmap and js[len(js) // 2][1] / NS > EPOCH_LIKE_S and cmap.sim[-1] / NS < EPOCH_LIKE_S:
        out["epoch_like"] = True
        return out
    js = _in_window(js, window)
    out["n"] = len(js)
    if not js or not cmap:
        return out
    lo, hi = cmap.wall[0], cmap.wall[-1]
    diffs = sorted(abs(s - cmap.sim_at_wall(w)) / NS for w, s, _ in js if lo <= w <= hi and s > 0)
    if not diffs:
        return out
    out["median_abs_s"] = diffs[len(diffs) // 2]
    out["max_abs_s"] = diffs[-1]
    out["ok"] = out["median_abs_s"] <= STAMP_SOURCE_MEDIAN_TOL_S
    return out


def _max_gap(points: list, lo: int, hi: int):
    """区間 [lo, hi] の中で、点の列（昇順の ns）が空けている一番長い切れ目（ns）。両端からの距離も含める。"""
    inside = [x for x in points if lo <= x <= hi]
    if not inside:
        return hi - lo
    gaps = [inside[0] - lo, hi - inside[-1]] + [b - a for a, b in zip(inside, inside[1:])]
    return max(gaps)


def check_coverage(cmap: ClockMap, js: list, t0_ns: int, n: int, fps: int) -> dict:
    """/clock と /joint_states が、踊りの区間（t0〜最後の格子点）を切れ目なく覆っているか。
    覆っていなければ、時計の比は外挿になり、動きのずれは足りない標本からの値になるので、解析は無効。"""
    t_end = t0_ns + (n - 1) * (NS // fps)
    sims = cmap.sim if cmap else []
    clock_covers = bool(sims) and sims[0] <= t0_ns and sims[-1] >= t_end
    clock_gap = _max_gap(sims, t0_ns, t_end) / NS
    js_stamps = sorted(s for _, s, _ in js if s > 0)
    js_in = sum(1 for s in js_stamps if t0_ns <= s <= t_end)
    js_gap = _max_gap(js_stamps, t0_ns, t_end) / NS
    ok = clock_covers and clock_gap <= CLOCK_GAP_MAX_S and js_in >= 2 and js_gap <= JS_GAP_MAX_S
    return {"clock_covers": clock_covers, "clock_max_gap_s": clock_gap, "js_in_window": js_in,
            "js_max_gap_s": js_gap, "window": (t0_ns, t_end), "ok": ok}


# ---------------------------------------------------------------- 時計の計算
def bar_ranges(downbeats_s: list[float], fps: int, n: int) -> list[tuple[int, int, int]]:
    """(小節番号, 始まりの k, 終わりの k)。番号は downbeat_times_s の添字（0 は前奏・plan.bar_heads と同じ）。
    最初の小節は k=0 から始める（曲頭の数コマを前奏に含める）。最後は n-1 まで。"""
    heads = sorted({(round(d * fps), i) for i, d in enumerate(downbeats_s)})
    heads = [(k, i) for k, i in heads if 0 <= k < n]
    out = []
    if heads and heads[0][0] > 0:
        heads[0] = (0, heads[0][1])
    for j, (k, i) in enumerate(heads):
        end = heads[j + 1][0] if j + 1 < len(heads) else n - 1
        if end > k:
            out.append((i, k, end))
    return out


def clock_report(cmap: ClockMap, jt: list, t0_ns: int, n: int, fps: int, bars: list,
                 window_s: float = 2.0) -> dict:
    """シム時刻と実時間の対応。全体・小節ごと・時間窓ごと。"""
    step = NS // fps
    sim_end = t0_ns + (n - 1) * step
    out = {"dance_sim_s": (n - 1) / fps}
    w0 = cmap.wall_at_sim(t0_ns)
    w1 = cmap.wall_at_sim(sim_end)
    out["dance_wall_s"] = (w1 - w0) / NS
    out["rtf_clock"] = out["dance_sim_s"] / out["dance_wall_s"] if out["dance_wall_s"] > 0 else None
    # joint_target の受信時刻での比（再生ノードの遅れが乗る。/clock と並べる用）
    first = next(((w, s) for w, s, _ in jt if s == t0_ns), None)
    last = next(((w, s) for w, s, _ in reversed(jt) if s == sim_end), None)
    if first and last and last[0] > first[0]:
        out["rtf_target"] = (last[1] - first[1]) / (last[0] - first[0])
    else:
        out["rtf_target"] = None
    recv = {}
    for w, s, _ in jt:
        if (s - t0_ns) % step == 0:
            recv.setdefault((s - t0_ns) // step, w)
    rows = []
    for bar, ka, kb in bars:
        sim_s = (kb - ka) / fps
        wall_c = (cmap.wall_at_sim(t0_ns + kb * step) - cmap.wall_at_sim(t0_ns + ka * step)) / NS
        wall_t = (recv[kb] - recv[ka]) / NS if ka in recv and kb in recv else None
        rows.append({
            "bar": bar, "k_start": ka, "k_end": kb, "sim_s": sim_s,
            "wall_s_clock": wall_c, "rtf_clock": sim_s / wall_c if wall_c > 0 else None,
            "wall_s_target": wall_t,
            "rtf_target": sim_s / wall_t if wall_t and wall_t > 0 else None,
            "wall_from_t0_s": (cmap.wall_at_sim(t0_ns + ka * step) - w0) / NS,
        })
    out["bars"] = rows
    # 時間窓（/clock だけで出す）
    wins = []
    win = int(window_s * NS)
    a = cmap.wall[0]
    while a + win <= cmap.wall[-1]:
        b = a + win
        ds = cmap.sim_at_wall(b) - cmap.sim_at_wall(a)
        wins.append({"wall_from_dance_start_s": (a - w0) / NS, "sim_advance_s": ds / NS,
                     "rtf": ds / win})
        a = b
    out["windows"] = wins
    return out


# ---------------------------------------------------------------- 動きの計算
def tracking_report(js: list, plan: list[list[float]], choreo: list[list[float]] | None,
                    t0_ns: int, fps: int, bars: list, cmap: ClockMap | None = None) -> dict:
    """同じシム時刻で 計画⇔実際（exec）・振付⇔実際（total）・振付⇔計画（learn）を比べる。"""
    n = len(plan)
    t_end = (n - 1) / fps
    samples = []
    for w, s, pos in js:
        if s <= 0:
            continue
        t = (s - t0_ns) / NS
        if not 0.0 <= t <= t_end:
            continue
        p = grid_interp(plan, fps, t)
        c = grid_interp(choreo, fps, t) if choreo else None
        samples.append({"wall_ns": w, "t": t, "actual": pos, "plan": p, "choreo": c})
    nj = len(plan[0])
    out = {"n_samples": len(samples)}

    def stats(rows, key_a, key_b):
        res = []
        for j in range(nj):
            pairs = [(r[key_a][j] - r[key_b][j], r["t"]) for r in rows if r[key_b] is not None]
            if not pairs:
                res.append({"max": None, "rms": None, "mean": None, "t_at_max": None})
                continue
            e = [x for x, _ in pairs]
            worst = max(pairs, key=lambda q: abs(q[0]))
            res.append({"max": abs(worst[0]), "t_at_max": worst[1],
                        "rms": math.sqrt(sum(x * x for x in e) / len(e)),
                        "mean": sum(e) / len(e)})
        return res

    out["exec"] = stats(samples, "actual", "plan")
    out["total"] = stats(samples, "actual", "choreo") if choreo else None
    if choreo:
        learn = []
        for j in range(nj):
            e = [plan[k][j] - choreo[k][j] for k in range(n)]
            learn.append({"max": max(abs(x) for x in e),
                          "rms": math.sqrt(sum(x * x for x in e) / n), "mean": sum(e) / n})
        out["learn"] = learn
    else:
        out["learn"] = None
    rows = []
    for bar, ka, kb in bars:
        ta, tb = ka / fps, kb / fps
        sub = [r for r in samples if ta <= r["t"] < tb]
        ex = stats(sub, "actual", "plan") if sub else [{"max": None, "rms": None, "mean": None}] * nj
        to = stats(sub, "actual", "choreo") if (sub and choreo) else [{"max": None, "rms": None, "mean": None}] * nj
        le = None
        if choreo:
            le = [max(abs(plan[k][j] - choreo[k][j]) for k in range(ka, kb + 1)) for j in range(nj)]
        rows.append({"bar": bar, "k_start": ka, "k_end": kb, "n": len(sub), "exec": ex, "total": to,
                     "learn_max": le})
    out["bars"] = rows
    ref = choreo or plan
    out["plan_jumps"] = plan_jumps(plan, fps)
    out["choreo_max_step"] = (max(abs(choreo[k + 1][j] - choreo[k][j]) for k in range(n - 1) for j in range(nj))
                              if choreo else None)
    out["moving"] = [max(r[j] for r in ref) - min(r[j] for r in ref) > MOVING_MIN_RAD for j in range(nj)]
    out["lag"] = lag_scan(samples, plan, fps)
    out["samples"] = samples
    return out


def plan_jumps(plan: list[list[float]], fps: int, thresh: float = JUMP_RAD, top: int = 10) -> dict:
    """計画の隣り合う格子の差が thresh を超える所（段差）。開ループの推論の切れ目で出やすい。"""
    nj = len(plan[0])
    found = []
    for k in range(len(plan) - 1):
        for j in range(nj):
            dv = plan[k + 1][j] - plan[k][j]
            if abs(dv) > thresh:
                found.append({"k": k + 1, "t": (k + 1) / fps, "joint": j, "delta": dv})
    found.sort(key=lambda x: -abs(x["delta"]))
    return {"count": len(found), "thresh": thresh, "top": found[:top]}


def lag_scan(samples: list[dict], plan: list[list[float]], fps: int) -> dict:
    """実際(t) と 計画(t − d) の差の RMS が最小になる d（0〜LAG_MAX_S）。腕全体と関節ごと。
    どの候補 d でも同じ標本（t ≧ LAG_MAX_S のもの）で比べる。比べられる標本が無ければ推定不能（best_d_s=None）。"""
    nj = len(plan[0])
    steps = int(round(LAG_MAX_S / LAG_STEP_S))
    common = [r for r in samples if r["t"] >= LAG_MAX_S - 1e-12]
    curve = []
    for i in range(steps + 1):
        d = i * LAG_STEP_S
        per = [0.0] * nj
        cnt = 0
        for r in common:
            p = grid_interp(plan, fps, r["t"] - d)
            for j in range(nj):
                e = r["actual"][j] - p[j]
                per[j] += e * e
            cnt += 1
        if cnt == 0:
            continue
        per = [math.sqrt(x / cnt) for x in per]
        arm = math.sqrt(sum(x * x for x in per[:ARM]) / ARM)
        curve.append({"d_s": d, "rms_arm": arm, "rms": per, "n": cnt})
    if not curve:
        return {"curve": [], "best_d_s": None, "best_rms_arm": None, "per_joint_best_d_s": None,
                "at_edge": None}
    best = min(curve, key=lambda c: c["rms_arm"])
    per_best = [min(curve, key=lambda c: c["rms"][j])["d_s"] for j in range(nj)]
    return {"curve": curve, "best_d_s": best["d_s"], "best_rms_arm": best["rms_arm"],
            "rms_arm_at_0": curve[0]["rms_arm"], "per_joint_best_d_s": per_best,
            "at_edge": best["d_s"] >= LAG_MAX_S - 1e-12}


def controller_check(ctl: list[dict], plan: list[list[float]], joint_names: list[str], t0_ns: int,
                     fps: int) -> dict | None:
    """制御器の state（任意）の reference と計画の差。制御器が計画どおりに補間しているかを見る。"""
    if not ctl:
        return None
    n = len(plan)
    worst = {}
    cnt = 0
    for r in ctl:
        t = (int(r["stamp_ns"]) - t0_ns) / NS
        if not 0.0 <= t <= (n - 1) / fps:
            continue
        p = grid_interp(plan, fps, t)
        for name, val in zip(r.get("names", []), r.get("reference", [])):
            if name in joint_names:
                d = abs(float(val) - p[joint_names.index(name)])
                worst[name] = max(worst.get(name, 0.0), d)
        cnt += 1
    return {"n": cnt, "max_ref_minus_plan": worst}


# ---------------------------------------------------------------- 再生ノードのログ
def parse_player_log(lines: list[str]) -> dict:
    """README の計測の手順で付けた「壁時計 戻らない時計 本文」の行から、t0 と小節の行の壁時計を取る。"""
    import re
    out = {"t0_wall_s": None, "bars": [], "end_wall_s": None}
    for line in lines:
        parts = line.split(" ", 2)
        if len(parts) < 3:
            continue
        try:
            wall = float(parts[0])
        except ValueError:
            continue
        text = parts[2]
        if text.startswith("t0 = "):
            out["t0_wall_s"] = wall
        m = re.match(r"小節 (\d+)（k=(\d+)）", text)
        if m:
            out["bars"].append({"bar": int(m.group(1)), "k": int(m.group(2)), "wall_s": wall})
        if text.startswith("終了"):
            out["end_wall_s"] = wall
    return out
