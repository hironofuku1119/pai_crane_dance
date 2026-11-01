#!/usr/bin/env python3
"""踊りの時間のずれ（シム時刻と実時間）と動きのずれ（目標と実際の角度）を、ros2 bag の記録から出す。

2 段で使う。
1. extract（コンテナの中・ROS が要る）：bag を 1 行 1 メッセージの JSONL に書き出す
     python3 /workspace/scripts/analyze_timing.py extract /workspace/ros/bags/timing/run1_norec \\
         /workspace/ros/bags/timing/run1_norec.jsonl
   ROS の部品（rosbag2_py・rclpy・rosidl_runtime_py）は講座イメージに入っているものを使う。
2. analyze（どこでも・標準ライブラリだけ）：JSONL から表とグラフと要約を作る
     python3 scripts/analyze_timing.py analyze ros/bags/timing/run1_norec.jsonl \\
         --npz ros/replay/b5_ckpt20000_open.npz --player-log ros/bags/timing/run1_norec_player.txt \\
         --out ros/bags/timing/run1_norec_analysis
   --npz を省くと、お手本（振付ファイル）を再生した記録として扱う。

出力（--out のフォルダ）：summary.md・clock_bars.csv・rtf_windows.csv・tracking_bars.csv・
lag_curve.csv・samples.csv・tracking.svg・error.svg・clock.svg
最終行に `TIMING: PASS` か `TIMING: FAIL (n)`（n は落ちた検査の数）を出す。落ちた検査があっても数値は出すが、
要約の先頭に「無効」と書く。計算の中身は scripts/timing_core.py。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "ros" / "src" / "crane_dance"))

import timing_core as tc  # noqa: E402
import timing_svg  # noqa: E402

JOINT_SHORT = ["j1", "j2", "j3", "j4", "hand"]


# ================================================================ extract（ROS が要る）
def _storage_id(bag_dir: Path) -> str:
    meta = bag_dir / "metadata.yaml"
    if not meta.exists():
        raise SystemExit(f"{meta} が無い → 記録を Ctrl+C 1 回で止めたか確かめる。無ければ `ros2 bag reindex {bag_dir}` を打つ")
    m = re.search(r"storage_identifier:\s*(\S+)", meta.read_text(encoding="utf-8"))
    return m.group(1) if m else "sqlite3"


def _stamp_ns(t) -> int:
    return int(t.sec) * tc.NS + int(t.nanosec)


def _first_with_positions(msg, fields):
    for name in fields:
        v = getattr(msg, name, None)
        if v is not None and len(getattr(v, "positions", [])) > 0:
            return v
    return None


def _to_record(topic: str, typ: str, msg, wall_ns: int):
    if typ == "rosgraph_msgs/msg/Clock":
        return {"topic": topic, "wall_ns": wall_ns, "stamp_ns": _stamp_ns(msg.clock)}
    stamp = _stamp_ns(msg.header.stamp) if hasattr(msg, "header") else 0
    if typ == "sensor_msgs/msg/JointState":
        return {"topic": topic, "wall_ns": wall_ns, "stamp_ns": stamp,
                "names": list(msg.name), "positions": [float(x) for x in msg.position]}
    if hasattr(msg, "joint_names"):
        # 制御器の state（control_msgs/msg/JointTrajectoryControllerState）。Humble は reference/feedback、
        # 古い版は desired/actual
        ref = _first_with_positions(msg, ("reference", "desired"))
        fb = _first_with_positions(msg, ("feedback", "actual"))
        rec = {"topic": topic, "wall_ns": wall_ns, "stamp_ns": stamp, "names": list(msg.joint_names)}
        if ref is not None:
            rec["reference"] = [float(x) for x in ref.positions]
        if fb is not None:
            rec["feedback"] = [float(x) for x in fb.positions]
        return rec
    return None


def cmd_extract(args) -> int:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    bag = Path(args.bag)
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=_storage_id(bag)),
                rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                            output_serialization_format="cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    classes = {}
    counts = {}
    with open(args.out, "w", encoding="utf-8") as f:
        while reader.has_next():
            topic, data, wall_ns = reader.read_next()
            typ = types[topic]
            if typ not in classes:
                classes[typ] = get_message(typ)
            rec = _to_record(topic, typ, deserialize_message(data, classes[typ]), int(wall_ns))
            if rec is not None:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[topic] = counts.get(topic, 0) + 1
    for t, n in sorted(counts.items()):
        print(f"{t}: {n} 本")
    print(f"書き出した → {args.out}")
    return 0


# ================================================================ analyze（標準ライブラリだけ）
def load_inputs(args):
    from crane_dance import contract_math, npz_io, plan as plan_mod

    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    choreo_path = Path(args.choreo)
    choreo = json.loads(choreo_path.read_text(encoding="utf-8"))
    sha = hashlib.sha256(choreo_path.read_bytes()).hexdigest()
    if args.npz:
        p = plan_mod.from_npz(contract, choreo, npz_io.read_npz(args.npz), Path(args.npz).name,
                              choreo_sha256=sha)
    else:
        p = plan_mod.from_choreography(contract, choreo)
    targets = [p.target(k) for k in range(p.n)]
    choreo_t = [[float(x) for x in contract_math.target_at(k / p.fps, choreo["keyframes"])]
                for k in range(p.n)]
    return contract, choreo, p, targets, choreo_t


def _f(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def cmd_analyze(args) -> int:
    contract, choreo, plan, targets, choreo_t = load_inputs(args)
    names = contract["joint_names"]
    fps = plan.fps
    records = [json.loads(line) for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    d = tc.split_records(records, names)
    cmap = tc.ClockMap(d["clock"])
    tgt = tc.check_targets(d["jt"], targets, fps)
    cov = (tc.check_coverage(cmap, d["js"], tgt["t0_ns"], plan.n, fps) if tgt["t0_ns"] is not None
           else {"ok": False, "clock_covers": False, "clock_max_gap_s": None, "js_in_window": 0, "js_max_gap_s": None,
                 "window": None})
    src = tc.check_stamp_source(d["js"], cmap, window=cov["window"])

    checks = [
        ("/clock が 2 本以上ある", bool(cmap), f"{len(d['clock'])} 本"),
        ("joint_target が 1 エピソードぶん（本数・連続・後戻りなし・計画と一致）", tgt["ok"],
         f"{tgt['count']} / {tgt['expected']} 本・欠け {tgt.get('n_missing', '—')}・重複 {tgt['dups']}・"
         f"後戻り {tgt['backsteps']}・格子外 {tgt['off_grid']}・計画との差の最大 {_f(tgt['plan_max_diff'], 9)} rad"),
        ("/joint_states の時刻がシム時刻（同じ受信時刻の /clock との差の中央値 ≦ "
         f"{tc.STAMP_SOURCE_MEDIAN_TOL_S} s）", src["ok"],
         "壁時計のエポック秒に見える" if src["epoch_like"] else
         f"区間の中の {src['n']} 本・差の中央値 {_f(src['median_abs_s'], 4)} s・最大 {_f(src['max_abs_s'], 4)} s"),
        ("/joint_states の stamp が 0 の行が無い", d["js_zero_stamp"] == 0, f"{d['js_zero_stamp']} 本"),
        (f"記録が踊りの区間（t0〜最後の格子点）を覆っている（/clock の切れ目 ≦ {tc.CLOCK_GAP_MAX_S} s・"
         f"/joint_states の切れ目 ≦ {tc.JS_GAP_MAX_S} s）", cov["ok"],
         f"/clock が区間を覆う {'はい' if cov['clock_covers'] else 'いいえ'}・切れ目の最大 {_f(cov['clock_max_gap_s'], 4)} s・"
         f"区間の /joint_states {cov['js_in_window']} 本・切れ目の最大 {_f(cov['js_max_gap_s'], 4)} s"),
    ]
    fails = sum(1 for _, ok, _ in checks if not ok)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clock = track = None
    bars = tc.bar_ranges(choreo["song"]["downbeat_times_s"], fps, plan.n)
    if cmap and tgt["t0_ns"] is not None:
        t0 = tgt["t0_ns"]
        clock = tc.clock_report(cmap, d["jt"], t0, plan.n, fps, bars)
        track = tc.tracking_report(d["js"], targets, choreo_t, t0, fps, bars)
        ctl = tc.controller_check(d["ctl"], targets, names, t0, fps)
    else:
        ctl = None
    player = None
    if args.player_log and Path(args.player_log).exists():
        player = tc.parse_player_log(Path(args.player_log).read_text(encoding="utf-8").splitlines())

    write_csvs(out, clock, track, names)
    if clock and track:
        write_svgs(out, clock, track, cmap, tgt["t0_ns"], fps, args.label)
    summary = render_summary(args, plan, checks, fails, clock, track, ctl, player, cmap, tgt)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"出力 → {out}")
    print("TIMING: PASS" if fails == 0 else f"TIMING: FAIL ({fails})")
    return 0 if fails == 0 else 1


def write_csvs(out: Path, clock, track, names) -> None:
    if clock:
        with open(out / "clock_bars.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["bar", "k_start", "k_end", "sim_s", "wall_s_clock", "rtf_clock", "wall_s_target",
                        "rtf_target", "wall_from_t0_s"])
            for r in clock["bars"]:
                w.writerow([r[k] for k in ("bar", "k_start", "k_end", "sim_s", "wall_s_clock", "rtf_clock",
                                           "wall_s_target", "rtf_target", "wall_from_t0_s")])
        with open(out / "rtf_windows.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["wall_from_dance_start_s", "sim_advance_s", "rtf"])
            for r in clock["windows"]:
                w.writerow([r["wall_from_dance_start_s"], r["sim_advance_s"], r["rtf"]])
    if track:
        with open(out / "tracking_bars.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["bar", "joint", "n", "exec_max", "exec_rms", "total_max", "total_rms", "learn_max"])
            for r in track["bars"]:
                for j, nm in enumerate(JOINT_SHORT):
                    w.writerow([r["bar"], nm, r["n"], r["exec"][j]["max"], r["exec"][j]["rms"],
                                r["total"][j]["max"], r["total"][j]["rms"],
                                r["learn_max"][j] if r["learn_max"] else None])
        with open(out / "lag_curve.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["d_s", "rms_arm"] + [f"rms_{n}" for n in JOINT_SHORT])
            for c in track["lag"]["curve"]:
                w.writerow([c["d_s"], c["rms_arm"]] + c["rms"])
        with open(out / "samples.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["wall_ns", "t_sim_s"] + [f"{k}_{n}" for n in JOINT_SHORT for k in ("actual", "plan", "choreo")])
            for r in track["samples"]:
                row = [r["wall_ns"], r["t"]]
                for j in range(len(JOINT_SHORT)):
                    row += [r["actual"][j], r["plan"][j], r["choreo"][j] if r["choreo"] else None]
                w.writerow(row)


def write_svgs(out: Path, clock, track, cmap, t0, fps, label) -> None:
    s = track["samples"]
    ts = [r["t"] for r in s]
    panels = []
    for j, nm in enumerate(JOINT_SHORT):
        panels.append({"title": nm, "ylabel": "rad", "xlabel": "シム時刻（t0 から・秒）" if j == 4 else "",
                       "min_span": 0.2,
                       "series": [
                           {"x": ts, "y": [r["choreo"][j] for r in s], "color": timing_svg.COLORS["choreo"],
                            "dash": "5,4", "label": "振付（お手本）"},
                           {"x": ts, "y": [r["plan"][j] for r in s], "color": timing_svg.COLORS["plan"],
                            "label": "計画（B5 の出力）"},
                           {"x": ts, "y": [r["actual"][j] for r in s], "color": timing_svg.COLORS["actual"],
                            "label": "実際の角度"}]})
    (out / "tracking.svg").write_text(timing_svg.panels_svg(panels, title=f"目標と実際の角度（{label}）"),
                                      encoding="utf-8")
    err = [{"title": "計画との差（実際 − 計画）", "ylabel": "rad", "xlabel": "シム時刻（t0 から・秒）",
            "series": [{"x": ts, "y": [r["actual"][j] - r["plan"][j] for r in s],
                        "color": timing_svg.COLORS[nm], "label": nm} for j, nm in enumerate(JOINT_SHORT)]}]
    (out / "error.svg").write_text(timing_svg.panels_svg(err, panel_h=260, title=f"動きのずれ（{label}）"),
                                   encoding="utf-8")
    w0 = cmap.wall_at_sim(t0)
    xs = [(w - w0) / tc.NS for w in cmap.wall]
    ys = [(v - t0) / tc.NS for v in cmap.sim]
    keep = [(x, y) for x, y in zip(xs, ys) if -2 <= x]
    xs, ys = [p[0] for p in keep], [p[1] for p in keep]
    lim = max(xs) if xs else 1
    clk = [{"title": f"シム時刻と実時間（踊り本体の実時間比 {clock['rtf_clock']:.3f}）", "ylabel": "シム時刻（秒）",
            "xlabel": "実時間（t0 から・秒）",
            "series": [{"x": [0, lim], "y": [0, lim], "color": timing_svg.COLORS["ideal"], "dash": "5,4",
                        "label": "実時間比 1（ずれなし）"},
                       {"x": xs, "y": ys, "color": timing_svg.COLORS["plan"], "label": "/clock の記録"}]}]
    (out / "clock.svg").write_text(timing_svg.panels_svg(clk, panel_h=320, title=f"時計のずれ（{label}）"),
                                   encoding="utf-8")


def render_summary(args, plan, checks, fails, clock, track, ctl, player, cmap, tgt) -> str:
    L = [f"# 時間と動きのずれの解析（{args.label}）", ""]
    L.append(f"- 記録：`{args.jsonl}`")
    L.append(f"- 計画：{plan.kind}" + (f"（`{args.npz}`）" if args.npz else "") + f"・{plan.n} 点・{plan.fps} fps")
    L.append(f"- 解析した日時：{time.strftime('%Y-%m-%d %H:%M:%S %z')}")
    L.append("")
    if fails:
        L.append(f"> **無効：検査が {fails} 個落ちた。下の数値はそのまま信じない。**")
        L.append("")
    L += ["## 1. 検査", "", "| 検査 | 結果 | 中身 |", "|---|---|---|"]
    for name, ok, detail in checks:
        L.append(f"| {name} | {'OK' if ok else '**NG**'} | {detail} |")
    L.append("")
    if clock:
        r = clock["rtf_clock"]
        L += ["## 2. 時計（シム時刻と実時間）", ""]
        L.append(f"- 踊り本体：シム時刻で **{clock['dance_sim_s']:.2f} 秒** → 実時間で **{clock['dance_wall_s']:.2f} 秒**"
                 f"（実時間比 **{_f(r)}**。シム時刻どおりの速さに戻す倍率は {_f(1 / r if r else None)} 倍）")
        L.append(f"- 同じ区間を joint_target の受信時刻で測った実時間比：{_f(clock['rtf_target'])}"
                 "（再生ノードの遅れが乗るので参考）")
        if clock["windows"]:
            rs = sorted(w["rtf"] for w in clock["windows"])
            L.append(f"- 2 秒ごとの実時間比（/clock・記録全体）：最小 {rs[0]:.3f}・中央 {rs[len(rs) // 2]:.3f}・最大 {rs[-1]:.3f}")
        L += ["", "| 小節 | シム時刻（秒） | 実時間（秒・/clock） | 実時間比 | 実時間（秒・受信） | t0 からの実時間（秒） |",
              "|---|---|---|---|---|---|"]
        for b in clock["bars"]:
            L.append(f"| {b['bar']} | {b['sim_s']:.3f} | {b['wall_s_clock']:.3f} | {_f(b['rtf_clock'])} | "
                     f"{_f(b['wall_s_target'])} | {b['wall_from_t0_s']:.3f} |")
        L.append("")
    if track:
        L += ["## 3. 動き（同じシム時刻で比べる・rad）", "",
              f"使った /joint_states：{track['n_samples']} 本（t0 から踊りの終わりまで）", "",
              "| 関節 | 学習の差 最大（振付−計画） | 実行の差 最大（実際−計画） | 実行の差 RMS | 全体の差 最大（実際−振付） | 全体の差 RMS |",
              "|---|---|---|---|---|---|"]
        for j, nm in enumerate(JOINT_SHORT):
            le = track["learn"][j] if track["learn"] else {"max": None}
            ex = track["exec"][j]
            to = track["total"][j] if track["total"] else {"max": None, "rms": None}
            L.append(f"| {nm} | {_f(le['max'])} | {_f(ex['max'])} | {_f(ex['rms'])} | {_f(to['max'])} | {_f(to['rms'])} |")
        lag = track["lag"]
        moving = track["moving"]
        L.append("")
        L.append("実行の差が一番大きかった時刻（シム時刻・t0 から）：" + "・".join(
            f"{nm} {track['exec'][j]['t_at_max']:.2f} 秒（k≈{round(track['exec'][j]['t_at_max'] * plan.fps)}）"
            for j, nm in enumerate(JOINT_SHORT) if moving[j] and track["exec"][j]["t_at_max"] is not None))
        pj = track["plan_jumps"]
        L.append(f"- 計画の段差（隣り合うコマの差が {pj['thresh']} rad 超）：{pj['count']} か所"
                 + (f"（振付は 1 コマで最大 {track['choreo_max_step']:.3f} rad）" if track["choreo_max_step"] is not None else ""))
        if pj["top"]:
            L.append("  - 大きい順：" + "・".join(
                f"{JOINT_SHORT[x['joint']]} k={x['k']}（{x['t']:.2f} 秒）{x['delta']:+.3f}" for x in pj["top"][:6]))
        L.append("")
        if lag["best_d_s"] is None:
            L.append(f"- 遅れ：推定不能（t ≧ {tc.LAG_MAX_S} s の標本が無い）")
        else:
            L.append(f"- 遅れ（シム時刻の上）：実際の角度は計画から **{lag['best_d_s']:.3f} 秒** 遅れて追うと一番よく合う"
                     f"（腕 4 関節の RMS が {lag['rms_arm_at_0']:.4f} → {lag['best_rms_arm']:.4f} rad）。"
                     "関節ごと：" + "・".join(
                         f"{n} {v:.3f} s" if moving[j] else f"{n} —（振付で動かない）"
                         for j, (n, v) in enumerate(zip(JOINT_SHORT, lag["per_joint_best_d_s"]))))
            if lag["at_edge"]:
                L.append(f"  - **注意：探索の端（{tc.LAG_MAX_S} s）で最小。遅れはもっと大きいか、遅れでは説明できないずれ**")
            L.append("  - これは制御と物理の遅れで、シム時刻と実時間のずれ（§2）とは別物")
        L += ["", "小節ごと（腕 4 関節の中で一番大きい実行の差・全体の差）：", "",
              "| 小節 | 本数 | 実行の差 最大 | 全体の差 最大 | 学習の差 最大 |", "|---|---|---|---|---|"]
        for b in track["bars"]:
            ex = [e["max"] for e in b["exec"][:tc.ARM] if e["max"] is not None]
            to = [e["max"] for e in b["total"][:tc.ARM] if e["max"] is not None]
            le = max(b["learn_max"][:tc.ARM]) if b["learn_max"] else None
            L.append(f"| {b['bar']} | {b['n']} | {_f(max(ex) if ex else None)} | {_f(max(to) if to else None)} | {_f(le)} |")
        L.append("")
    if ctl:
        L += ["## 4. 制御器の state（記録してあれば）", "",
              f"- {ctl['n']} 本。制御器の reference と計画の差の最大：" +
              "・".join(f"{k} {v:.4f}" for k, v in ctl["max_ref_minus_plan"].items()), ""]
    if player and player["t0_wall_s"] is not None and cmap and tgt["t0_ns"] is not None:
        L += ["## 5. 再生ノードのログとの照合", ""]
        w0 = cmap.wall_at_sim(tgt["t0_ns"]) / tc.NS
        step = tc.NS // plan.fps
        L += ["| 小節 | ログの行の実時間（t0 の行から・秒） | /clock から計算した実時間（t0 から・秒） |", "|---|---|---|"]
        for b in player["bars"]:
            wc = cmap.wall_at_sim(tgt["t0_ns"] + b["k"] * step) / tc.NS - w0
            L.append(f"| {b['bar']} | {b['wall_s'] - player['t0_wall_s']:.3f} | {wc:.3f} |")
        if player["end_wall_s"] is not None:
            L.append(f"| 終了の行 | {player['end_wall_s'] - player['t0_wall_s']:.3f} | — |")
        L += ["", "t0 の行は t0（シム時刻）の 0.5 秒前に出る。ログの時刻は print の時刻なので、少し遅れて出る。", ""]
    L += ["## 読み方の注意", "",
          "- 再生ノードのログの「追従誤差の最大」は、時刻を格子に切り捨てて比べている（`player.py` の on_joint_states）。"
          "この道具は格子の間を直線で補間して比べるので、数値は一致しない",
          "- 制御器は位置だけの軌道点を直線補間する前提で計画を補間している（制御器の state を記録すると §4 で確かめられる）",
          "- /clock は毎秒約 1000 本あり、その記録自体がシムを遅らせうる。動画を撮った回とは条件が違う",
          "- 手（hand）は物に触れたり力の上限に当たると目標に届かないことがある。腕と分けて読む", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="bag を JSONL に書き出す（コンテナの中）")
    e.add_argument("bag")
    e.add_argument("out")
    a = sub.add_parser("analyze", help="JSONL から表とグラフを作る（どこでも）")
    a.add_argument("jsonl")
    a.add_argument("--out", required=True)
    a.add_argument("--npz", default="")
    a.add_argument("--player-log", default="")
    a.add_argument("--contract", default=str(REPO / "contract" / "contract.json"))
    a.add_argument("--choreo", default=str(REPO / "contract" / "choreography.json"))
    a.add_argument("--label", default="")
    args = ap.parse_args(argv)
    if args.cmd == "extract":
        return cmd_extract(args)
    if not args.label:
        args.label = Path(args.jsonl).stem
    return cmd_analyze(args)


if __name__ == "__main__":
    sys.exit(main())
