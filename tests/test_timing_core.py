"""scripts/timing_core.py・analyze_timing.py の検査（標準ライブラリの unittest・ROS なし）。

答えが分かっている合成の記録（実時間比 0.84・遅れ 0.10 s）を作り、道具がその値を出すかで縛る。
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "ros" / "src" / "crane_dance"))
import timing_core as tc  # noqa: E402
from crane_dance import plan as plan_mod  # noqa: E402

CONTRACT = json.loads((REPO / "contract" / "contract.json").read_text(encoding="utf-8"))
CHOREO = json.loads((REPO / "contract" / "choreography.json").read_text(encoding="utf-8"))
NAMES = CONTRACT["joint_names"]
NS = tc.NS
PLAN = plan_mod.from_choreography(CONTRACT, CHOREO)
FPS = PLAN.fps
TARGETS = [PLAN.target(k) for k in range(PLAN.n)]

RTF = 0.84
LAG_S = 0.10
S0 = 12_300_000_000          # 記録を始めたときのシム時刻
T0 = S0 + 2_000_000_000      # 再生の t0
W0 = 1_761_980_000 * NS      # 記録を始めたときの壁時計（エポック ns）


def wall_of(sim_ns: int) -> int:
    return W0 + round((sim_ns - S0) / RTF)


def synth(lag_s=LAG_S, epoch_stamps=False, drop_k=None, second_episode=False, bump_target=False,
          js_period_ms=10, clock_period_ms=10):
    """合成の記録（JSONL の行の list）。"""
    step = NS // FPS
    end = T0 + (PLAN.n - 1) * step
    recs = []
    s = S0
    while s <= end + 3 * NS:
        recs.append({"topic": "/clock", "wall_ns": wall_of(s), "stamp_ns": s})
        s += clock_period_ms * 1_000_000
    ks = list(range(PLAN.n)) + (list(range(50)) if second_episode else [])
    for i, k in enumerate(ks):
        if k == drop_k:
            continue
        st = T0 + k * step
        if i >= PLAN.n:  # 2 本目は 1 本目の後ろに受信される
            w = wall_of(end) + (i - PLAN.n + 1) * 20_000_000
        else:
            w = wall_of(st) + 2_000_000
        pos = list(TARGETS[k])
        if bump_target and k == 100:
            pos[0] += 0.01
        recs.append({"topic": "/crane_dance/joint_target", "wall_ns": w, "stamp_ns": st,
                     "names": NAMES, "positions": pos})
    s = T0 - NS
    while s <= end + NS:
        t = (s - T0) / NS
        actual = tc.grid_interp(TARGETS, FPS, t - lag_s)
        w = wall_of(s) + 1_000_000
        recs.append({"topic": "/joint_states", "wall_ns": w, "stamp_ns": w if epoch_stamps else s,
                     "names": list(reversed(NAMES)), "positions": list(reversed(actual))})
        s += js_period_ms * 1_000_000
    recs.sort(key=lambda r: r["wall_ns"])
    return recs


def run_all(recs):
    d = tc.split_records(recs, NAMES)
    cmap = tc.ClockMap(d["clock"])
    tgt = tc.check_targets(d["jt"], TARGETS, FPS)
    src = tc.check_stamp_source(d["js"], cmap)
    bars = tc.bar_ranges(CHOREO["song"]["downbeat_times_s"], FPS, PLAN.n)
    clock = tc.clock_report(cmap, d["jt"], tgt["t0_ns"], PLAN.n, FPS, bars)
    track = tc.tracking_report(d["js"], TARGETS, TARGETS, tgt["t0_ns"], FPS, bars)
    return d, cmap, tgt, src, clock, track


class OutputAnchors(unittest.TestCase):
    """答えが分かっている入力 → 出るべき値。"""

    @classmethod
    def setUpClass(cls):
        cls.d, cls.cmap, cls.tgt, cls.src, cls.clock, cls.track = run_all(synth())

    def test_checks_pass(self):
        self.assertTrue(self.tgt["ok"], self.tgt)
        self.assertTrue(self.src["ok"], self.src)
        self.assertEqual(self.tgt["t0_ns"], T0)

    def test_rtf_and_dance_wall_time(self):
        self.assertAlmostEqual(self.clock["rtf_clock"], RTF, delta=0.001)
        self.assertAlmostEqual(self.clock["dance_sim_s"], (PLAN.n - 1) / FPS, places=9)
        self.assertAlmostEqual(self.clock["dance_wall_s"], (PLAN.n - 1) / FPS / RTF, delta=0.02)
        self.assertAlmostEqual(self.clock["rtf_target"], RTF, delta=0.002)

    def test_every_bar_has_rtf(self):
        bars = self.clock["bars"]
        self.assertEqual([b["bar"] for b in bars], list(range(len(CHOREO["song"]["downbeat_times_s"]))))
        self.assertEqual(bars[0]["k_start"], 0)
        self.assertEqual(bars[-1]["k_end"], PLAN.n - 1)
        for b in bars:
            self.assertAlmostEqual(b["rtf_clock"], RTF, delta=0.002)

    def test_windows_rtf(self):
        for w in self.clock["windows"]:
            self.assertAlmostEqual(w["rtf"], RTF, delta=0.002)

    def test_lag_found_and_residual_zero(self):
        lag = self.track["lag"]
        self.assertAlmostEqual(lag["best_d_s"], LAG_S, delta=0.006)
        self.assertLess(lag["best_rms_arm"], 1e-9)
        self.assertGreater(lag["rms_arm_at_0"], 1e-3)  # 遅れのぶんだけ差が出ている
        self.assertFalse(lag["at_edge"])

    def test_learn_zero_when_plan_is_choreo(self):
        self.assertTrue(all(x["max"] == 0.0 for x in self.track["learn"]))

    def test_joint_order_follows_names(self):
        # /joint_states の名前を逆順にしても、関節ごとの差の最大は振付に沿う
        self.assertEqual(len(self.track["exec"]), len(NAMES))


class ChecksFail(unittest.TestCase):
    def test_epoch_like_stamps(self):
        d = tc.split_records(synth(epoch_stamps=True), NAMES)
        src = tc.check_stamp_source(d["js"], tc.ClockMap(d["clock"]))
        self.assertFalse(src["ok"])
        self.assertTrue(src["epoch_like"])

    def test_missing_grid_point(self):
        d = tc.split_records(synth(drop_k=700), NAMES)
        tgt = tc.check_targets(d["jt"], TARGETS, FPS)
        self.assertFalse(tgt["ok"])
        self.assertEqual(tgt["n_missing"], 1)
        self.assertEqual(tgt["gaps"], [700])

    def test_second_episode_mixed_in(self):
        d = tc.split_records(synth(second_episode=True), NAMES)
        tgt = tc.check_targets(d["jt"], TARGETS, FPS)
        self.assertFalse(tgt["ok"])
        self.assertGreater(tgt["backsteps"], 0)

    def test_target_not_matching_plan(self):
        d = tc.split_records(synth(bump_target=True), NAMES)
        tgt = tc.check_targets(d["jt"], TARGETS, FPS)
        self.assertFalse(tgt["ok"])
        self.assertAlmostEqual(tgt["plan_max_diff"], 0.01, places=9)

    def test_large_lag_hits_edge(self):
        _, _, _, _, _, track = run_all(synth(lag_s=0.5))
        self.assertTrue(track["lag"]["at_edge"])


def _run_cli(tmp, recs):
    j = Path(tmp) / "run.jsonl"
    j.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    out = Path(tmp) / "out"
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "analyze_timing.py"), "analyze", str(j), "--out", str(out)],
                       capture_output=True, text=True)
    return r, out


class Coverage(unittest.TestCase):
    """踊りの区間を覆わない記録を PASS にしない。"""

    def test_no_joint_states_after_t0_fails(self):
        recs = [r for r in synth() if not (r["topic"] == "/joint_states" and r["stamp_ns"] >= T0)]
        with tempfile.TemporaryDirectory() as tmp:
            r, out = _run_cli(tmp, recs)
            self.assertEqual(r.returncode, 1, r.stdout[-400:])
            self.assertTrue(r.stdout.strip().splitlines()[-1].startswith("TIMING: FAIL"))
            self.assertIn("無効", (out / "summary.md").read_text(encoding="utf-8"))

    def test_clock_only_at_start_fails(self):
        recs = synth()
        # t0 の少し前の /clock 2 点だけと、その間の /joint_states 1 本（全体を外挿すれば実時間比は出てしまう）
        clocks = [r for r in recs if r["topic"] == "/clock" and T0 - 20_000_000 <= r["stamp_ns"] <= T0 - 10_000_000]
        js_one = [r for r in recs if r["topic"] == "/joint_states" and clocks[0]["stamp_ns"] <= r["stamp_ns"] <= clocks[-1]["stamp_ns"]][:1]
        self.assertEqual((len(clocks), len(js_one)), (2, 1))
        jt = [r for r in recs if r["topic"] == "/crane_dance/joint_target"]
        with tempfile.TemporaryDirectory() as tmp:
            r, out = _run_cli(tmp, sorted(clocks + js_one + jt, key=lambda x: x["wall_ns"]))
            self.assertEqual(r.returncode, 1, r.stdout[-400:])
            self.assertTrue(r.stdout.strip().splitlines()[-1].startswith("TIMING: FAIL"))
            self.assertIn("無効", (out / "summary.md").read_text(encoding="utf-8"))

    def test_full_record_still_passes(self):
        d = tc.split_records(synth(), NAMES)
        cov = tc.check_coverage(tc.ClockMap(d["clock"]), d["js"], T0, PLAN.n, FPS)
        self.assertTrue(cov["ok"], cov)


class LagCommonSamples(unittest.TestCase):
    """どの遅れの候補でも同じ標本で比べる。"""

    def test_initial_transient_does_not_fake_a_lag(self):
        fps = 50
        plan = [[k / fps] for k in range(fps + 1)]  # 0〜1 秒のランプ
        samples = []
        for i in range(101):  # 100 Hz・実際は遅れ 0
            t = i / 100
            a = [t + (1.0 if i == 0 else 0.0)]  # 先頭 1 点だけ +1 rad の過渡誤差
            samples.append({"t": t, "actual": a})
        lag = tc.lag_scan(samples, plan, fps)
        self.assertEqual(lag["best_d_s"], 0.0)

    def test_no_common_samples_is_unknown(self):
        lag = tc.lag_scan([{"t": 0.1, "actual": [0.0]}], [[0.0], [0.0]], 50)
        self.assertIsNone(lag["best_d_s"])


class Jumps(unittest.TestCase):
    def test_plan_jump_found_where_planted(self):
        plan = [[0.0, 0.0] for _ in range(100)]
        for k in range(60, 100):
            plan[k][1] = 0.3  # k=60 で 0.3 rad の段差
        j = tc.plan_jumps(plan, 50)
        self.assertEqual(j["count"], 1)
        self.assertEqual((j["top"][0]["k"], j["top"][0]["joint"]), (60, 1))
        self.assertAlmostEqual(j["top"][0]["t"], 1.2)

    def test_t_at_max_points_to_the_jump(self):
        # 計画に段差があり、実際はそれを 0.1 s 遅れて追う → 実行の差の最大は段差の直後
        global TARGETS
        saved = TARGETS
        try:
            TARGETS = [list(t) for t in saved]
            for k in range(800, len(TARGETS)):
                TARGETS[k][2] += 0.3
            _, _, _, _, _, track = run_all(synth())
            self.assertAlmostEqual(track["exec"][2]["t_at_max"], 16.0, delta=0.11)
            self.assertGreaterEqual(track["plan_jumps"]["count"], 1)
        finally:
            TARGETS = saved


class Pieces(unittest.TestCase):
    def test_grid_interp(self):
        t = [[0.0], [1.0], [3.0]]
        self.assertEqual(tc.grid_interp(t, 10, -1), [0.0])
        self.assertAlmostEqual(tc.grid_interp(t, 10, 0.15)[0], 2.0)
        self.assertEqual(tc.grid_interp(t, 10, 9), [3.0])

    def test_player_log(self):
        lines = ["100.000 5.000 t0 = 123（シム時刻）・壁時計 x", "101.500 6.500 小節 1（k=96）",
                 "140.000 45.000 終了（飛んだ格子 0 本）", "ゴミの行"]
        p = tc.parse_player_log(lines)
        self.assertEqual(p["t0_wall_s"], 100.0)
        self.assertEqual(p["bars"], [{"bar": 1, "k": 96, "wall_s": 101.5}])
        self.assertEqual(p["end_wall_s"], 140.0)


class CommandLine(unittest.TestCase):
    def test_analyze_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            j = Path(tmp) / "run.jsonl"
            j.write_text("\n".join(json.dumps(r) for r in synth()) + "\n", encoding="utf-8")
            out = Path(tmp) / "out"
            r = subprocess.run([sys.executable, str(REPO / "scripts" / "analyze_timing.py"), "analyze", str(j),
                                "--out", str(out), "--label", "合成"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual(r.stdout.strip().splitlines()[-1], "TIMING: PASS")
            for name in ("summary.md", "clock_bars.csv", "rtf_windows.csv", "tracking_bars.csv",
                         "lag_curve.csv", "samples.csv", "tracking.svg", "error.svg", "clock.svg"):
                self.assertTrue((out / name).exists(), name)
            s = (out / "summary.md").read_text(encoding="utf-8")
            self.assertIn("実時間比 **0.840**", s)
            self.assertNotIn("無効", s)

    def test_analyze_reports_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            j = Path(tmp) / "run.jsonl"
            j.write_text("\n".join(json.dumps(r) for r in synth(drop_k=10)) + "\n", encoding="utf-8")
            r = subprocess.run([sys.executable, str(REPO / "scripts" / "analyze_timing.py"), "analyze", str(j),
                                "--out", str(Path(tmp) / "out")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)
            self.assertEqual(r.stdout.strip().splitlines()[-1], "TIMING: FAIL (1)")
            self.assertIn("無効", (Path(tmp) / "out" / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
