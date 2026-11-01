"""crane_dance.plan の検査（標準ライブラリの unittest・ROS なし）。"""
import json
import math
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "ros" / "src" / "crane_dance"))
from crane_dance import npz_io, plan  # noqa: E402

CONTRACT = json.loads((REPO / "contract" / "contract.json").read_text(encoding="utf-8"))
CHOREO = json.loads((REPO / "contract" / "choreography.json").read_text(encoding="utf-8"))
FPS = CONTRACT["recording"]["fps"]


def npy(descr, shape, payload) -> bytes:
    text = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape!r}, }}"
    base = 10 + len(text) + 1
    header = (text + " " * ((-base) % 64) + "\n").encode("latin1")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + payload


def u(text, width=None):
    return text.ljust(width or len(text), "\x00").encode("utf-32-le")


SHA = "ab" * 32


def small_choreo(n, fps=FPS):
    """小さいテスト用 npz（N=n）に対応する小さい振付：曲長だけ n / fps にした今の振付の写し。"""
    c = json.loads(json.dumps(CHOREO))
    c["song"]["duration_s"] = n / fps
    return c


def make_npz(path, n=4, fps=FPS, extra=False, action_cols=5, version=CONTRACT["contract_version"], act_descr="<f4",
             obs_rows=None, bad=None, ver_descr=None, ts=None, dur=None):
    act = [[0.1 * (k % 4) + 0.01 * j for j in range(action_cols)] for k in range(n)]  # 可動範囲の中に収める
    if bad is not None:
        act[1][2] = bad
    flat = [x for r in act for x in r]
    ts = [k / fps for k in range(n)] if ts is None else ts  # 契約：timestamp[k] = k / fps
    members = {
        "observation_state.npy": npy("<f4", (obs_rows or n, 5), b"\x00" * ((obs_rows or n) * 5 * 4)),
        "observation_environment_state.npy": npy("<f4", (n, 3), b"\x00" * (n * 3 * 4)),
        "action.npy": npy(act_descr, (n, action_cols),
                          struct.pack(f"<{len(flat)}{'d' if act_descr == '<f8' else 'f'}", *flat)),
        "timestamp.npy": npy("<f8", (len(ts),), struct.pack(f"<{len(ts)}d", *ts)),
        "t0_s.npy": npy("<f8", (), struct.pack("<d", 10.0)),
        "duration_s.npy": npy("<f8", (), struct.pack("<d", n / fps if dur is None else dur)),
        "fps.npy": npy("<i8", (), struct.pack("<q", fps)),
        "beat_times_s.npy": npy("<f8", (2,), struct.pack("<2d", 0.0, 0.5)),
        "contract_version.npy": npy(ver_descr or f"<U{len(version)}", (), u(version)),
        "choreography_sha256.npy": npy("<U64", (), u(SHA)),
    }
    if extra:
        members["extra.npy"] = npy("<f8", (), struct.pack("<d", 1.0))
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return act


class FromChoreographyTest(unittest.TestCase):
    def test_n_and_initial(self):
        p = plan.from_choreography(CONTRACT, CHOREO)
        self.assertEqual(p.fps, FPS)
        self.assertEqual(p.n, math.floor(CHOREO["song"]["duration_s"] * FPS + 1e-9))
        self.assertEqual(p.kind, "お手本（振付ファイル）")
        self.assertEqual(p.joint_names, CONTRACT["joint_names"])
        self.assertEqual(p.initial(), CHOREO["keyframes"][0]["joints_rad"])
        with self.assertRaises(IndexError):
            p.target(p.n)
        with self.assertRaises(IndexError):
            p.target(-1)

    def test_target_at_keyframe_times(self):
        p = plan.from_choreography(CONTRACT, CHOREO)
        for kf in CHOREO["keyframes"]:
            k = round(kf["time_s"] * FPS)
            if k >= p.n:
                continue
            for got, want in zip(p.target(k), kf["joints_rad"]):
                self.assertAlmostEqual(got, want, delta=0.02)

    def test_golden_interp(self):
        g = CONTRACT["golden_interp"]
        choreo = {"song": {"duration_s": 10.0, "downbeat_times_s": []}, "keyframes": g["keyframes"]}
        p = plan.from_choreography(CONTRACT, choreo)
        checked = 0
        for row in g["rows"]:
            x = row["t"] * FPS
            if abs(x - round(x)) > 1e-9 or round(x) >= p.n:
                continue
            for got, want in zip(p.target(round(x)), row["want"]):
                self.assertAlmostEqual(got, want, delta=1e-6)
            checked += 1
        self.assertGreater(checked, 0)


class FromNpzTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def load(self, name="run.npz", **kw):
        path = self.dir / name
        act = make_npz(path, **kw)
        return act, npz_io.read_npz(path), name

    def choreo(self, n=4):
        return small_choreo(n)

    def test_targets_and_kinds(self):
        for name, kind in (("x_open.npz", "開ループの出力"), ("x_closed.npz", "閉ループの出力"),
                           ("x.npz", "種類不明"), ("closed_open.npz", "閉ループの出力")):
            act, npz, fname = self.load(name)
            p = plan.from_npz(CONTRACT, self.choreo(), npz, fname, choreo_sha256=SHA)
            self.assertEqual(p.kind, kind)
            self.assertEqual(p.n, 4)
            self.assertEqual(p.fps, FPS)
            for k in range(4):
                for got, want in zip(p.target(k), act[k]):
                    self.assertAlmostEqual(got, want, places=5)
            self.assertEqual(p.initial(), p.target(0))

    def test_reject(self):
        _, npz, fname = self.load(extra=True)
        with self.assertRaises(ValueError):
            plan.from_npz(CONTRACT, self.choreo(), npz, fname, choreo_sha256=SHA)
        _, npz, fname = self.load(fps=25)
        with self.assertRaises(ValueError):
            plan.from_npz(CONTRACT, self.choreo(), npz, fname, choreo_sha256=SHA)
        _, npz, fname = self.load(action_cols=4)
        with self.assertRaises(ValueError):
            plan.from_npz(CONTRACT, self.choreo(), npz, fname, choreo_sha256=SHA)

    def test_contract_version_mismatch_is_only_a_warning(self):
        # 判断：契約の版の違いだけは警告にして再生は続ける（形が同じなら支障がない）
        _, npz, fname = self.load(version="0.1.0")
        p = plan.from_npz(CONTRACT, self.choreo(), npz, fname, choreo_sha256=SHA)
        self.assertEqual(len(p.warnings), 1)
        self.assertIn("0.1.0", p.warnings[0])
        _, npz, fname = self.load()
        self.assertEqual(plan.from_npz(CONTRACT, self.choreo(), npz, fname, choreo_sha256=SHA).warnings, [])

    def test_reject_contract_violations(self):
        # float64 の action・行数の不一致・NaN・inf・範囲外・振付の sha の違いは、再生の前に拒む
        cases = [
            ("float64 の action", dict(act_descr="<f8")),
            ("行数の不一致", dict(obs_rows=3)),
            ("NaN", dict(bad=float("nan"))),
            ("inf", dict(bad=float("inf"))),
            ("範囲外", dict(bad=-3.0)),
        ]
        for label, kw in cases:
            _, npz, fname = self.load(**kw)
            with self.assertRaises(ValueError, msg=label):
                plan.from_npz(CONTRACT, self.choreo(), npz, fname, choreo_sha256=SHA)
        _, npz, fname = self.load()
        with self.assertRaises(ValueError):
            plan.from_npz(CONTRACT, self.choreo(), npz, fname, choreo_sha256="cd" * 32)

    def test_reject_unicode_descr_with_too_many_digits(self):
        # <U と 7 桁（上限 6 桁のすぐ外）・5000 桁は、npz_io の段で例外でなく ValueError
        for descr in ("<U1000000", "<U" + "9" * 5000):
            path = self.dir / "u.npz"
            make_npz(path, ver_descr=descr)
            with self.assertRaises(ValueError, msg=descr[:12]):
                npz_io.read_npz(path)

    def test_real_choreography_npz_shape_passes(self):
        # 実物の振付と同じ長さ（N=1560）の正常な npz は通る（N が floor(duration × fps) と一致）
        n = math.floor(CHOREO["song"]["duration_s"] * FPS + 1e-9)
        _, npz, fname = self.load(n=n, dur=CHOREO["song"]["duration_s"])
        p = plan.from_npz(CONTRACT, CHOREO, npz, fname, choreo_sha256=SHA)
        self.assertEqual(p.n, n)

    def test_reject_timestamp_and_song_length(self):
        # timestamp の全ゼロ・ずれ・不連続と、今の振付と違う曲長は、初期姿勢への指令より前に拒む
        step = 1 / FPS
        cases = [
            ("全ゼロ", dict(ts=[0.0] * 4), self.choreo()),
            ("ずれ（全体が 1 格子遅い）", dict(ts=[(k + 1) * step for k in range(4)]), self.choreo()),
            ("不連続（3 行目が飛ぶ）", dict(ts=[0.0, step, 3 * step, 4 * step]), self.choreo()),
            ("わずかなずれ（許容 1e-6 の外）", dict(ts=[0.0, step, 2 * step + 2e-6, 3 * step]), self.choreo()),
            ("曲長が今の振付より短い", dict(), small_choreo(6)),
            ("曲長が今の振付より長い", dict(n=8), self.choreo()),
        ]
        for label, kw, choreo in cases:
            _, npz, fname = self.load(**kw)
            with self.assertRaises(ValueError, msg=label):
                plan.from_npz(CONTRACT, choreo, npz, fname, choreo_sha256=SHA)
        # 同じ sha でも、npz の duration だけ短くしたものは今の振付（実物）に対して拒む
        _, npz, fname = self.load()
        with self.assertRaises(ValueError):
            plan.from_npz(CONTRACT, CHOREO, npz, fname, choreo_sha256=SHA)
        # 許容の内側（1e-7 のずれ）は通る
        _, npz, fname = self.load(ts=[0.0, step + 1e-7, 2 * step, 3 * step])
        self.assertEqual(plan.from_npz(CONTRACT, self.choreo(), npz, fname, choreo_sha256=SHA).n, 4)


class BarHeadsTest(unittest.TestCase):
    def test_real_choreography(self):
        heads = plan.bar_heads(CHOREO, FPS)
        downbeats = CHOREO["song"]["downbeat_times_s"]
        self.assertEqual(len(heads), len(downbeats))
        self.assertNotIn(0, heads)
        self.assertEqual(heads[5], 0)
        self.assertEqual(heads[round(downbeats[1] * FPS)], 1)
        self.assertEqual(plan.from_choreography(CONTRACT, CHOREO).bar_heads, heads)


class FullTrajectoryTest(unittest.TestCase):
    def test_choreography(self):
        pl = plan.from_choreography(CONTRACT, CHOREO)
        pts = plan.full_trajectory(pl)
        step = 10**9 // FPS
        self.assertEqual(len(pts), pl.n + 1)
        ts = [t for t, _ in pts]
        self.assertTrue(all(isinstance(t, int) for t in ts))
        self.assertEqual(ts[:-1], [k * step for k in range(pl.n)])
        self.assertTrue(all(a < b for a, b in zip(ts, ts[1:])))
        self.assertEqual(pts[0][1], pl.initial())
        self.assertEqual(ts[-1], (pl.n - 1) * step + 2 * 10**9)
        self.assertEqual(pts[-1][1], pl.initial())

    def test_npz_exact(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "run.npz"
            act = make_npz(path, n=6)
            pl = plan.from_npz(CONTRACT, small_choreo(6), npz_io.read_npz(str(path)), "run.npz",
                               choreo_sha256=SHA)
            pts = plan.full_trajectory(pl)
        self.assertEqual(len(pts), 7)
        for k in range(6):
            self.assertEqual(pts[k][1], [float(x) for x in pl.target(k)])
            # npz は float32 で保存されるので、読んだ値（plan.target）を float にしただけであること
            for a, b in zip(pts[k][1], act[k]):
                self.assertAlmostEqual(a, b, places=5)

    def test_controller_points(self):
        pl = plan.from_choreography(CONTRACT, CHOREO)
        pts = plan.full_trajectory(pl)
        ctl = CONTRACT["controller"]
        names = CONTRACT["joint_names"]
        cp = plan.controller_points(CONTRACT, pts)
        self.assertEqual(cp["arm"][0], ctl["arm_controller_joints"])
        self.assertEqual(cp["gripper"][0], ctl["gripper_controller_joints"])
        self.assertEqual(len(cp["arm"][0]), 4)
        self.assertEqual(len(cp["gripper"][0]), 1)
        for (t, vals), (t0, pos) in zip(cp["arm"][1], pts):
            self.assertEqual(t, t0)
            self.assertEqual(vals, [pos[names.index(j)] for j in ctl["arm_controller_joints"]])
        for (t, vals), (t0, pos) in zip(cp["gripper"][1], pts):
            self.assertEqual(vals, [pos[names.index(j)] for j in ctl["gripper_controller_joints"]])

    def test_controller_points_reordered_contract(self):
        names = CONTRACT["joint_names"]
        order = list(reversed(range(len(names))))
        swapped = dict(CONTRACT)
        swapped["joint_names"] = [names[i] for i in order]
        pos = [float(i) for i in range(len(names))]  # 元の順での値
        pos_swapped = [pos[i] for i in order]  # swapped の順に並べた値
        pts = [(0, pos_swapped)]
        cp = plan.controller_points(swapped, pts)
        ctl = CONTRACT["controller"]
        self.assertEqual(cp["arm"][1][0][1], [pos[names.index(j)] for j in ctl["arm_controller_joints"]])
        self.assertEqual(cp["gripper"][1][0][1], [pos[names.index(j)] for j in ctl["gripper_controller_joints"]])


if __name__ == "__main__":
    unittest.main()
