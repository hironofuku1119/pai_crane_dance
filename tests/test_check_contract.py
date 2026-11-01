"""scripts/check_contract.py の固定（標準ライブラリ unittest）。

振付は choreography.json から作らず、テスト内で独立に合成する。
契約本体（contract.json）は実物を写して使い、必要な所だけ書き換える。
本物のファイルは書き換えない（検査は一時ディレクトリの写しに対して行う）。
"""
from __future__ import annotations

import copy
import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from bisect import bisect_right
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "ros" / "src" / "crane_dance"))
from crane_dance import contract_math  # noqa: E402
SCRIPT = REPO / "scripts" / "check_contract.py"
CONTRACT_JSON = json.loads((REPO / "contract" / "contract.json").read_text(encoding="utf-8"))
JOINTS = CONTRACT_JSON["joint_names"]
NAN = float("nan")
INF = float("inf")


def synth_choreography() -> dict:
    """独立に合成した振付（拍 0.7 s 間隔・4 拍で 1 小節・キーフレーム 4 件）。"""
    beats = [round(0.3 + 0.7 * k, 6) for k in range(28)]
    downs = beats[0::4]
    kfs = []
    for n, bi in enumerate((0, 8, 16, 24)):
        kfs.append(
            {
                "name": f"kf{n}",
                "label": f"仮{n}",
                "beat_index": bi,
                "time_s": beats[bi],
                "joints_rad": [0.0, 0.1 * n, 0.2, 0.0, 0.3],
                "primary_joints": [JOINTS[1]] if n % 2 else [],
                "measured_in_gazebo": True,
            }
        )
    return {
        "choreography_version": "0.0.0",
        "joint_names": JOINTS,
        "song": {
            "duration_s": 20.0,
            "beat_times_s": beats,
            "downbeat_times_s": downs,
            "source": {"kind": "measured", "method": "テスト用", "audio_sha256": "ab" * 32},
        },
        "keyframes": kfs,
    }


def synth_beats(ch: dict) -> dict:
    s = ch["song"]
    return {
        "bpm": 85.0,
        "beat_times_s": list(s["beat_times_s"]),
        "downbeat_times_s": list(s["downbeat_times_s"]),
        "duration_s": s["duration_s"],
        "sections": [{"start_s": 0.0, "end_s": s["duration_s"], "bpm": 85.0, "meter": "4/4"}],
        "warnings": [],
        "source": {
            "audio_sha256": s["source"]["audio_sha256"],
            "librosa_version": "x",
            "sample_rate_hz": 22050,
            "hop_length": 512,
            "downbeat_method": "テスト",
            "load_kwargs": {"sr": 22050, "mono": True, "res_type": "soxr_hq",
                            "offset": 0.0, "duration": None},
            "beat_track_kwargs": {"start_bpm": 120.0, "tightness": 100, "trim": True,
                                  "units": "frames"},
            "soxr_version": "x",
        },
    }


class Case(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "contract").mkdir()
        self.contract = copy.deepcopy(CONTRACT_JSON)
        self.ch = synth_choreography()
        self.stderr = ""

    def write_files(self, beats: dict | None = None) -> None:
        d = self.root / "contract"
        (d / "contract.json").write_text(json.dumps(self.contract, ensure_ascii=False),
                                         encoding="utf-8")
        (d / "choreography.json").write_text(json.dumps(self.ch, ensure_ascii=False),
                                             encoding="utf-8")
        if beats is not None:
            (self.root / "act" / "measurements").mkdir(parents=True, exist_ok=True)
            (self.root / "act" / "measurements" / "beats.json").write_text(
                json.dumps(beats, ensure_ascii=False), encoding="utf-8"
            )

    def invoke(self, *flags: str) -> tuple[int, str]:
        p = subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(self.root), *flags],
            capture_output=True,
            text=True,
        )
        self.stderr = p.stderr
        return p.returncode, p.stdout

    def run_check(self, *flags: str, beats: dict | None = None) -> tuple[int, str]:
        self.write_files(beats)
        return self.invoke(*flags)

    def assertFails(self, *flags: str, beats: dict | None = None) -> str:
        code, out = self.run_check(*flags, beats=beats)
        self.assertNotEqual(code, 0, out)
        self.assertIn("CONTRACT: FAIL", out.splitlines()[-1])
        self.assertNotIn("Traceback", self.stderr)  # 例外の表示でなく FAIL の行で止まる
        return out


class TestPass(Case):
    def test_good_input_passes(self):
        code, out = self.run_check()
        self.assertEqual(code, 0, out)
        self.assertEqual(out.splitlines()[-1], "CONTRACT: PASS")

    def test_real_files_pass(self):
        p = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout)
        self.assertEqual(p.stdout.splitlines()[-1], "CONTRACT: PASS")

    def test_missing_beats_is_unchecked_but_passes(self):
        code, out = self.run_check()
        self.assertEqual(code, 0, out)
        self.assertIn("beats: 未照合", out)

    def test_matching_beats_passes(self):
        code, out = self.run_check(beats=synth_beats(self.ch))
        self.assertEqual(code, 0, out)
        self.assertIn("beats: 照合した", out)


class TestGolden(Case):
    def test_changing_one_golden_row_fails(self):
        self.contract["golden"]["rows"][2]["sin"] = 0.5
        self.assertFails()

    def test_golden_table_c_has_first_interval_check_point(self):
        rows = [r for r in CONTRACT_JSON["golden"]["rows"] if r["table"] == "C"]
        taus = [r["tau"] for r in rows]
        self.assertIn(0.125, taus)  # 最初の区間（0〜0.5）の検算点（φ = π/2）
        self.assertTrue(any(t >= 1.0 for t in taus))  # 最後の区間の点も残す

    @staticmethod
    def _bad_taus(edge_of, interval_of) -> list[float]:
        """「端の拍を起点に一定の拍間隔」で計算する誤った実装で落ちる行の tau（表 C）。"""
        g = CONTRACT_JSON["golden"]
        bad = []
        for r in g["rows"]:
            if r["table"] != "C":
                continue
            tb = g["tables"]["C"]
            x = (r["tau"] - edge_of(tb)) / interval_of(tb)
            phi = 2 * math.pi * (x - math.floor(x))
            if abs(math.sin(phi) - r["sin"]) > 1e-6 or abs(math.cos(phi) - r["cos"]) > 1e-6:
                bad.append(r["tau"])
        return bad

    def test_first_interval_constant_bpm_fails_on_table_c(self):
        """最初の拍間隔だけで周期を決める実装は、表 C の後ろの区間の行で落ちる。"""
        bad = self._bad_taus(lambda tb: tb[0], lambda tb: tb[1] - tb[0])
        self.assertEqual(bad, [1.0, 2.0])

    def test_last_interval_constant_bpm_fails_on_first_interval_row(self):
        """最後の拍を起点に最後の拍間隔だけで周期を決める実装は、τ=0.125 の行で落ちる。"""
        bad = self._bad_taus(lambda tb: tb[-1], lambda tb: tb[-1] - tb[-2])
        self.assertEqual(bad, [0.125])

    def test_checker_catches_constant_bpm_golden(self):
        # 表 C の期待値を「一定 BPM」の答えに書き換えると、式の再計算と合わず失敗する
        for r in self.contract["golden"]["rows"]:
            if r["table"] == "C":
                r["sin"], r["cos"] = 0.0, 1.0
        self.assertFails()

    def test_checker_catches_last_interval_answer_on_first_row(self):
        # τ=0.125 の期待値を「最後の拍間隔で一定」の答え（φ = 1.25π）に書き換えると失敗する
        for r in self.contract["golden"]["rows"]:
            if r["table"] == "C" and r["tau"] == 0.125:
                r["sin"], r["cos"] = math.sin(1.25 * math.pi), math.cos(1.25 * math.pi)
        self.assertFails()

    def test_table_with_small_gap_fails(self):
        self.contract["golden"]["tables"]["C"] = [0.0, 0.05, 1.5]
        self.assertFails()


class TestChoreography(Case):
    def test_shifted_time_fails(self):
        self.ch["keyframes"][1]["time_s"] += 0.01
        self.assertFails()

    def test_time_on_non_downbeat_fails(self):
        # beat_times_s の中の小節頭でない拍に置く（time_s と beat_index は整合させる）
        self.ch["keyframes"][1]["beat_index"] = 9
        self.ch["keyframes"][1]["time_s"] = self.ch["song"]["beat_times_s"][9]
        self.assertFails()

    def test_joint1_nonzero_fails(self):
        self.ch["keyframes"][1]["joints_rad"][0] = 0.1
        self.assertFails()

    def test_duplicate_name_fails(self):
        self.ch["keyframes"][1]["name"] = self.ch["keyframes"][0]["name"]
        self.assertFails()

    def test_unknown_primary_joint_fails(self):
        self.ch["keyframes"][1]["primary_joints"] = ["no_such_joint"]
        self.assertFails()

    def test_out_of_range_fails(self):
        self.ch["keyframes"][1]["joints_rad"][4] = 5.0
        self.assertFails()

    def test_non_monotonic_beats_fails(self):
        self.ch["song"]["beat_times_s"][3] = self.ch["song"]["beat_times_s"][2]
        self.assertFails()

    def test_too_small_beat_gap_fails(self):
        b = self.ch["song"]["beat_times_s"]
        b[5] = b[4] + 0.05
        self.assertFails()

    def test_song_with_sections_fails(self):
        self.ch["song"]["sections"] = [{"start_s": 0, "end_s": 20.0, "bpm": 85, "meter": "4/4"}]
        self.assertFails()

    def test_non_bool_measured_fails(self):
        self.ch["keyframes"][0]["measured_in_gazebo"] = "yes"
        self.assertFails()


class TestJointNames(Case):
    def test_swapped_joint_order_between_files_fails(self):
        self.ch["joint_names"] = [JOINTS[1], JOINTS[0], *JOINTS[2:]]
        out = self.assertFails()
        self.assertIn("joint_names", out)

    def test_renamed_joint_between_files_fails(self):
        self.ch["joint_names"] = [*JOINTS[:4], "crane_plus_joint_other"]
        self.assertFails()

    def test_without_joint_names_passes_when_lengths_match(self):
        del self.ch["joint_names"]
        code, out = self.run_check()
        self.assertEqual(code, 0, out)

    def test_without_joint_names_fails_when_joints_rad_length_differs(self):
        del self.ch["joint_names"]
        self.ch["keyframes"][1]["joints_rad"] = [0.0, 0.1, 0.2, 0.0]
        self.assertFails()


class TestNpzDeclaration(Case):
    def test_second_dim_must_match_vector_dim(self):
        self.contract["npz"]["keys"]["observation_environment_state"]["shape"] = ["N", 4]
        out = self.assertFails()
        self.assertIn("vectors.observation_environment_state.dim", out)

    def test_timestamp_must_be_float64(self):
        self.contract["npz"]["keys"]["timestamp"]["dtype"] = "float32"
        self.assertFails()

    def test_beat_times_meta_must_be_float64(self):
        self.contract["npz"]["meta"]["beat_times_s"]["dtype"] = "float32"
        self.assertFails()

    def test_lerobot_name_must_match_exactly(self):
        # 下線区切りのままだと LeRobot は環境状態と判定しない
        keys = self.contract["npz"]["keys"]
        keys["observation_environment_state"]["lerobot_name"] = "observation_environment_state"
        out = self.assertFails()
        self.assertIn("lerobot_name", out)

    def test_missing_lerobot_name_fails(self):
        del self.contract["npz"]["keys"]["action"]["lerobot_name"]
        self.assertFails()

    def test_lerobot_names_are_the_dotted_ones(self):
        keys = CONTRACT_JSON["npz"]["keys"]
        self.assertEqual(keys["observation_state"]["lerobot_name"], "observation.state")
        self.assertEqual(
            keys["observation_environment_state"]["lerobot_name"], "observation.environment_state"
        )
        self.assertEqual(keys["action"]["lerobot_name"], "action")
        self.assertEqual(keys["timestamp"]["lerobot_name"], "timestamp")

    def test_choreography_sha256_meta_is_declared(self):
        del self.contract["npz"]["meta"]["choreography_sha256"]
        self.assertFails()

    def test_lerobot_task_must_be_dance_v1(self):
        self.contract["lerobot"]["task"] = "dance"
        self.assertFails()

    def test_old_global_storage_dtype_fails(self):
        self.contract["dtype_storage"] = "float32"
        self.assertFails()

    def test_old_tolerance_key_fails(self):
        self.contract["tolerance_abs"] = 1e-6
        self.assertFails()


class TestToleranceFromJson(Case):
    def test_loosening_env_tolerance_in_json_changes_the_verdict(self):
        # 許容誤差は json の欄から読む。golden の期待値を少しずらすと、広げたときだけ通る
        self.contract["golden"]["rows"][2]["sin"] = 1.0 - 5e-7
        self.contract["tolerance"]["env_abs"] = 1e-9
        self.assertFails()
        self.contract["tolerance"]["env_abs"] = 1e-6
        code, out = self.run_check()
        self.assertEqual(code, 0, out)

    def test_time_tolerance_in_json_is_used_for_beats_sync(self):
        b = synth_beats(self.ch)
        b["beat_times_s"][3] += 5e-7
        code, out = self.run_check(beats=b)
        self.assertEqual(code, 0, out)  # 既定の 1e-6 以内
        self.contract["tolerance"]["time_abs_s"] = 1e-7
        self.assertFails(beats=b)


class TestFiniteness(Case):
    """NaN・Infinity・文字列は、例外の表示でなく `CONTRACT: FAIL` の行で止まる。"""

    def check_fails(self, mutate, with_beats: bool = False) -> None:
        self.contract = copy.deepcopy(CONTRACT_JSON)
        self.ch = synth_choreography()
        beats = synth_beats(self.ch) if with_beats else None
        mutate(self.contract, self.ch, beats)
        self.assertFails(beats=beats)

    def test_contract_and_choreography_values(self):
        cases = {
            "golden tau が NaN": lambda c, h, b: c["golden"]["rows"][0].update(tau=NAN),
            "golden sin が Infinity": lambda c, h, b: c["golden"]["rows"][2].update(sin=INF),
            "golden tau が文字列": lambda c, h, b: c["golden"]["rows"][2].update(tau="0.125"),
            "golden 表の要素が NaN": lambda c, h, b: c["golden"]["tables"]["C"].__setitem__(1, NAN),
            "golden duration_s が NaN": lambda c, h, b: c["golden"].update(duration_s=NAN),
            "許容誤差 env_abs が NaN": lambda c, h, b: c["tolerance"].update(env_abs=NAN),
            "許容誤差 time_abs_s が Infinity": lambda c, h, b: c["tolerance"].update(time_abs_s=INF),
            "許容誤差が文字列": lambda c, h, b: c["tolerance"].update(env_abs="1e-6"),
            "downbeat が NaN": lambda c, h, b: h["song"]["downbeat_times_s"].__setitem__(1, NAN),
            "beat_times_s が Infinity": lambda c, h, b: h["song"]["beat_times_s"].__setitem__(
                2, INF
            ),
            "song.duration_s が NaN": lambda c, h, b: h["song"].update(duration_s=NAN),
            "time_s が NaN": lambda c, h, b: h["keyframes"][1].update(time_s=NAN),
            "joints_rad が NaN": lambda c, h, b: h["keyframes"][1]["joints_rad"].__setitem__(
                2, NAN
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name):
                self.check_fails(mutate)

    def test_beats_json_values(self):
        cases = {
            "sections.start_s が NaN": lambda c, h, b: b["sections"][0].update(start_s=NAN),
            "sections.end_s が Infinity": lambda c, h, b: b["sections"][0].update(end_s=INF),
            "sections.start_s が文字列": lambda c, h, b: b["sections"][0].update(start_s="0"),
            "beat_times_s が Infinity": lambda c, h, b: b["beat_times_s"].__setitem__(3, INF),
            "downbeat_times_s が NaN": lambda c, h, b: b["downbeat_times_s"].__setitem__(1, NAN),
            "duration_s が NaN": lambda c, h, b: b.update(duration_s=NAN),
        }
        for name, mutate in cases.items():
            with self.subTest(name):
                self.check_fails(mutate, with_beats=True)


class TestRequireMeasured(Case):
    def test_false_keyframe_fails_only_with_flag(self):
        self.ch["keyframes"][2]["measured_in_gazebo"] = False
        code, out = self.run_check()
        self.assertEqual(code, 0, out)
        self.assertFails("--require-measured", beats=synth_beats(self.ch))

    def test_provisional_source_fails_only_with_flag(self):
        self.ch["song"]["source"] = {"kind": "provisional", "method": "テスト", "audio_sha256": None}
        code, out = self.run_check()
        self.assertEqual(code, 0, out)
        self.assertFails("--require-measured")

    def test_all_measured_passes_with_flag(self):
        code, out = self.run_check("--require-measured", beats=synth_beats(self.ch))
        self.assertEqual(code, 0, out)

    def test_missing_beats_json_fails_with_flag(self):
        out = self.assertFails("--require-measured")
        self.assertIn("beats.json", out)

    def test_null_sha256_fails_with_flag(self):
        # kind を手で measured にしても、sha256 が null なら通らない
        self.ch["song"]["source"]["audio_sha256"] = None
        beats = synth_beats(self.ch)
        beats["source"]["audio_sha256"] = None
        out = self.assertFails("--require-measured", beats=beats)
        self.assertIn("audio_sha256", out)

    def test_malformed_sha256_fails_with_flag(self):
        for bad in ("AB" * 32, "ab" * 31, "zz" * 32):
            with self.subTest(bad):
                self.ch["song"]["source"]["audio_sha256"] = bad
                self.assertFails("--require-measured", beats=synth_beats(self.ch))

    def test_malformed_sha256_fails_even_without_flag(self):
        self.ch["song"]["source"]["audio_sha256"] = "not-a-hash"
        self.assertFails()


class TestBeatsSync(Case):
    def test_beat_times_mismatch_fails(self):
        b = synth_beats(self.ch)
        b["beat_times_s"][3] += 0.01
        self.assertFails(beats=b)

    def test_sha256_mismatch_fails(self):
        b = synth_beats(self.ch)
        b["source"]["audio_sha256"] = "cd" * 32
        self.assertFails(beats=b)

    def test_sections_gap_in_beats_fails(self):
        b = synth_beats(self.ch)
        b["sections"] = [
            {"start_s": 0.0, "end_s": 8.0, "bpm": 85.0, "meter": "4/4"},
            {"start_s": 9.0, "end_s": 20.0, "bpm": 85.0, "meter": "4/4"},
        ]
        self.assertFails(beats=b)

    def test_bad_meter_in_beats_fails(self):
        b = synth_beats(self.ch)
        b["sections"][0]["meter"] = "four"
        self.assertFails(beats=b)

    def test_missing_kwargs_key_in_beats_source_fails(self):
        b = synth_beats(self.ch)
        del b["source"]["load_kwargs"]["res_type"]
        out = self.assertFails(beats=b)
        self.assertIn("res_type", out)

    def test_missing_soxr_version_fails(self):
        b = synth_beats(self.ch)
        del b["source"]["soxr_version"]
        self.assertFails(beats=b)

    def test_empty_downbeat_method_fails(self):
        for bad in ("", "   ", None):
            with self.subTest(repr(bad)):
                b = synth_beats(self.ch)
                b["source"]["downbeat_method"] = bad
                self.assertFails(beats=b)


class TestTimestampStorage(unittest.TestCase):
    """npz を書く処理はここには無いので、保存と読み戻しの代わりの試験。

    格子 t_k = k/fps を、契約に宣言した dtype で保存して読み戻し、誤差が
    `tolerance.time_abs_s` 以内に収まることを struct で確かめる（numpy は使わない）。
    """

    FMT = {"float64": "d", "float32": "f"}
    FPS = CONTRACT_JSON["recording"]["fps"]
    TOL = CONTRACT_JSON["tolerance"]["time_abs_s"]

    @staticmethod
    def n_samples(duration_s: float, fps: int) -> int:
        return math.floor(duration_s * fps + 1e-9)  # contract.json の recording.n_samples

    def max_error(self, duration_s: float, fmt: str) -> float:
        n = self.n_samples(duration_s, self.FPS)
        grid = [k / self.FPS for k in range(n)]
        back = struct.unpack(f"<{n}{fmt}", struct.pack(f"<{n}{fmt}", *grid))
        return max(abs(a - b) for a, b in zip(grid, back))

    def test_grid_size_for_31_20_s_at_50_hz(self):
        self.assertEqual(self.FPS, 50)
        self.assertEqual(self.n_samples(31.20, self.FPS), 1560)

    def test_epsilon_in_n_formula_matters(self):
        # 4.35 s × 100 Hz は浮動小数で 434.99999999999994 になる。補正が無いと 1 本欠ける
        self.assertEqual(math.floor(4.35 * 100), 434)
        self.assertEqual(self.n_samples(4.35, 100), 435)

    def test_declared_timestamp_dtype_roundtrips_within_tolerance(self):
        declared = CONTRACT_JSON["npz"]["keys"]["timestamp"]["dtype"]
        self.assertEqual(declared, "float64")
        self.assertLessEqual(self.max_error(31.20, self.FMT[declared]), self.TOL)

    def test_float32_is_lossy_on_the_same_grid(self):
        # 31.20 s の格子（最大 31.18 s）では float32 の誤差は約 9.2e-7 で、許容誤差の 1e-6 に
        # ぎりぎり収まる。許容誤差を超えるのは、格子が 32 s を越える（旧版 32.69 s の曲）とき
        err = self.max_error(31.20, self.FMT["float32"])
        self.assertGreater(err, 0.0)
        self.assertGreater(self.max_error(32.69, self.FMT["float32"]), self.TOL)

    def test_float64_is_exact_on_the_longer_grid_too(self):
        self.assertLessEqual(self.max_error(32.69, self.FMT["float64"]), self.TOL)


# ---------------------------------------------------------------- 契約 0.2.0

GRID = CONTRACT_JSON["golden_grid"]
INTERP = CONTRACT_JSON["golden_interp"]
T0G = CONTRACT_JSON["golden_t0"]
TARGET_TOL = CONTRACT_JSON["tolerance"]["target_abs_rad"]


def wrong_steps(nows, t0_ns, fps, *, index=None, detect_backward=True, count_first=True):
    """grid_steps の誤った実装（反証用）。引数で一つずつ壊す。"""
    index = index or (lambda now: (now - t0_ns) * fps // contract_math.NS)
    emit, skipped, invalid_at, last_k, prev = [], 0, None, None, None
    for pos, now in enumerate(nows):
        k = index(now)
        if detect_backward and prev is not None and now < prev:
            invalid_at = pos
            break
        prev = now
        if k < 0:
            continue
        if last_k is None:
            skipped += k if count_first else 0
        elif k > last_k:
            skipped += k - last_k - 1
        else:
            continue
        emit.append(k)
        last_k = k
    return {"emit": emit, "skipped": skipped, "invalid_at": invalid_at}


def wrong_target_extrapolating(t, kfs):
    """端で保持せず、端の区間の傾きで延ばす誤った実装。"""
    times = [k["time_s"] for k in kfs]
    i = min(max(bisect_right(times, t) - 1, 0), len(kfs) - 2)
    a, b = kfs[i], kfs[i + 1]
    w = (t - a["time_s"]) / (b["time_s"] - a["time_s"])
    return [x + w * (y - x) for x, y in zip(a["joints_rad"], b["joints_rad"])]


def float_second_index(t0_ns, fps):
    """秒の float に直してから floor する誤った実装。"""
    return lambda now: math.floor((now / 1e9 - t0_ns / 1e9) * fps)


class TestContractMath(unittest.TestCase):
    def failing_cases(self, impl) -> list[str]:
        bad = []
        for case in GRID["cases"]:
            want = {k: case[k] for k in ("emit", "skipped", "invalid_at")}
            if impl(case["nows_ns"], GRID["t0_ns"], GRID["fps"]) != want:
                bad.append(case["name"])
        return bad

    def test_grid_steps_matches_golden(self):
        self.assertEqual(self.failing_cases(contract_math.grid_steps), [])

    def test_reference_toggles_match_golden_when_not_broken(self):
        self.assertEqual(self.failing_cases(wrong_steps), [])

    def test_ignoring_backward_time_fails_on_backward_case(self):
        bad = self.failing_cases(lambda n, t, f: wrong_steps(n, t, f, detect_backward=False))
        self.assertIn("後戻り", bad)

    def test_not_counting_first_k_fails_on_late_start_case(self):
        bad = self.failing_cases(lambda n, t, f: wrong_steps(n, t, f, count_first=False))
        self.assertIn("最初が遅れる", bad)

    def test_float_second_floor_fails_somewhere(self):
        bad = self.failing_cases(
            lambda n, t, f: wrong_steps(n, t, f, index=float_second_index(t, f)))
        print(f"\n[float 秒 floor の誤実装が落ちたケース] {bad}", file=sys.stderr)
        self.assertTrue(bad, "秒の float 経由の誤実装が golden_grid のどのケースでも落ちない")

    def test_grid_index_rejects_float_and_bool(self):
        with self.assertRaises(TypeError):
            contract_math.grid_index(1.0, 0, 50)
        with self.assertRaises(TypeError):
            contract_math.grid_index(0, 0.0, 50)
        with self.assertRaises(TypeError):
            contract_math.grid_index(True, 0, 50)
        with self.assertRaises(TypeError):
            contract_math.grid_time_ns(1, 0, 50.0)
        with self.assertRaises(TypeError):
            contract_math.grid_steps([0.5], 0, 50)

    def test_grid_time_ns_requires_fps_dividing_a_second(self):
        self.assertEqual(contract_math.grid_time_ns(3, 10, 50), 10 + 3 * 20_000_000)
        with self.assertRaises(ValueError):
            contract_math.grid_time_ns(1, 0, 30 + 7)

    def test_grid_index_is_negative_before_t0(self):
        self.assertEqual(contract_math.grid_index(-1, 0, 50), -1)
        self.assertEqual(contract_math.grid_index(19_999_999, 0, 50), 0)
        self.assertEqual(contract_math.grid_index(20_000_000, 0, 50), 1)

    def test_target_at_matches_golden(self):
        for r in INTERP["rows"]:
            got = contract_math.target_at(r["t"], INTERP["keyframes"])
            self.assertEqual(len(got), len(r["want"]))
            for a, b in zip(got, r["want"]):
                self.assertAlmostEqual(a, b, delta=TARGET_TOL, msg=f"t={r['t']}")

    def test_extrapolating_target_fails_at_both_ends(self):
        bad = []
        for r in INTERP["rows"]:
            got = wrong_target_extrapolating(r["t"], INTERP["keyframes"])
            if any(abs(a - b) > TARGET_TOL for a, b in zip(got, r["want"])):
                bad.append(r["t"])
        self.assertIn(0.0, bad)
        self.assertIn(9.0, bad)

    def t0_series(self, t0_ns, n=None, kfs=None):
        n = n or T0G["n"]
        kfs = kfs or INTERP["keyframes"]
        stamps = [contract_math.grid_time_ns(k, t0_ns, T0G["fps"]) for k in range(n)]
        pos = [contract_math.target_at(k / T0G["fps"], kfs) for k in range(n)]
        return stamps, pos

    def case_series(self, case):
        kfs = T0G["keyframes_same_ends"] if case.get("keyframes") == "same_ends" else INTERP["keyframes"]
        stamps, pos = self.t0_series(T0G["t0_ns"], kfs=kfs)
        drop = case.get("drop_first", 0)
        stamps, pos = stamps[drop:], pos[drop:]
        if "keep_last" in case:
            stamps, pos = stamps[-case["keep_last"]:], pos[-case["keep_last"]:]
        if "append_second_t0_ns" in case:
            s2, p2 = self.t0_series(case["append_second_t0_ns"], case.get("append_second_n"), kfs)
            stamps, pos = stamps + s2, pos + p2
        return stamps, pos, kfs

    def test_t0_consistent_matches_golden_cases(self):
        for case in T0G["cases"]:
            stamps, pos, kfs = self.case_series(case)
            got = contract_math.t0_consistent(stamps, pos, kfs, TARGET_TOL)
            self.assertEqual(got, case["consistent"], case["name"])

    def test_episode_complete_matches_golden_cases(self):
        for case in T0G["cases"]:
            stamps, pos, kfs = self.case_series(case)
            got = contract_math.episode_complete(stamps, T0G["n"], T0G["fps"])
            self.assertEqual(got, case["complete"], case["name"])
            accepted = contract_math.episode_accepted(stamps, pos, kfs, TARGET_TOL, T0G["n"], T0G["fps"])
            self.assertEqual(accepted, case["consistent"] and case["complete"], case["name"])

    def test_pose_alone_cannot_reject_review_examples(self):
        # 姿勢の照合だけでは正常と見えてしまい、完全性で初めて落ちる
        names = {"2 本目の止まっている先頭 1 秒を足した（姿勢だけでは見分けられない）",
                 "最後の止まっている区間だけ（姿勢だけでは見分けられない）"}
        cases = [c for c in T0G["cases"] if c["name"] in names]
        self.assertEqual(len(cases), 2)
        for case in cases:
            stamps, pos, kfs = self.case_series(case)
            self.assertTrue(contract_math.t0_consistent(stamps, pos, kfs, TARGET_TOL), case["name"])
            self.assertFalse(contract_math.episode_accepted(stamps, pos, kfs, TARGET_TOL,
                                                            T0G["n"], T0G["fps"]), case["name"])

    def test_episode_complete_rejects_gap_and_duplicate(self):
        stamps, _ = self.t0_series(T0G["t0_ns"])
        gap = stamps[:100] + stamps[101:] + [stamps[-1] + 20_000_000]
        dup = stamps[:100] + [stamps[99]] + stamps[100:-1]
        self.assertTrue(contract_math.episode_complete(stamps, T0G["n"], T0G["fps"]))
        self.assertFalse(contract_math.episode_complete(gap, T0G["n"], T0G["fps"]))
        self.assertFalse(contract_math.episode_complete(dup, T0G["n"], T0G["fps"]))
        self.assertFalse(contract_math.episode_complete([], T0G["n"], T0G["fps"]))
        self.assertFalse(contract_math.episode_complete([float(s) for s in stamps], T0G["n"], T0G["fps"]))

    def test_dropping_before_motion_starts_still_detected(self):
        # 40 本（0.8 s）落としても、動き出した後（1.0 s 以降）で食い違う
        stamps, pos = self.t0_series(T0G["t0_ns"])
        self.assertFalse(
            contract_math.t0_consistent(stamps[40:], pos[40:], INTERP["keyframes"], TARGET_TOL))
        # 始めの区間だけ（動き出す前の 40 本を落とした列の先頭 10 本）なら一致してしまう
        self.assertTrue(
            contract_math.t0_consistent(stamps[40:50], pos[40:50], INTERP["keyframes"],
                                        TARGET_TOL))

    def test_t0_consistent_rejects_wrong_joint_count(self):
        # 関節の数が振付と違う（4 つしか無い）列は、値が合っていても一致としない
        stamps, pos = self.t0_series(T0G["t0_ns"])
        short = [p[:4] for p in pos]
        self.assertFalse(contract_math.t0_consistent(stamps, short, INTERP["keyframes"], TARGET_TOL))

    def test_non_finite_position_is_not_accepted(self):
        # 本数と stamp が正常でも、関節位置の 1 要素が NaN・+inf・-inf なら認めない
        stamps, pos = self.t0_series(T0G["t0_ns"])
        kfs, n, fps = INTERP["keyframes"], T0G["n"], T0G["fps"]
        self.assertTrue(contract_math.episode_accepted(stamps, pos, kfs, TARGET_TOL, n, fps))
        for bad in (float("nan"), float("inf"), float("-inf")):
            broken = [list(p) for p in pos]
            broken[100][2] = bad
            self.assertFalse(contract_math.t0_consistent(stamps, broken, kfs, TARGET_TOL), bad)
            self.assertFalse(contract_math.episode_accepted(stamps, broken, kfs, TARGET_TOL, n, fps), bad)
        # 数でない値も認めない
        broken = [list(p) for p in pos]
        broken[0][0] = True
        self.assertFalse(contract_math.t0_consistent(stamps, broken, kfs, TARGET_TOL))

    def test_env_state_matches_golden_rows(self):
        g = CONTRACT_JSON["golden"]
        for r in g["rows"]:
            got = contract_math.env_state(g["tables"][r["table"]], r["tau"], g["duration_s"])
            for a, b in zip(got, (r["sin"], r["cos"], r["progress"])):
                self.assertAlmostEqual(a, b, delta=CONTRACT_JSON["tolerance"]["env_abs"])


class TestContractV02(Case):
    def test_real_contract_passes_and_prints_new_lines(self):
        code, out = self.run_check()
        self.assertEqual(code, 0, out)
        self.assertEqual(out.splitlines()[-1], "CONTRACT: PASS")
        self.assertIn("golden_grid: 7 ケース", out)
        self.assertIn("golden_interp: 9 行", out)
        self.assertIn("golden_t0: 7 ケース", out)

    def test_music_topic_back_fails(self):
        self.contract["topics"]["music"] = {"name": "/crane_dance/music",
                                            "type": "geometry_msgs/msg/Vector3Stamped"}
        self.assertFails()

    def test_type_without_msg_fails(self):
        self.contract["topics"]["joint_states"]["type"] = "sensor_msgs/JointState"
        self.assertFails()

    def test_clock_in_record_topics_fails(self):
        self.contract["recording"]["record_topics"].append("/clock")
        self.assertFails()

    def test_missing_recording_field_fails(self):
        del self.contract["recording"]["episode_rule"]
        self.assertFails()

    def test_joint_target_qos_changed_fails(self):
        self.contract["topics"]["joint_target"]["qos"]["depth"] = 1
        self.assertFails()

    def test_meta_fps_shape_changed_fails(self):
        self.contract["npz"]["meta"]["fps"]["shape"] = [1]
        self.assertFails()

    def test_sha_length_changed_fails(self):
        self.contract["npz"]["meta"]["choreography_sha256"]["length"] = 32
        self.assertFails()

    def test_missing_target_tolerance_fails(self):
        del self.contract["tolerance"]["target_abs_rad"]
        self.assertFails()

    def test_golden_grid_wrong_expectation_fails(self):
        self.contract["golden_grid"]["cases"][3]["skipped"] = 1
        self.assertFails()

    def test_golden_interp_wrong_expectation_fails(self):
        self.contract["golden_interp"]["rows"][6]["want"][2] = -0.4
        self.assertFails()

    def test_golden_t0_wrong_expectation_fails(self):
        self.contract["golden_t0"]["cases"][2]["consistent"] = True
        self.assertFails()

    def test_golden_t0_wrong_complete_fails(self):
        # 「最後の止まっている区間だけ」を完全と書き換えると、検査器が落とす
        self.contract["golden_t0"]["cases"][6]["complete"] = True
        self.assertFails()

    def test_golden_t0_missing_complete_fails(self):
        del self.contract["golden_t0"]["cases"][0]["complete"]
        self.assertFails()

# ---------------------------------------------------------------- --npz


def npy(descr: str, shape: tuple, payload: bytes, *, version=(1, 0), total_header=None) -> bytes:
    """標準ライブラリだけで npy を作る（版 1.0 の構造。64 バイト境界の詰め物）。"""
    text = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape!r}, }}"
    base = 10 + len(text) + 1
    pad = (-base) % 64 if total_header is None else max(total_header - len(text) - 1, 0)
    header = (text + " " * pad + "\n").encode("latin1")
    return b"\x93NUMPY" + bytes(version) + struct.pack("<H", len(header)) + header + payload


def u(text: str, width: int | None = None) -> bytes:
    width = width or len(text)
    return text.ljust(width, "\x00").encode("utf-32-le")


N_OK, DUR_OK, FPS_OK = 100, 2.0, 50


def good_members(version: str = "0.3.0", n: int = N_OK) -> dict:
    return {
        "observation_state.npy": npy("<f4", (n, 5), b"\x00" * (n * 5 * 4)),
        "observation_environment_state.npy": npy("<f4", (n, 3), b"\x00" * (n * 3 * 4)),
        "action.npy": npy("<f4", (n, 5), b"\x00" * (n * 5 * 4)),
        "timestamp.npy": npy("<f8", (n,), b"\x00" * (n * 8)),
        "t0_s.npy": npy("<f8", (), struct.pack("<d", 10.0)),
        "duration_s.npy": npy("<f8", (), struct.pack("<d", DUR_OK)),
        "fps.npy": npy("<i8", (), struct.pack("<q", FPS_OK)),
        "beat_times_s.npy": npy("<f8", (4,), struct.pack("<4d", 0.0, 0.5, 1.0, 1.5)),
        "contract_version.npy": npy(f"<U{len(version)}", (), u(version)),
        "choreography_sha256.npy": npy("<U64", (), u("ab" * 32)),
    }


def write_zip(path: Path, members, compression=zipfile.ZIP_STORED) -> None:
    """members は (名前, bytes) の並び（同じ名前を 2 回入れてよい）か辞書。"""
    items = list(members.items()) if isinstance(members, dict) else list(members)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with zipfile.ZipFile(path, "w", compression) as zf:
            for name, data in items:
                zf.writestr(name, data)


class TestNpzCheck(Case):
    def setUp(self) -> None:
        super().setUp()
        self.zip_path = self.root / "ep.npz"

    def check_npz(self, members, compression=zipfile.ZIP_STORED) -> tuple[int, str]:
        self.write_files()
        write_zip(self.zip_path, members, compression)
        code, out = self.invoke("--npz", str(self.zip_path))
        self.assertNotIn("CONTRACT:", out)
        self.assertNotIn("Traceback", self.stderr)
        return code, out

    def assertNpzPass(self, members, compression=zipfile.ZIP_STORED) -> None:
        code, out = self.check_npz(members, compression)
        self.assertEqual(code, 0, out)
        self.assertEqual(out.splitlines()[-1], "NPZ: PASS")

    def assertNpzFails(self, members) -> str:
        code, out = self.check_npz(members)
        self.assertEqual(code, 1, out)
        self.assertRegex(out.splitlines()[-1], r"^NPZ: FAIL \(\d+\)$")
        return out

    def test_stored_passes(self):
        self.assertNpzPass(good_members())

    def test_deflate_passes(self):
        self.assertNpzPass(good_members(), zipfile.ZIP_DEFLATED)

    def test_missing_key_fails(self):
        m = good_members()
        del m["action.npy"]
        self.assertNpzFails(m)

    def test_extra_key_fails(self):
        m = good_members()
        m["task.npy"] = npy("<i8", (), struct.pack("<q", 1))
        self.assertNpzFails(m)

    def test_fps_float_fails(self):
        m = good_members()
        m["fps.npy"] = npy("<f8", (), struct.pack("<d", 50.0))
        self.assertNpzFails(m)

    def test_t0_shape_1_fails(self):
        m = good_members()
        m["t0_s.npy"] = npy("<f8", (1,), struct.pack("<d", 10.0))
        self.assertNpzFails(m)

    def test_bytes_string_fails(self):
        m = good_members()
        m["choreography_sha256.npy"] = npy("|S64", (), b"a" * 64)
        self.assertNpzFails(m)

    def test_object_string_fails(self):
        m = good_members()
        m["choreography_sha256.npy"] = npy("|O", (), b"\x00" * 8)
        self.assertNpzFails(m)

    def test_oversized_header_fails(self):
        m = good_members()
        m["t0_s.npy"] = npy("<f8", (), struct.pack("<d", 10.0), total_header=20000)
        out = self.assertNpzFails(m)
        self.assertIn("ヘッダ長", out)

    def test_unknown_version_fails(self):
        m = good_members()
        m["t0_s.npy"] = npy("<f8", (), struct.pack("<d", 10.0), version=(9, 0))
        self.assertNpzFails(m)

    def test_nested_name_fails(self):
        m = good_members()
        m["a/b.npy"] = m.pop("action.npy")
        self.assertNpzFails(m)

    def test_wrong_n_fails(self):
        self.assertNpzFails(good_members(n=N_OK + 1))

    def test_header_only_body_fails(self):
        # action.npy をヘッダだけにする（本体が無い）
        m = good_members()
        m["action.npy"] = npy("<f4", (N_OK, 5), b"")
        out = self.assertNpzFails(m)
        self.assertIn("本体の大きさ", out)

    def test_truncated_body_fails_even_deflated(self):
        m = good_members()
        m["timestamp.npy"] = npy("<f8", (N_OK,), b"\x00" * (N_OK * 8 - 8))
        self.write_files()
        write_zip(self.zip_path, m, zipfile.ZIP_DEFLATED)
        code, out = self.invoke("--npz", str(self.zip_path))
        self.assertEqual(code, 1, out)
        self.assertIn("本体の大きさ", out)

    def test_extra_body_bytes_fail(self):
        m = good_members()
        m["fps.npy"] = npy("<i8", (), struct.pack("<q", FPS_OK) + b"\x00")
        self.assertNpzFails(m)

    def test_negative_dimension_fails(self):
        m = good_members()
        for key, cols in (("observation_state.npy", 5), ("observation_environment_state.npy", 3),
                          ("action.npy", 5)):
            m[key] = npy("<f4", (-1, cols), b"")
        m["timestamp.npy"] = npy("<f8", (-1,), b"")
        out = self.assertNpzFails(m)
        self.assertIn("負の次元", out)

    def test_negative_duration_fails(self):
        m = good_members()
        m["duration_s.npy"] = npy("<f8", (), struct.pack("<d", -0.02))
        out = self.assertNpzFails(m)
        self.assertIn("duration_s", out)

    def test_zero_length_episode_fails(self):
        # N=0 と duration_s=0.01（floor(0.01*50)=0 で式とは合う）でも、1 エピソードとは認めない
        m = good_members(n=0)
        m["duration_s.npy"] = npy("<f8", (), struct.pack("<d", 0.01))
        out = self.assertNpzFails(m)
        self.assertIn("N=0", out)

    def test_bad_unicode_descr_fails_without_exception(self):
        # <Ubad・<U0・<U のような不正な Unicode の型表記は、例外でなく NPZ: FAIL・終了コード 1
        # 「<U と数字 5000 桁」は、Python 3.11 以降の整数変換の桁数制限で int() が例外になる入力。
        # 「<U と数字 7 桁」は、上限（6 桁）のすぐ外側（境界）。
        for bad in ("<Ubad", "<U0", "<U", "<U-5", "<U" + "9" * 5000, "<U1000000"):
            m = good_members()
            m["contract_version.npy"] = npy(bad, (), u("0.2.0"))
            out = self.assertNpzFails(m)
            self.assertIn("contract_version の descr", out, bad)

    def test_duplicate_member_fails(self):
        items = list(good_members().items()) + [("action.npy", good_members()["action.npy"])]
        out = self.assertNpzFails(items)
        self.assertIn("重複", out)

    def test_big_endian_duration_fails(self):
        m = good_members()
        m["duration_s.npy"] = npy(">f8", (), struct.pack(">d", DUR_OK))
        self.assertNpzFails(m)

    def test_wrong_version_value_fails(self):
        self.assertNpzFails(good_members(version="0.1.0"))

    def test_wrong_fps_value_fails(self):
        m = good_members()
        m["fps.npy"] = npy("<i8", (), struct.pack("<q", 30))
        self.assertNpzFails(m)

    def test_sha_not_u64_fails(self):
        m = good_members()
        m["choreography_sha256.npy"] = npy("<U32", (), u("ab" * 16))
        self.assertNpzFails(m)

    def test_not_a_zip_fails(self):
        self.write_files()
        self.zip_path.write_bytes(b"not a zip")
        code, out = self.invoke("--npz", str(self.zip_path))
        self.assertEqual(code, 1)
        self.assertRegex(out.splitlines()[-1], r"^NPZ: FAIL")

    def test_data_keys_with_different_n_fail(self):
        m = good_members()
        m["action.npy"] = npy("<f4", (N_OK + 5, 5), b"\x00" * ((N_OK + 5) * 5 * 4))
        self.assertNpzFails(m)

    def test_header_with_extra_key_fails(self):
        m = good_members()
        raw = m["t0_s.npy"]
        text = "{'descr': '<f8', 'fortran_order': False, 'shape': (), 'x': 1}"
        header = (text + " " * ((-(10 + len(text) + 1)) % 64) + "\n").encode("latin1")
        m["t0_s.npy"] = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + raw[-8:]
        self.assertNpzFails(m)


# ---------------------------------------------------------------- npz の確定形（版とハッシュの対応）


class TestNpzSchemaPin(Case):
    def test_pass_and_count(self):
        code, out = self.run_check()
        self.assertEqual(code, 0, out)

    def test_missing_npz_schema_fails(self):
        del self.contract["npz_schema"]
        self.assertFails()

    def test_npz_field_change_fails(self):
        self.contract["npz"]["layout"] = self.contract["npz"]["layout"] + "x"
        self.assertFails()

    def _rehash_npz(self):
        import hashlib
        text = json.dumps(self.contract["npz"], sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))
        self.contract["npz_schema_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()

    def test_npz_change_with_matching_sha_but_same_schema_fails(self):
        # 形式を変えてハッシュだけ直し、整数を据え置くと FAIL
        self.contract["npz"]["layout"] = self.contract["npz"]["layout"] + "x"
        self._rehash_npz()
        out = self.assertFails()
        self.assertIn("確定形と違う", out)

    def test_npz_schema_bump_without_pinned_row_fails(self):
        # 版を上げても対応表に行が無ければ FAIL
        self.contract["npz"]["layout"] = self.contract["npz"]["layout"] + "x"
        self._rehash_npz()
        self.contract["npz_schema"] = 2
        out = self.assertFails()
        self.assertIn("NPZ_SCHEMA_SHA256", out)

    def test_pinned_table_matches_current_contract(self):
        sys.path.insert(0, str(REPO / "scripts"))
        import check_contract
        self.assertEqual(check_contract.NPZ_SCHEMA_SHA256[CONTRACT_JSON["npz_schema"]],
                         CONTRACT_JSON["npz_schema_sha256"])


if __name__ == "__main__":
    unittest.main()
