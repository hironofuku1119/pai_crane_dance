"""書き出し（時刻の割り当て・npz（契約の検査器）・YAML）。"""
import json
import re
import subprocess
import sys

import numpy as np
import pytest
from helpers_ckpt import ACT_DIR, CHOREO_PATH, CONTRACT_PATH, make_checkpoint, make_dataset

from crane_dance_act import evaluate, export, state_mode, synth

REPO_ROOT = ACT_DIR.parent
FPS = 50


@pytest.fixture(scope="module")
def contract():
    return synth.load_contract(CONTRACT_PATH)


@pytest.fixture(scope="module")
def choreo():
    return synth.load_choreography(CHOREO_PATH)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("exp")
    root, _ = make_dataset(tmp)
    return make_checkpoint(tmp, root, state_mode.lagged(), chunk=10)


# ---------------------------------------------------------------- 時刻の割り当て（golden）
def test_trajectory_times_golden():
    assert export.trajectory_times_ns(4, 50) == [0, 20_000_000, 40_000_000, 60_000_000]
    assert export.trajectory_times_ns(3, 50, offset_s=0.5) == [500_000_000, 520_000_000, 540_000_000]
    s = export.trajectory_times_s(1560, 50)
    assert s.dtype == np.float64 and s[0] == 0.0
    assert np.allclose(s, np.arange(1560) / 50, atol=1e-12)  # t_k = k/fps（(k+1)/fps ではない）
    for bad in (0, 3, 7, 3_000_000_000):
        with pytest.raises(ValueError):
            export.trajectory_times_ns(2, bad)
    with pytest.raises(ValueError):
        export.trajectory_times_ns(2, 50, offset_s=float("nan"))


def test_yaml_uses_the_same_times_and_keeps_last_point():
    for n in (6, 7):
        action = np.linspace(0, 0.5, n * 5).reshape(n, 5)
        text = export.trajectory_yaml(["a", "b"], [0, 1], action, FPS)
        assert text.endswith("\n") and text.count("\n") == 1  # 1 行 YAML
        ks = list(range(0, n, 2)) + ([n - 1] if (n - 1) % 2 else [])
        got = [(int(a), int(b)) for a, b in re.findall(r"sec: (\d+), nanosec: (\d+)", text)]
        want = export.trajectory_times_ns(n, FPS)
        assert got == [(want[k] // 10**9, want[k] % 10**9) for k in ks]
        assert got[-1] == (want[-1] // 10**9, want[-1] % 10**9)  # 最後の点を残す
        assert text.startswith("{joint_names: [a, b], points: [{positions: [0, 0.0")
    assert "nanosec: 0}" in export.trajectory_yaml(["a"], [0], np.zeros((4, 5)), FPS).split("points:")[1].split("},")[0]
    with pytest.raises(export.ExportError):
        export.trajectory_yaml(["a"], [0], np.zeros((4, 5)), 30)  # 25 Hz に割り切れない


def test_yaml_size_for_full_reference_is_within_limit(contract, choreo):
    ep = synth.build_episode(contract, choreo, synth.file_sha256(CHOREO_PATH))
    for _, names, cols in export.yaml_parts(contract):
        text = export.trajectory_yaml(names, cols, ep["action"].astype(np.float64), FPS)
        assert len(text.encode()) <= export.YAML_BYTE_LIMIT


def test_check_action(contract):
    ok = np.zeros((3, 5))
    assert export.check_action(ok, contract) == []
    bad = ok.copy()
    bad[1, 2] = np.nan
    assert export.check_action(bad, contract)
    bad[1, 2] = np.inf
    assert export.check_action(bad, contract)
    bad[1, 2] = 9.0
    assert export.check_action(bad, contract)
    assert export.check_action(np.zeros((3, 4)), contract)


def test_build_npz_episode_rejects_bad_input(contract, choreo):
    ep = synth.build_episode(contract, choreo, synth.file_sha256(CHOREO_PATH))
    sha = synth.file_sha256(CHOREO_PATH)
    args = (ep["observation_state"], ep["observation_environment_state"], contract, choreo, sha)
    good = ep["action"].astype(np.float64)
    assert export.build_npz_episode(good, *args)["action"].dtype == np.float32
    nan = good.copy()
    nan[5, 0] = np.nan
    with pytest.raises(export.ExportError):
        export.build_npz_episode(nan, *args)
    with pytest.raises(export.ExportError):
        export.build_npz_episode(good[:-1], *args)  # フレーム数が契約の N と違う
    with pytest.raises(export.ExportError):
        export.build_npz_episode(good, *args, offset_s=0.1)


# ---------------------------------------------------------------- 書き出し（小さい本物の方策・CPU）
def _run_check_contract(npz):
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_contract.py"), "--npz", str(npz)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )  # fmt: skip
    return r.returncode, r.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("mode", ["open", "closed"])
def test_write_export_npz_passes_check_contract(tmp_path, ckpt, contract, mode):
    out = tmp_path / f"x-{mode}"
    meta = export.write_export(export.ExportSpec(checkpoint=ckpt, out_dir=out, mode=mode, m=10, tau=0.06, device="cpu"))
    npz = out / "trajectory.npz"
    rc, last = _run_check_contract(npz)
    assert (rc, last) == (0, "NPZ: PASS")
    with np.load(npz, allow_pickle=False) as z:
        assert sorted(z.files) == sorted(
            ["observation_state", "observation_environment_state", "action", "timestamp",
             "t0_s", "duration_s", "fps", "beat_times_s", "contract_version", "choreography_sha256"]
        )  # fmt: skip
        n = len(z["timestamp"])
        assert np.allclose(z["timestamp"], np.arange(n) / FPS, atol=1e-9)
        assert str(z["contract_version"]) == contract["contract_version"]
        assert str(z["choreography_sha256"]) == synth.file_sha256(CHOREO_PATH)
        act = z["action"]
    assert np.all(np.isfinite(act))
    # action は方策の出力そのもの（同じ呼び方で作った列と一致・float32 に丸めた分だけ違う）
    v = evaluate.validation_episodes(contract, synth.load_choreography(CHOREO_PATH), synth.file_sha256(CHOREO_PATH), 0, 20, 1)[0]
    lp = evaluate.load_policy(ckpt, "cpu")
    col = {"kind": "open", "m": 10} if mode == "open" else {"kind": "closed", "m": 10, "tau": 0.06}
    want = evaluate.predict_column(lp, v["ep"], col, v["noise_seq"], contract)["action"]
    assert np.allclose(act, want, atol=1e-6)
    # export.json
    on_disk = json.loads((out / "export.json").read_text(encoding="utf-8"))
    assert on_disk == meta
    assert meta["kind"] == "policy_output（記録ではない）" and meta["mode"] == mode and meta["m"] == 10
    assert meta["checkpoint"] == "000010" and meta["state_mode"] == {"kind": "lagged"}
    assert meta["choreography_sha256"] == synth.file_sha256(CHOREO_PATH)
    assert meta["tau_s"] == (0.06 if mode == "closed" else None)
    assert ("Gazebo の閉ループではない" in meta["note"]) == (mode == "closed")
    assert sorted(meta["files"]["yaml"]) == ["arm_25hz.yaml", "gripper_25hz.yaml"]
    # YAML：1 行・上限以下・最初の点は t=0
    for f in meta["files"]["yaml"]:
        text = (out / f).read_text(encoding="utf-8")
        assert text.count("\n") == 1 and len(text.encode()) <= export.YAML_BYTE_LIMIT
        assert "time_from_start: {sec: 0, nanosec: 0}" in text
    # 作りかけが残らない
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []


def test_write_export_refuses_existing_and_leaves_nothing_on_failure(tmp_path, ckpt, monkeypatch):
    out = tmp_path / "x"
    out.mkdir()
    with pytest.raises(export.ExportError, match="既に"):
        export.write_export(export.ExportSpec(checkpoint=ckpt, out_dir=out, device="cpu"))
    monkeypatch.setattr(export, "YAML_BYTE_LIMIT", 100)
    out2 = tmp_path / "y"
    with pytest.raises(export.ExportError, match="上限"):
        export.write_export(export.ExportSpec(checkpoint=ckpt, out_dir=out2, device="cpu"))
    assert not out2.exists() and list(tmp_path.glob(".*")) == []


def test_write_export_refuses_nan_policy_output(tmp_path, ckpt, monkeypatch):
    real = evaluate.predict_column

    def nan_pred(*a, **k):
        r = real(*a, **k)
        r["action"] = np.array(r["action"], copy=True)
        r["action"][3, 1] = np.nan
        return r

    monkeypatch.setattr(evaluate, "predict_column", nan_pred)
    with pytest.raises(export.ExportError, match="NaN"):
        export.write_export(export.ExportSpec(checkpoint=ckpt, out_dir=tmp_path / "z", device="cpu"))
    assert not (tmp_path / "z").exists()


def test_write_export_records_constant_state_mode(tmp_path, choreo):
    sm = state_mode.constant_from_choreography(choreo)
    root, _ = make_dataset(tmp_path, sm, name="c")
    ck = make_checkpoint(tmp_path, root, sm, chunk=10)
    meta = export.write_export(export.ExportSpec(checkpoint=ck, out_dir=tmp_path / "e", device="cpu", formats=("npz",)))
    assert meta["state_mode"] == sm and meta["files"] == {"npz": ["trajectory.npz"]}
    assert _run_check_contract(tmp_path / "e" / "trajectory.npz") == (0, "NPZ: PASS")
