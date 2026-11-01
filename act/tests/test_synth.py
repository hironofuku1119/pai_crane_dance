import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from crane_dance_act import synth

ACT_DIR = pathlib.Path(__file__).resolve().parents[1]
CONTRACT_PATH = ACT_DIR.parent / "contract" / "contract.json"
CHOREO_PATH = ACT_DIR.parent / "contract" / "choreography.json"


@pytest.fixture(scope="module")
def contract():
    return synth.load_contract(CONTRACT_PATH)


@pytest.fixture(scope="module")
def choreo():
    return synth.load_choreography(CHOREO_PATH)


@pytest.fixture(scope="module")
def ep(contract, choreo):
    return synth.build_episode(contract, choreo, synth.file_sha256(CHOREO_PATH))


def test_shape_dtype_and_n(ep, contract, choreo):
    assert synth.check_episode(ep, contract) == []
    fps = contract["recording"]["fps"]
    n = math.floor(choreo["song"]["duration_s"] * fps + 1e-9)
    assert len(ep["timestamp"]) == n
    assert n == 1560


def test_target_at(choreo):
    kfs = choreo["keyframes"]
    for k in kfs:
        assert np.allclose(synth.target_at(k["time_s"], kfs), k["joints_rad"], atol=1e-6, rtol=0)
    for a, b in zip(kfs, kfs[1:]):
        mid = synth.target_at((a["time_s"] + b["time_s"]) / 2, kfs)
        avg = (np.asarray(a["joints_rad"]) + np.asarray(b["joints_rad"])) / 2
        assert np.allclose(mid, avg, atol=1e-6, rtol=0)
    assert np.allclose(synth.target_at(kfs[0]["time_s"] - 1.0, kfs), kfs[0]["joints_rad"], atol=1e-6, rtol=0)
    assert np.allclose(synth.target_at(kfs[-1]["time_s"] + 1.0, kfs), kfs[-1]["joints_rad"], atol=1e-6, rtol=0)


def test_within_limits_and_joint1_zero(ep, contract):
    lim = np.array([contract["joint_limits_rad"][n] for n in contract["joint_names"]])
    for key in ("action", "observation_state"):
        a = ep[key]
        assert np.all(a >= lim[:, 0]) and np.all(a <= lim[:, 1]), key
    assert np.all(ep["action"][:, 0] == 0)
    assert np.all(ep["observation_state"][:, 0] == 0)


def test_state_is_previous_action(ep):
    assert np.array_equal(ep["observation_state"][1:], ep["action"][:-1])
    assert np.array_equal(ep["observation_state"][0], ep["action"][0])


def test_golden_rows(contract):
    g = contract["golden"]
    tol = contract["tolerance"]["env_abs"]
    assert len(g["rows"]) >= 1
    for r in g["rows"]:
        tab = g["tables"][r["table"]]
        ph = synth.phase(tab, r["tau"])
        assert abs(math.sin(ph) - r["sin"]) <= tol, r
        assert abs(math.cos(ph) - r["cos"]) <= tol, r
        assert abs(synth.progress(r["tau"], g["duration_s"]) - r["progress"]) <= tol, r


def test_env_and_timestamp_recomputed(ep, contract):
    tol = contract["tolerance"]
    fps = int(ep["fps"])
    beats = [float(b) for b in ep["beat_times_s"]]
    dur = float(ep["duration_s"])
    ts = ep["timestamp"]
    assert np.max(np.abs(ts - np.arange(len(ts)) / fps)) <= tol["time_abs_s"]
    want = np.array([[math.sin(synth.phase(beats, float(t))), math.cos(synth.phase(beats, float(t))),
                      synth.progress(float(t), dur)] for t in ts])
    assert np.max(np.abs(ep["observation_environment_state"].astype(np.float64) - want)) <= tol["env_abs"]


def test_meta(ep, contract):
    assert str(ep["contract_version"]) == contract["contract_version"]
    assert str(ep["choreography_sha256"]) == synth.file_sha256(CHOREO_PATH)


def test_deterministic_and_roundtrip(contract, choreo, tmp_path):
    sha = synth.file_sha256(CHOREO_PATH)
    a = synth.build_episode(contract, choreo, sha)
    b = synth.build_episode(contract, choreo, sha)
    for k in a:
        assert np.array_equal(a[k], b[k]), k
    p = tmp_path / "sub" / "ep.npz"
    synth.save_npz(p, a, contract)
    with np.load(p, allow_pickle=False) as z:
        assert set(z.files) == set(a)
        for k in a:
            assert np.array_equal(z[k], a[k]), k
            assert z[k].dtype == np.asarray(a[k]).dtype, k
        loaded = {k: z[k] for k in z.files}
    assert synth.check_episode(loaded, contract) == []


def test_cli(tmp_path):
    out = tmp_path / "ep.npz"
    r = subprocess.run([sys.executable, "scripts/make_synth_npz.py", "--out", str(out)],
                       cwd=ACT_DIR, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.is_file()


# --- 観測のばらつき ---
def _varied(contract, choreo, k, seed, tmp_path, name):
    d = tmp_path / name
    m = synth.write_varied_set(contract, choreo, synth.file_sha256(CHOREO_PATH), k, seed, d)
    return d, m


def _load(d, i):
    with np.load(d / f"ep{i:03d}.npz") as z:
        return {k: z[k] for k in z.files}


def test_varied_seed_deterministic_and_k_independent(contract, choreo, tmp_path):
    d1, _ = _varied(contract, choreo, 3, 7, tmp_path, "a")
    d2, _ = _varied(contract, choreo, 5, 7, tmp_path, "b")
    d3, _ = _varied(contract, choreo, 3, 7, tmp_path, "c")
    for i in range(3):
        e1, e2, e3 = _load(d1, i), _load(d2, i), _load(d3, i)
        for key in e1:
            assert np.array_equal(e1[key], e2[key]), (i, key)
            assert np.array_equal(e1[key], e3[key]), (i, key)


def test_varied_shape_limits_action_and_difference(contract, choreo, ep, tmp_path):
    d, m = _varied(contract, choreo, 3, 1, tmp_path, "v")
    lim = np.array([contract["joint_limits_rad"][n] for n in contract["joint_names"]])
    states = []
    for i in range(3):
        e = _load(d, i)
        assert sorted(e) == sorted(ep)
        assert synth.check_episode(e, contract) == []
        s = e["observation_state"]
        assert np.all(s >= lim[:, 0]) and np.all(s <= lim[:, 1])
        assert np.array_equal(e["action"], ep["action"])
        assert np.all(s[:, 0] == 0)
        states.append(s)
    assert not np.array_equal(states[0], states[1])
    assert not np.array_equal(states[1], states[2])
    assert not np.array_equal(states[0], ep["observation_state"])
    taus = [it["tau_s"] for it in m["items"]]
    assert all(synth.TAU_RANGE_S[0] <= t <= synth.TAU_RANGE_S[1] for t in taus)
    assert len({round(t, 9) for t in taus}) == 3


def test_noise_function_bounds():
    rng = np.random.default_rng(0)
    n = synth.observation_noise(rng, (20000, 5), synth.NOISE_SIGMA_RAD, synth.NOISE_CLIP_RAD)
    assert np.all(np.abs(n) <= synth.NOISE_CLIP_RAD)
    assert np.all(n[:, 0] == 0)
    assert np.all(np.abs(n[:, 1:]).max(axis=0) == synth.NOISE_CLIP_RAD)  # 切り詰めが実際に効く
    assert abs(n[:, 1:].std() - synth.NOISE_SIGMA_RAD) < 0.002


def test_lagged_state_first_frame_and_direction():
    a = np.zeros((5, 5))
    a[1:] = 1.0
    s = synth.lagged_state(a, 50, 0.05)
    assert np.all(s[0] == a[0])
    assert np.all(s[1] == 0)  # k=1 は a[0] までにしか追従しない
    assert np.all(s[2] > 0) and np.all(s[2] < 1)


def test_manifest_matches_json(contract, choreo, tmp_path):
    import json

    d, m = _varied(contract, choreo, 2, 3, tmp_path, "m")
    disk = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert disk == m
    assert disk["choreography_sha256"] == synth.file_sha256(CHOREO_PATH)
    assert disk["contract_version"] == contract["contract_version"]
    assert [it["spawn_index"] for it in disk["items"]] == [0, 1]
    for i in range(2):
        assert str(_load(d, i)["choreography_sha256"]) == disk["choreography_sha256"]


def test_out_dir_exists_stops(contract, choreo, tmp_path):
    d, _ = _varied(contract, choreo, 1, 0, tmp_path, "x")
    with pytest.raises(FileExistsError):
        synth.write_varied_set(contract, choreo, synth.file_sha256(CHOREO_PATH), 1, 0, d)
