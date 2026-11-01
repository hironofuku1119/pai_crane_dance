import json
import math
import pathlib
from bisect import bisect_right

import numpy as np
import pytest

from crane_dance_act import evaluate, state_mode, synth, to_lerobot, variants

ACT_DIR = pathlib.Path(__file__).resolve().parents[1]
CONTRACT_PATH = ACT_DIR.parent / "contract" / "contract.json"
CHOREO_PATH = ACT_DIR.parent / "contract" / "choreography.json"
REPO_ID = "local/crane_dance_eval_test"
FPS = 50


@pytest.fixture(scope="module")
def contract():
    return synth.load_contract(CONTRACT_PATH)


@pytest.fixture(scope="module")
def choreo():
    return synth.load_choreography(CHOREO_PATH)


@pytest.fixture(scope="module")
def sha():
    return synth.file_sha256(CHOREO_PATH)


def _grid_action64(choreo, fps=FPS):
    n = int(math.floor(float(choreo["song"]["duration_s"]) * fps + 1e-9))
    return np.stack([synth.target_at(k / fps, choreo["keyframes"]) for k in range(n)])


# ---------------------------------------------------------------- 検証データ
def test_validation_episodes_match_write_varied_set(tmp_path, contract, choreo, sha):
    d = tmp_path / "set"
    synth.write_varied_set(contract, choreo, sha, 22, 0, d)
    vals = evaluate.validation_episodes(contract, choreo, sha, 0, 18, 4)
    assert [v["index"] for v in vals] == [18, 19, 20, 21]
    for v in vals:  # 生成手順が同じなので、番号が同じなら 22 本の組と完全一致
        with np.load(d / f"ep{v['index']:03d}.npz") as z:
            for k in z.files:
                assert np.array_equal(z[k], np.asarray(v["ep"][k])), (v["index"], k)
    # i >= 20 は学習の本（0〜19）のどれとも違う
    train = []
    for i in range(20):
        with np.load(d / f"ep{i:03d}.npz") as z:
            train.append(z["observation_state"])
    for v in vals[2:]:
        assert all(not np.array_equal(v["ep"]["observation_state"], t) for t in train)
    # ノイズ用の乱数は決まった列が繰り返し得られる
    a = np.random.default_rng(vals[0]["noise_seq"]).normal(size=3)
    b = np.random.default_rng(vals[0]["noise_seq"]).normal(size=3)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------- ChunkBuffer
def _fake_predict(k, length):
    return np.tile(np.arange(k, k + length, dtype=np.float64)[:, None], (1, 5))


def test_chunk_buffer_call_counts():
    n = 100
    for m, want in ((1, 100), (10, 10)):
        buf = evaluate.ChunkBuffer(m)
        calls = []
        for k in range(n):
            a = buf.get(k, lambda k=k: (calls.append(k), _fake_predict(k, 15))[1])
            assert a[0] == k
        assert len(calls) == want == buf.calls
        if m == 10:
            assert calls == list(range(0, 100, 10))


def test_chunk_buffer_skip_and_reset():
    buf = evaluate.ChunkBuffer(10)
    calls = []

    def get(k):
        return buf.get(k, lambda: (calls.append(k), _fake_predict(k, 15))[1])

    assert get(0)[0] == 0
    assert get(3)[0] == 3  # 保持範囲の中へ飛んだ → 呼ばない
    assert calls == [0]
    assert get(25)[0] == 25  # 範囲外へ飛んだ → その時点で呼び直す
    assert calls == [0, 25]
    assert get(28)[0] == 28
    assert get(30)[0] == 30  # m の倍数 → 呼び直す
    assert calls == [0, 25, 30]
    assert get(12)[0] == 12  # 戻った → 呼び直す
    assert calls[-1] == 12
    buf.reset()
    assert buf.is_empty()
    assert get(7)[0] == 7
    assert calls[-1] == 7
    with pytest.raises(ValueError):
        evaluate.ChunkBuffer(10).get(0, lambda: _fake_predict(0, 5))  # 塊が m より短い


# ---------------------------------------------------------------- 閉ループの一歩
def test_closed_loop_matches_lagged_state_and_noise(contract, choreo, sha):
    var = {"tau_s": 0.05, "noise_sigma_rad": synth.NOISE_SIGMA_RAD, "noise_clip_rad": synth.NOISE_CLIP_RAD}
    ep = synth.build_episode(contract, choreo, sha, variation=var, rng=np.random.default_rng(1))
    ref = ep["action"].astype(np.float64)
    n = len(ref)
    obs_seen = []
    counter = {"k": 0}

    def fake(obs, env):
        k = counter["k"]
        counter["k"] += 1
        obs_seen.append(np.array(obs))
        out = np.empty((10, 5))
        for i in range(10):
            out[i] = ref[min(k + i, n - 1)]
        return out

    tau = 0.06
    run = evaluate.closed_loop(fake, ep, 1, tau, np.random.default_rng(5), contract)
    assert counter["k"] == n
    assert np.allclose(run["action"], ref, atol=0, rtol=0)
    assert np.allclose(run["state"], synth.lagged_state(ref, FPS, tau), atol=1e-12, rtol=0)
    diff = np.stack(obs_seen) - run["state"]
    assert np.max(np.abs(diff)) <= synth.NOISE_CLIP_RAD + 1e-12
    assert np.max(np.abs(diff)) > 0.0  # ノイズは実際に足されている
    assert np.all(diff[:, 0] == 0.0)  # joint1 には足さない
    # 同じ乱数なら同じ結果
    counter["k"] = 0
    run2 = evaluate.closed_loop(fake, ep, 1, tau, np.random.default_rng(5), contract)
    assert np.array_equal(run["state"], run2["state"])


# ---------------------------------------------------------------- 指標
def test_keyframe_errors_zero_for_reference(choreo):
    # お手本の action そのものなら、振付の時刻が格子に乗らなくても誤差 0（格子点のお手本と比べるため）
    ref = _grid_action64(choreo)
    kf = evaluate.keyframe_errors(ref, ref, choreo, FPS)
    assert kf["max_judge"] == 0.0 and kf["max_other"] == 0.0
    assert kf["per_keyframe"].shape == (len(choreo["keyframes"]), 5)
    # float32 に落とした action でも丸め以内
    kf32 = evaluate.keyframe_errors(ref.astype(np.float32).astype(np.float64), ref, choreo, FPS)
    assert kf32["max_judge"] <= 1e-6
    with pytest.raises(ValueError):
        evaluate.keyframe_errors(ref[:-1], ref, choreo, FPS)


def test_keyframe_errors_detects_offset(choreo):
    ref = _grid_action64(choreo)
    pred = ref.copy()
    pred[:, 2] += 0.05
    kf = evaluate.keyframe_errors(pred, ref, choreo, FPS)
    assert kf["max_judge"] == pytest.approx(0.05, abs=1e-9)
    assert kf["max_other"] <= 1e-9
    beats = choreo["song"]["beat_times_s"]
    assert evaluate.judge(kf, evaluate.timing_offsets(ref, choreo, beats, FPS))["pass_a"] is False


def _interval(beats, t):
    i = min(max(bisect_right(beats, t) - 1, 0), len(beats) - 2)
    return beats[i + 1] - beats[i]


def test_timing_offsets_delay_and_zero(choreo):
    beats = [float(b) for b in choreo["song"]["beat_times_s"]]
    a = _grid_action64(choreo)
    zero = evaluate.timing_offsets(a, choreo, beats, FPS)
    assert len(zero) >= 4
    for seg in zero:
        for level, r in seg["levels"].items():
            assert r["reached"]
            assert abs(r["dt_beats"]) <= 1e-6, (seg["segment"], level)
    d = 3
    delayed = np.concatenate([np.repeat(a[:1], d, axis=0), a[:-d]])
    offs = evaluate.timing_offsets(delayed, choreo, beats, FPS)
    assert len(offs) == len(zero)
    for seg in offs:
        r = seg["levels"][0.5]
        want = d / FPS / _interval(beats, r["t_ref_s"])
        assert r["reached"]
        assert r["dt_beats"] == pytest.approx(want, abs=1e-6)
        assert r["dt_s"] == pytest.approx(d / FPS, abs=1e-9)
    j = evaluate.judge(evaluate.keyframe_errors(delayed, a, choreo, FPS), offs)
    assert j["unreached_at_0.5"] == 0 and j["segments"] == len(offs)


def test_timing_offsets_excludes_small_and_marks_unreached():
    # 振幅 0.05 rad の区間は除外・動かない pred は「未到達」
    ch = {
        "keyframes": [
            {"time_s": 0.0, "joints_rad": [0, 0, 0, 0, 0]},
            {"time_s": 1.0, "joints_rad": [0, 0, 0.05, 0, 0]},
            {"time_s": 2.0, "joints_rad": [0, 0, 0.05, 0, 0.5]},
        ]
    }
    beats = [i * 0.5 for i in range(8)]
    pred = np.zeros((int(3 * FPS), 5))
    offs = evaluate.timing_offsets(pred, ch, beats, FPS)
    assert [s["segment"] for s in offs] == [1]
    assert offs[0]["joint"] == 4
    for r in offs[0]["levels"].values():
        assert r["reached"] is False and math.isnan(r["dt_beats"])
    j = evaluate.judge({"max_judge": 0.0, "max_other": 0.0}, offs)
    assert j["unreached_at_0.5"] == 1 and j["pass_b"] is False and j["pass"] is False


# ---------------------------------------------------------------- 感度
def test_sensitivity_with_fake_policies(contract, choreo, sha):
    ep = synth.build_episode(contract, choreo, sha)
    ks = [100, 400, 800]
    const = lambda s, e: np.zeros((10, 5))  # noqa: E731
    res = evaluate.sensitivity(const, ep, ks, contract)
    for name in ("phase_plus", "phase_minus", "progress_plus", "progress_minus"):
        assert res[name]["mean_abs_rad"] == 0.0
    # progress だけに反応する方策: 差は progress の変化量に比例
    prog = lambda s, e: np.full((10, 5), float(e[2]))  # noqa: E731
    res = evaluate.sensitivity(prog, ep, ks, contract)
    beats = [float(b) for b in ep["beat_times_s"]]
    want = np.mean([evaluate.beat_interval(beats, k / FPS) / 16 / float(ep["duration_s"]) for k in ks])
    assert res["progress_plus"]["mean_abs_rad"] == pytest.approx(float(want), rel=1e-6)
    assert res["phase_plus"]["mean_abs_rad"] == pytest.approx(0.0, abs=1e-6)
    # 位相だけに反応する方策
    phase = lambda s, e: np.full((10, 5), math.atan2(float(e[0]), float(e[1])))  # noqa: E731
    res = evaluate.sensitivity(phase, ep, [100], contract)
    assert res["phase_minus"]["mean_abs_rad"] == pytest.approx(2 * math.pi / 16, rel=1e-6)
    assert res["progress_plus"]["mean_abs_rad"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- 変種
def test_apply_env_variant(contract, choreo, sha):
    ep = synth.build_episode(contract, choreo, sha)
    env = ep["observation_environment_state"]
    before = env.copy()
    full = variants.apply_env_variant(env, "full")
    assert np.array_equal(full, env) and full is not env
    nph = variants.apply_env_variant(env, "no_phase")
    assert np.all(nph[:, 0] == 0.0) and np.all(nph[:, 1] == 1.0) and np.array_equal(nph[:, 2], env[:, 2])
    npr = variants.apply_env_variant(env, "no_progress")
    assert np.all(npr[:, 2] == 0.0) and np.array_equal(npr[:, :2], env[:, :2])
    for v in (full, nph, npr):
        assert v.dtype == env.dtype and v.shape == env.shape
    assert np.array_equal(env, before)  # 元は変えない
    with pytest.raises(ValueError):
        variants.apply_env_variant(env, "nope")


@pytest.fixture(scope="module")
def small_set(tmp_path_factory, contract, choreo, sha):
    d = tmp_path_factory.mktemp("evs") / "synth"
    synth.write_varied_set(contract, choreo, sha, 2, 0, d)
    return d


def test_make_variant_set_and_convert(tmp_path, contract, small_set):
    out = tmp_path / "v"
    m = variants.make_variant_set(small_set, "no_phase", out, contract)
    assert m["env_variant"] == "no_phase"
    saved = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    orig = json.loads((small_set / "manifest.json").read_text(encoding="utf-8"))
    assert saved["env_variant"] == "no_phase"
    assert {k: v for k, v in saved.items() if k != "env_variant"} == orig
    for p in sorted(small_set.glob("ep*.npz")):
        with np.load(p) as a, np.load(out / p.name) as b:
            assert sorted(a.files) == sorted(b.files)
            for k in a.files:
                if k == "observation_environment_state":
                    assert np.array_equal(b[k], variants.apply_env_variant(a[k], "no_phase"))
                else:
                    assert np.array_equal(a[k], b[k]), k
    with pytest.raises(FileExistsError):
        variants.make_variant_set(small_set, "no_phase", out, contract)
    # 変種の npz も取り込める
    ds = to_lerobot.convert(sorted(out.glob("ep*.npz")), contract, CHOREO_PATH, tmp_path / "lr" / "ds", REPO_ID, out / "manifest.json")
    assert (ds / "manifest.json").is_file()


# ---------------------------------------------------------------- 小さい実物の通し
def test_real_small_policy_roundtrip(tmp_path, contract, small_set):
    import torch
    from lerobot.configs import FeatureType, NormalizationMode
    from lerobot.datasets.io_utils import load_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.feature_utils import dataset_to_policy_features

    root = to_lerobot.convert(sorted(small_set.glob("ep*.npz")), contract, CHOREO_PATH, tmp_path / "lr" / "ds", REPO_ID, small_set / "manifest.json")
    stats = load_stats(root)
    probe = LeRobotDataset(REPO_ID, root=root)
    feats = dataset_to_policy_features(probe.meta.features)
    output_features = {k: f for k, f in feats.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in feats.items() if k not in output_features}
    chunk = 10
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        chunk_size=chunk,
        n_action_steps=chunk,
        pretrained_backbone_weights=None,
        normalization_mapping={"STATE": NormalizationMode.MEAN_STD, "ACTION": NormalizationMode.MEAN_STD},
    )
    torch.manual_seed(0)
    policy = ACTPolicy(cfg)
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    ck = tmp_path / "ck" / "pretrained_model"
    policy.save_pretrained(ck)
    pre.save_pretrained(ck)
    post.save_pretrained(ck)
    state_mode.write_atomic(state_mode.checkpoint_file(ck), state_mode.lagged())  # state_mode.json が無いと読み込みが止まる

    lp = evaluate.load_policy(ck, "cpu")
    s = np.array([0.0, 0.1, -0.5, 0.0, 0.1])
    e = np.array([0.0, 1.0, 0.3])
    out = lp(s, e)
    assert out.shape == (chunk, 5) and np.all(np.isfinite(out))
    # チェックポイント側の stats で逆正規化される（保存前の部品と同じ出力）
    direct = post(policy.predict_action_chunk(pre({
        "observation.state": torch.as_tensor(s.astype(np.float32)),
        "observation.environment_state": torch.as_tensor(e.astype(np.float32)),
    })))[0].numpy()
    assert np.allclose(out, direct, atol=1e-5)

    ep = synth.build_episode(contract, synth.load_choreography(CHOREO_PATH), synth.file_sha256(CHOREO_PATH),
                             variation={"tau_s": 0.05, "noise_sigma_rad": synth.NOISE_SIGMA_RAD, "noise_clip_rad": synth.NOISE_CLIP_RAD},
                             rng=np.random.default_rng(0))
    n = 200
    sub = {k: (v[:n] if k in synth.DATA_KEYS else v) for k, v in ep.items()}
    for m in (1, 10):
        ol = evaluate.open_loop(lp, sub, m)
        assert ol.shape == (n, 5) and np.all(np.isfinite(ol))
        cl = evaluate.closed_loop(lp, sub, m, 0.06, np.random.default_rng(0), contract)
        assert cl["action"].shape == (n, 5) and cl["state"].shape == (n, 5)
        assert np.all(np.isfinite(cl["action"])) and np.all(np.isfinite(cl["state"]))
    # m=10 は 10 フレームごと（塊の先頭 10 本）なので、m=1 と先頭の行動が一致する
    ol1 = evaluate.open_loop(lp, sub, 1)
    ol10 = evaluate.open_loop(lp, sub, 10)
    assert np.allclose(ol1[::10], ol10[::10], atol=1e-5)
