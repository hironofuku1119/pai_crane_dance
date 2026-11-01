"""state_mode（定数の入れ忘れを止める）。"""
import json

import numpy as np
import pytest
from helpers_ckpt import CHOREO_PATH, CONTRACT_PATH, make_checkpoint, make_dataset

from crane_dance_act import evaluate, state_mode, synth, train

ENV = np.array([0.0, 1.0, 0.3])


@pytest.fixture(scope="module")
def choreo():
    return synth.load_choreography(CHOREO_PATH)


@pytest.fixture(scope="module")
def const_sm(choreo):
    return state_mode.constant_from_choreography(choreo)


@pytest.fixture(scope="module")
def const_ds(tmp_path_factory, const_sm):
    return make_dataset(tmp_path_factory.mktemp("sm-const"), const_sm, name="c")[0]


@pytest.fixture(scope="module")
def lagged_ds(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("sm-lag"), None, name="l")[0]


# ---------------------------------------------------------------- 定義
def test_constant_comes_from_keyframes0(choreo, const_sm):
    assert const_sm["kind"] == "constant" and const_sm["source"] == "keyframes[0]"
    assert const_sm["values"] == [float(v) for v in choreo["keyframes"][0]["joints_rad"]]
    assert state_mode.validate(const_sm) == const_sm


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "nope"},
        {"kind": "lagged", "values": [0] * 5},
        {"kind": "constant", "values": [0.0] * 4, "source": "x"},
        {"kind": "constant", "values": [0.0, 0.0, 0.0, 0.0, float("nan")], "source": "x"},
        {"kind": "constant", "values": [0.0] * 5},
        {"kind": "constant", "values": [0.0] * 5, "source": "x", "extra": 1},
        "constant",
    ],
)
def test_validate_rejects(bad):
    with pytest.raises(state_mode.StateModeError):
        state_mode.validate(bad)


def test_dataset_manifest_carries_state_mode(const_ds, lagged_ds, const_sm):
    assert state_mode.read_manifest_state_mode(const_ds) == const_sm
    assert state_mode.read_manifest_state_mode(lagged_ds) == {"kind": "lagged"}
    # 定数のデータは全フレームが同じ値
    from lerobot.datasets.io_utils import load_stats

    st = load_stats(const_ds)["observation.state"]
    assert np.allclose(st["mean"], const_sm["values"], atol=1e-5)
    assert np.all(np.isclose(st["std"], train.STD_FLOOR_RAD))  # 実際の std は 0 → 下限


def test_manifest_without_state_mode_is_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"seed": 0}), encoding="utf-8")
    with pytest.raises(state_mode.StateModeError):
        state_mode.read_manifest_state_mode(tmp_path)


def test_preflight_state_mode_checks(tmp_path, const_ds, lagged_ds, const_sm):
    contract = synth.load_contract(CONTRACT_PATH)
    assert train.preflight(const_ds, contract, CHOREO_PATH, "cpu") == []
    assert train.preflight(lagged_ds, contract, CHOREO_PATH, "cpu") == []
    # manifest が constant と言うのに stats は lagged → 止まる
    import shutil

    bad = tmp_path / "bad"
    shutil.copytree(lagged_ds, bad)
    m = json.loads((bad / "manifest.json").read_text(encoding="utf-8"))
    m["state_mode"] = const_sm
    (bad / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    errs = train.preflight(bad, contract, CHOREO_PATH, "cpu")
    assert any("state_mode" in e and "std" in e for e in errs), errs
    # manifest が無印（旧形式）→ 止まる
    old = tmp_path / "old"
    shutil.copytree(lagged_ds, old)
    m = json.loads((old / "manifest.json").read_text(encoding="utf-8"))
    del m["state_mode"]
    (old / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    assert any("state_mode" in e for e in train.preflight(old, contract, CHOREO_PATH, "cpu"))


# ---------------------------------------------------------------- 読み込み
def test_constant_model_ignores_given_state(tmp_path, const_ds, const_sm):
    ck = make_checkpoint(tmp_path, const_ds, const_sm)
    lp = evaluate.load_policy(ck, "cpu")
    assert lp.state_mode == const_sm
    a = lp(np.full(5, -64.0), ENV)  # 黙って通っていた「−64 前後の入力」も定数に置き換わる
    b = lp(np.array([0.1, 0.2, -0.5, 0.0, 0.3]), ENV)
    c = lp(np.asarray(const_sm["values"]), ENV)
    assert np.array_equal(a, b) and np.allclose(a, c, atol=1e-6)


def test_constant_model_rejects_raw_state_by_default(tmp_path, const_ds, const_sm):
    ck = make_checkpoint(tmp_path, const_ds, const_sm)
    lp = evaluate.load_policy(ck, "cpu")
    with pytest.raises(state_mode.StateModeError):
        lp(np.zeros(5), ENV, raw_state=True)
    # 明示の引数でだけ許す。そのとき本物の state が効く（定数のときと出力が違う）
    lp2 = evaluate.load_policy(ck, "cpu", allow_raw_state=True)
    raw = lp2(np.array([0.5, 0.5, -0.5, 0.5, 0.3]), ENV, raw_state=True)
    assert not np.allclose(raw, lp2(np.zeros(5), ENV), atol=1e-4)


def test_lagged_model_unchanged(tmp_path, lagged_ds):
    ck = make_checkpoint(tmp_path, lagged_ds, state_mode.lagged())
    lp = evaluate.load_policy(ck, "cpu")
    assert lp.state_mode == {"kind": "lagged"}
    s1, s2 = np.array([0.0, 0.1, -0.5, 0.0, 0.1]), np.array([0.0, 0.2, -0.9, 0.1, 0.0])
    assert not np.allclose(lp(s1, ENV), lp(s2, ENV), atol=1e-5)  # state が効く
    assert np.array_equal(lp(s1, ENV), lp(s1, ENV, raw_state=True))  # raw_state は lagged では何も変えない


def test_missing_state_mode_stops(tmp_path, lagged_ds, const_ds, const_sm):
    ck = make_checkpoint(tmp_path, lagged_ds, None)
    with pytest.raises(state_mode.StateModeError, match="state_mode.json"):
        evaluate.load_policy(ck, "cpu")
    # 旧形式を lagged として読むと明示したときだけ通る
    assert evaluate.load_policy(ck, "cpu", assume_legacy_lagged=True).state_mode == {"kind": "lagged"}
    # それでも stats が定数で学習したものなら止まる（lagged と言い張れない）
    ck2 = make_checkpoint(tmp_path, const_ds, None, name="run2")
    with pytest.raises(state_mode.StateModeError, match="全部下限"):
        evaluate.load_policy(ck2, "cpu", assume_legacy_lagged=True)


def test_mismatch_with_stats_stops(tmp_path, lagged_ds, const_ds, const_sm):
    # 定数と言うが stats は lagged
    ck = make_checkpoint(tmp_path, lagged_ds, const_sm)
    with pytest.raises(state_mode.StateModeError, match="constant"):
        evaluate.load_policy(ck, "cpu")
    # lagged と言うが stats は定数
    ck = make_checkpoint(tmp_path, const_ds, state_mode.lagged(), name="run2")
    with pytest.raises(state_mode.StateModeError, match="lagged"):
        evaluate.load_policy(ck, "cpu")
    # 定数の値が stats の mean と違う
    other = dict(const_sm, values=[v + 0.1 for v in const_sm["values"]])
    ck = make_checkpoint(tmp_path, const_ds, other, name="run3")
    with pytest.raises(state_mode.StateModeError, match="mean"):
        evaluate.load_policy(ck, "cpu")


def test_broken_or_conflicting_state_mode_json_stops(tmp_path, const_ds, const_sm):
    ck = make_checkpoint(tmp_path, const_ds, const_sm)
    f = state_mode.checkpoint_file(ck)
    f.write_text(json.dumps({"kind": "constant", "values": [1, 2], "source": "x"}), encoding="utf-8")
    with pytest.raises(state_mode.StateModeError):
        evaluate.load_policy(ck, "cpu")
    # run.json の state_mode と食い違う
    f.write_text(json.dumps(const_sm), encoding="utf-8")
    (tmp_path / "run" / "run.json").write_text(json.dumps({"state_mode": {"kind": "lagged"}}), encoding="utf-8")
    with pytest.raises(state_mode.StateModeError, match="run.json"):
        evaluate.load_policy(ck, "cpu")
    (tmp_path / "run" / "run.json").write_text(json.dumps({"state_mode": const_sm}), encoding="utf-8")
    assert evaluate.load_policy(ck, "cpu").state_mode == const_sm


def test_post_check_state_mode_file(tmp_path, const_ds, const_sm):
    ck = make_checkpoint(tmp_path, const_ds, const_sm)
    exp = dict(train.resolve_policy({"chunk_size": 10}), state_mode=const_sm)
    # 小さい方策なので lr などの保存物は無い → state_mode 以外の指摘だけ除いて見る
    errs = [e for e in train.post_check(tmp_path / "run", exp) if "state_mode" in e]
    assert errs == []
    state_mode.checkpoint_file(ck).write_text(json.dumps(state_mode.lagged()), encoding="utf-8")
    assert any("state_mode.json が学習時の" in e for e in train.post_check(tmp_path / "run", exp))
    state_mode.checkpoint_file(ck).unlink()
    assert any("state_mode.json が無い" in e for e in train.post_check(tmp_path / "run", exp))
