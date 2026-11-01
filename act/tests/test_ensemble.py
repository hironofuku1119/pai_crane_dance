"""閉ループ m=1 + temporal ensemble（lerobot の ACTTemporalEnsembler）の golden。"""
import math
import pathlib

import numpy as np
import pytest

from crane_dance_act import evaluate, synth

ACT_DIR = pathlib.Path(__file__).resolve().parents[1]
CONTRACT_PATH = ACT_DIR.parent / "contract" / "contract.json"
CHOREO_PATH = ACT_DIR.parent / "contract" / "choreography.json"
N, L = 40, 8


@pytest.fixture(scope="module")
def contract():
    return synth.load_contract(CONTRACT_PATH)


@pytest.fixture(scope="module")
def ep(contract):
    choreo = synth.load_choreography(CHOREO_PATH)
    full = synth.build_episode(contract, choreo, synth.file_sha256(CHOREO_PATH))
    return {k: (v[:N] if k in synth.DATA_KEYS else v) for k, v in full.items()}


def make_chunks(seed=3):
    return np.random.default_rng(seed).uniform(-0.5, 0.5, size=(N, L, 5))  # k 回目に呼んだときの塊


class Seq:
    """呼ばれた順に塊を返す偽の方策（閉ループは毎フレーム 1 回ずつ呼ぶ）。"""

    def __init__(self, chunks):
        self.chunks, self.i = chunks, 0

    def __call__(self, state, env):
        c = self.chunks[self.i]
        self.i += 1
        return c


def offline_ensemble(chunks, coeff):
    """定義どおりの重ね合わせ：a[k] = Σ w_i·pred_i / Σ w_i （w_i = exp(-coeff·i)・i=0 が最も古い塊）。"""
    out = np.empty((len(chunks), chunks.shape[2]))
    for k in range(len(chunks)):
        js = list(range(max(0, k - L + 1), k + 1))  # 古い順
        w = np.exp(-coeff * np.arange(len(js)))
        preds = np.stack([chunks[j][k - j] for j in js])
        out[k] = (w[:, None] * preds).sum(0) / w.sum()
    return out


@pytest.mark.parametrize("coeff", [0.01, 0.0, 0.3, -0.2])
def test_ensemble_matches_offline_definition(ep, contract, coeff):
    chunks = make_chunks()
    got = evaluate.ensemble_closed_loop(Seq(chunks), ep, 0.06, np.random.default_rng(0), contract, coeff=coeff)
    assert np.allclose(got["action"], offline_ensemble(chunks, coeff), atol=1e-5)  # ensembler は float32 の重み


def test_ensemble_golden_first_steps(ep, contract):
    # k=0 は最初の塊の先頭そのもの。k=1 は (古い塊の [1]·w0 + 新しい塊の [0]·w1) / (w0+w1)
    chunks = make_chunks()
    a = evaluate.ensemble_closed_loop(Seq(chunks), ep, 0.06, np.random.default_rng(0), contract, coeff=0.01)["action"]
    assert np.allclose(a[0], chunks[0][0], atol=1e-6)
    w = np.exp(-0.01 * np.arange(2))
    assert np.allclose(a[1], (w[0] * chunks[0][1] + w[1] * chunks[1][0]) / w.sum(), atol=1e-6)
    # 古い塊に重い向き（coeff > 0）：w0 > w1
    assert w[0] > w[1]


def test_ensemble_machine_is_first_order_lag(ep, contract):
    chunks = make_chunks()
    tau = 0.06
    r = evaluate.ensemble_closed_loop(Seq(chunks), ep, tau, np.random.default_rng(0), contract)
    a, s = r["action"], r["state"]
    alpha = 1.0 - math.exp(-(1.0 / 50) / tau)
    assert np.allclose(s[0], ep["action"][0])
    assert np.allclose(s[1:], s[:-1] + alpha * (a[:-1] - s[:-1]))


def test_closed_loop_state_matches_between_kinds(ep, contract):
    # 同じ仮の機械・同じノイズ：m=1 の closed_loop（重ね合わせなし）と ensemble は、塊が全フレーム同じなら（先頭を共有する定数列）一致する
    const = np.tile(np.linspace(-0.2, 0.2, 5), (L, 1))
    a1 = evaluate.closed_loop(lambda s, e: const, ep, 1, 0.06, np.random.default_rng(0), contract)["action"]
    a2 = evaluate.ensemble_closed_loop(lambda s, e: const, ep, 0.06, np.random.default_rng(0), contract)["action"]
    assert np.allclose(a1, a2, atol=1e-6)


def test_columns_validation_and_names():
    assert evaluate.column_name({"kind": "open", "m": 10}) == "open_m10"
    assert evaluate.column_name({"kind": "closed", "m": 10, "tau": 0.06}) == "closed_m10_tau0.06"
    assert evaluate.column_name({"kind": "ensemble", "m": 1, "tau": 0.06}) == "closed_m1_ens_tau0.06"
    ok = evaluate.check_columns([{"kind": "ensemble", "m": 1, "tau": 0.06}])
    assert ok[0]["coeff"] == evaluate.ENSEMBLE_COEFF
    for bad in ([], [{"kind": "open"}], [{"kind": "closed", "m": 1, "tau": -1}], [{"kind": "ensemble", "m": 2, "tau": 0.1}],
                [{"kind": "open", "m": 1, "x": 1}], [{"kind": "open", "m": True}]):  # fmt: skip
        with pytest.raises(ValueError):
            evaluate.check_columns(bad)


def test_predict_column_dispatch(ep, contract):
    chunks = make_chunks()
    r = evaluate.predict_column(Seq(chunks), ep, {"kind": "ensemble", "m": 1, "tau": 0.06, "coeff": 0.01}, np.random.SeedSequence(0), contract)
    assert r["action"].shape == (N, 5) and r["state"].shape == (N, 5)
    o = evaluate.predict_column(lambda s, e: np.zeros((L, 5)), ep, {"kind": "open", "m": 4}, np.random.SeedSequence(0), contract)
    assert np.array_equal(o["state"], ep["observation_state"].astype(np.float64)) and not o["action"].any()
