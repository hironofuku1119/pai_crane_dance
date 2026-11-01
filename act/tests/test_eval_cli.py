"""scripts/eval_act.py が evaluate.evaluate_checkpoint を呼んでも、出力の形が今までどおりであること。"""
import json
import os
import subprocess
import sys

import pytest
from helpers_ckpt import ACT_DIR, make_checkpoint, make_dataset

from crane_dance_act import state_mode


def run_eval(ck, out, *extra):
    return subprocess.run(
        [sys.executable, str(ACT_DIR / "scripts" / "eval_act.py"), "--checkpoint", str(ck), "--device", "cpu", "--count", "1",
         "--m", "10", "--taus", "0.06", "--out", str(out), *extra],
        capture_output=True, text=True, env={**os.environ, "HF_HUB_OFFLINE": "1"}, timeout=600,
    )  # fmt: skip


@pytest.fixture(scope="module")
def lagged_ds(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("evcli"))[0]


def test_eval_act_cli_output_shape_is_unchanged(tmp_path, lagged_ds):
    ck = make_checkpoint(tmp_path, lagged_ds, state_mode.lagged())
    out = tmp_path / "ev.json"
    r = run_eval(ck, out)
    assert r.returncode == 0, r.stdout + r.stderr
    res = json.loads(out.read_text(encoding="utf-8"))
    assert list(res["open_loop"]) == ["m10"] and list(res["closed_loop"]) == ["m10_tau0.06"]
    ep = res["open_loop"]["m10"]["episodes"][0]
    assert {"index", "keyframe", "timing", "judge"} <= set(ep) and ep["index"] == 20
    assert res["state_mode"] == {"kind": "lagged"} and res["m_list"] == [10] and res["tau_list_s"] == [0.06]
    assert "open_loop m10:" in r.stdout and "closed_loop m10_tau0.06:" in r.stdout
    again = run_eval(ck, out)
    assert again.returncode == 1  # 出力先が既にある


def test_eval_act_cli_stops_on_missing_state_mode_unless_legacy_flag(tmp_path, lagged_ds):
    ck = make_checkpoint(tmp_path, lagged_ds, None)
    r = run_eval(ck, tmp_path / "a.json")
    assert r.returncode == 1 and "state_mode.json" in r.stderr and not (tmp_path / "a.json").exists()
    r = run_eval(ck, tmp_path / "b.json", "--assume-legacy-lagged")
    assert r.returncode == 0, r.stdout + r.stderr
