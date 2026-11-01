"""バッチ道具：CPU での極小の実走（合成 2 本・steps 2・評価 1 本・書き出し 1 つ）。tests/test_train.py の test_train_act_cpu_two_steps にならう。"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from helpers_batch import tiny_toml

from crane_dance_act import batch, state_mode, synth

ACT_DIR = batch.ACT_DIR
REPO_ROOT = batch.REPO_ROOT


@pytest.mark.skipif(os.environ.get("ACT_SKIP_SLOW") == "1", reason="ACT_SKIP_SLOW=1")
def test_batch_cpu_tiny_real_run(tmp_path):
    constant = tiny_toml(names=("t2",)).replace("episodes = 2", 'episodes = 2\nstate_mode = "constant"')
    cfgp = tmp_path / "tiny.toml"
    cfgp.write_text(tiny_toml(names=("t1",)) + constant, encoding="utf-8")
    root = tmp_path / "root"
    out = []
    args = ["--config", str(cfgp), "--device", "cpu", "--out-root", str(root)]

    def run():
        return batch.cli_main(args, echo=out.append, caffeinate=lambda log: None, preflight_kwargs={"env": {k: v for k, v in os.environ.items() if not k.startswith("ACCELERATE_")}})

    rc = run()
    assert rc == 0, "\n".join(out)

    for name, sm_kind in (("t1", "lagged"), ("t2", "constant")):
        rd = root / "runs" / name
        st = json.loads((rd / "status.json").read_text(encoding="utf-8"))
        assert all(st["stages"][s]["state"] == "done" for s in batch.STAGES), st
        assert st["last_attempt"]["label"] == "完了" and st["stages"]["train"]["detail"]["post_check_errors"] == []
        assert st["stages"]["train"]["detail"]["device_actual"] == "cpu"
        resolved = json.loads((rd / "resolved.json").read_text(encoding="utf-8"))
        assert len(resolved["sha256"]) == 64 and resolved["run"]["data"]["state_mode"] == sm_kind
        # 学習の保存物：run.json と state_mode.json（チェックポイントの横）
        run_json = json.loads((rd / "train" / "run.json").read_text(encoding="utf-8"))
        assert run_json["state_mode"]["kind"] == sm_kind and run_json["policy"]["chunk_size"] == 10
        ck = rd / "train" / "checkpoints" / "000002"
        assert json.loads((ck / "state_mode.json").read_text(encoding="utf-8")) == run_json["state_mode"]
        assert (ck / "training_state" / "optimizer_param_groups.json").is_file()
        # 評価：開ループと閉ループの列
        ev = json.loads((rd / "eval" / "000002.json").read_text(encoding="utf-8"))
        assert sorted(ev["columns"]) == ["closed_m10_tau0.06", "open_m10"] and ev["state_mode"]["kind"] == sm_kind
        # 書き出し：npz は契約の検査器を通る・export.json に「方策の出力」の印
        for mode in ("open", "closed"):
            d = rd / "export" / f"000002-{mode}"
            meta = json.loads((d / "export.json").read_text(encoding="utf-8"))
            assert meta["kind"] == "policy_output（記録ではない）" and meta["mode"] == mode and meta["state_mode"]["kind"] == sm_kind
            r = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "check_contract.py"), "--npz", str(d / "trajectory.npz")],
                               capture_output=True, text=True, cwd=REPO_ROOT)  # fmt: skip
            assert r.stdout.strip().splitlines()[-1] == "NPZ: PASS", r.stdout
            with np.load(d / "trajectory.npz", allow_pickle=False) as z:
                assert np.allclose(z["timestamp"], np.arange(len(z["timestamp"])) / 50, atol=1e-9)
                if sm_kind == "constant" and mode == "open":
                    pass
        # 子プロセスのログは伏せてから保存される
        assert (rd / "logs" / "train.child.log").is_file() and (rd / "logs" / "eval-000002.child.log").is_file()

    # データセットは設定のハッシュで名前が決まり、別の設定なので 2 つ
    assert len([p for p in (root / "datasets").iterdir() if p.name.startswith("ds-")]) == 2
    ds = next(p for p in (root / "datasets").iterdir() if p.name.startswith("ds-"))
    m = json.loads((ds / "manifest.json").read_text(encoding="utf-8"))
    assert state_mode.validate(m["state_mode"]) and len(m["data_config_sha256"]) == 64
    assert m["choreography_sha256"] == synth.file_sha256(batch.DEFAULT_CHOREOGRAPHY)

    # 実パス・利用者名・秘密が、この道具の保存物（ログ・status・run.json・集計表・書き出し）に残らない。
    # lerobot 自身が checkpoints/ の中に書く train_config.json は lerobot の成果物で、実パスを含む（この道具では伏せない）ので対象外
    home = os.path.expanduser("~")
    user = os.path.basename(home)
    texts = []
    for p in root.rglob("*"):
        if "checkpoints" in p.relative_to(root).parts:
            continue
        if p.is_file() and p.suffix in (".json", ".md", ".log", ".yaml", ".toml"):
            texts.append((p, p.read_text(encoding="utf-8", errors="replace")))
    assert texts
    for p, t in texts:
        assert home not in t and str(root) not in t and str(tmp_path) not in t, p
        assert user not in t, p
    assert not any(p.name.startswith(".spec-") for p in root.iterdir())  # 入力の spec（実パス入り）は残らない

    # 集計表：冪等
    s1 = (root / "summary.md").read_text(encoding="utf-8")
    assert "| t1 |" in s1 and "| t2 |" in s1 and "open_m10" in s1 and "state 定数（keyframes[0]）" in s1
    # もう一度流すと全部飛ばして、集計表は変わらない
    before = {p: p.read_bytes() for p in (root / "runs").rglob("*") if p.is_file() and p.name not in ("status.json",)}
    out.clear()
    assert run() == 0
    assert (root / "summary.md").read_text(encoding="utf-8") == s1
    assert sum("飛ばす" in ln for ln in out) == 10  # 2 run × 5 段階
    assert before == {p: p.read_bytes() for p in (root / "runs").rglob("*") if p.is_file() and p.name not in ("status.json",)}
