"""バッチ道具の設定（TOML）の検査・解決・ハッシュ。"""
import tomllib

import pytest
from helpers_batch import EXAMPLE_TOML, tiny_toml

from crane_dance_act import batchconf, train


def resolve(text, name="t"):
    return batchconf.resolve_config(tomllib.loads(text), name)


def one(extra_top="", data="", train_t="steps = 100\n", eval_t="", export_t=None):
    text = f'[[run]]\nname = "r1"\n{extra_top}\n[run.data]\n{data}\n[run.train]\n{train_t}\n[run.eval]\n{eval_t}\n'
    if export_t is not None:
        text += f"[run.export]\n{export_t}\n"
    return text


def test_example_has_five_runs_with_night_contents():
    cfg = resolve(EXAMPLE_TOML, "example")
    assert [r["name"] for r in cfg["runs"]] == ["b1", "b2", "b3", "b4", "b5"]
    b1, b2, b3, b4, b5 = cfg["runs"]
    assert b1["data"]["sigma_rad"] == 0.0175 and b1["data"]["clip_rad"] == 0.035 and b1["data"]["tau_range_s"] == [0.02, 0.10]
    assert (b2["data"]["sigma_rad"], b2["data"]["clip_rad"], b2["data"]["tau_range_s"]) == (0.10, 0.30, [0.02, 0.25])
    assert b3["data"]["state_mode"] == "constant" and b5["data"]["env_variant"] == "no_phase"
    assert b4["train"]["policy"]["chunk_size"] == 100 and b4["train"]["policy"]["n_action_steps"] == 100
    for r in cfg["runs"]:
        t = r["train"]
        assert (t["steps"], t["batch_size"], t["policy"]["lr"]) == (20000, 32, 1e-4)
        assert r["eval"]["ckpts"] == [5000, 10000, 20000] and r["export"]["ckpts"] == [10000, 20000]
        assert [c["kind"] for c in r["eval"]["columns"]] == ["open", "closed", "ensemble"]  # 既定の列
        assert len(r["config_sha256"]) == 64


def test_defaults_are_filled_and_sha_is_stable():
    a = resolve(one())
    b = resolve(one(extra_top='description = "違う説明"'))
    r = a["runs"][0]
    assert r["data"] == {"seed": 0, "episodes": 20, "sigma_rad": 0.0175, "clip_rad": 0.035, "state_mode": "lagged", "env_variant": "full", "tau_range_s": [0.02, 0.1]}
    assert r["train"]["policy"] == {"lr": 1e-5, "lr_backbone": 1e-5, "use_vae": True, "kl_weight": 10.0, "dropout": 0.1, "chunk_size": 50, "n_action_steps": 50}
    assert r["eval"]["ckpts"] == [100] and r["export"] is None
    assert r["config_sha256"] == resolve(one())["runs"][0]["config_sha256"]
    assert r["config_sha256"] == b["runs"][0]["config_sha256"]  # description はハッシュに入らない
    c = resolve(one(train_t="steps = 100\nlr = 2e-5\n"))
    assert c["runs"][0]["config_sha256"] != r["config_sha256"]


@pytest.mark.parametrize(
    "text, frag",
    [
        (one(extra_top="bogus = 1"), "未知の鍵"),
        (one(data="bogus = 1"), "未知の鍵"),
        (one(train_t="steps = 100\nbogus = 1"), "未知の鍵"),
        (one(train_t="steps = 100\nweight_decay = 1e-4"), "未知の鍵"),  # 許可リストの外
        (one(eval_t="bogus = 1"), "未知の鍵"),
        (one(export_t="bogus = 1"), "未知の鍵"),
        (one(train_t=""), "steps は必須"),
        (one(train_t='steps = "100"'), "整数"),
        (one(train_t="steps = 0"), "整数"),
        (one(train_t="steps = 100\nlr = -1.0"), "lr"),
        (one(train_t="steps = 100\nlr = 0.0"), "lr"),
        (one(train_t="steps = 100\nuse_vae = 1"), "use_vae"),
        (one(train_t="steps = 100\ndropout = 1.5"), "dropout"),
        (one(train_t="steps = 100\nchunk_size = 0"), "chunk_size"),
        (one(train_t="steps = 100\nchunk_size = 20\nn_action_steps = 30"), "超える"),
        (one(train_t="steps = 100\nn_action_steps = 5"), "最大 m"),  # 評価の最大 m=10 より小さい
        (one(data="episodes = 21"), "episodes"),
        (one(data="episodes = 0"), "episodes"),
        (one(data='state_mode = "const"'), "state_mode"),
        (one(data='env_variant = "no_such"'), "env_variant"),
        (one(data="sigma_rad = -0.1"), "sigma_rad"),
        (one(data="tau_range_s = [0.2, 0.1]"), "tau_range_s"),
        (one(data="tau_range_s = [0.1]"), "tau_range_s"),
        (one(data="clip_rad = true"), "clip_rad"),
        (one(eval_t="ckpts = [50]"), "保存される step"),  # save_freq=1000 の倍数でも steps でもない
        (one(eval_t="ckpts = [200]"), "保存される step"),  # steps を超える
        (one(eval_t="count = 0"), "count"),
        (one(eval_t='columns = [{kind = "open"}]'), "m は"),
        (one(eval_t='columns = [{kind = "closed", m = 1}]'), "tau"),
        (one(eval_t='columns = [{kind = "ensemble", m = 5, tau = 0.06}]'), "m=1"),
        (one(eval_t='columns = [{kind = "weird", m = 1}]'), "kind"),
        (one(eval_t='columns = [{kind = "open", m = 1}, {kind = "open", m = 1}]'), "重複"),
        (one(export_t='modes = ["sideways"]'), "modes"),
        (one(export_t='formats = ["csv"]'), "formats"),
        (one(export_t="ckpts = [7]"), "保存される step"),
        ("", "[[run]]"),
        ('[[run]]\nname = "Bad Name"\n[run.train]\nsteps=1', "name"),
        ('[[run]]\nname = "-x"\n[run.train]\nsteps=1', "name"),
        ('[[run]]\nname = "a"\n[run.train]\nsteps=1\n[[run]]\nname = "a"\n[run.train]\nsteps=1', "重複"),
        ("bogus = 1\n" + one(), "未知の鍵"),
    ],
)
def test_config_errors(text, frag):
    with pytest.raises(batchconf.ConfigError) as ei:
        resolve(text)
    assert frag in str(ei.value), str(ei.value)


def test_load_config_reports_unreadable_file(tmp_path):
    with pytest.raises(batchconf.ConfigError):
        batchconf.load_config(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("[[run\n", encoding="utf-8")
    with pytest.raises(batchconf.ConfigError):
        batchconf.load_config(bad)
    ok = tmp_path / "ok.toml"
    ok.write_text(tiny_toml(), encoding="utf-8")
    assert batchconf.load_config(ok)["name"] == "ok"


def test_resolve_policy_allow_list_and_ranges():
    assert train.resolve_policy({"chunk_size": 100})["n_action_steps"] == 100
    with pytest.raises(ValueError, match="許可リスト"):
        train.resolve_policy({"optimizer_weight_decay": 1.0})
    with pytest.raises(ValueError):
        train.resolve_policy({"kl_weight": -1.0})
    with pytest.raises(ValueError):
        train.resolve_policy({"use_vae": "false"})  # 文字列で通さない
    p = train.resolve_policy({"lr": 3e-4, "use_vae": False, "kl_weight": 5, "dropout": 0.0})
    cmd = train.build_command("/x/ds", "local/x", "/x/out", 10, 8, 1000, 10, 0, "cpu", policy=p)
    for f in ("--policy.optimizer_lr=0.0003", "--policy.use_vae=false", "--policy.kl_weight=5.0", "--policy.dropout=0.0", "--policy.chunk_size=50", "--policy.n_action_steps=50"):
        assert f in cmd


def test_stage_hashes_chain():
    cfg = resolve(EXAMPLE_TOML, "example")
    b1, b2 = cfg["runs"][0], cfg["runs"][1]
    h1 = batchconf.data_hash(b1, "s" * 64, "0.2.0")
    h2 = batchconf.data_hash(b2, "s" * 64, "0.2.0")
    assert h1 != h2 and h1 != batchconf.data_hash(b1, "t" * 64, "0.2.0") and h1 != batchconf.data_hash(b1, "s" * 64, "0.3.0")
    s_mps, s_cpu = batchconf.stage_hashes(b1, h1, "mps"), batchconf.stage_hashes(b1, h1, "cpu")
    assert s_mps["data"] == s_cpu["data"] and s_mps["train"] != s_cpu["train"] and s_mps["eval"] != s_cpu["eval"]
    assert set(s_mps) == {"data", "dataset", "train", "eval", "export"}
    # 学習設定を変えると学習以降だけが変わる
    b1b = resolve(EXAMPLE_TOML.replace("chunk_size = 50", "chunk_size = 60"), "example")["runs"][0]
    s2 = batchconf.stage_hashes(b1b, h1, "mps")
    assert s2["data"] == s_mps["data"] and s2["train"] != s_mps["train"]
