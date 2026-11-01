"""バッチ道具：計画と状態遷移（偽の学習器・評価器）・集計の冪等性・事前検査・伏せ字・dry-run。重い処理はしない。"""
import json
import os
import sys

import pytest
from helpers_batch import EXAMPLE_TOML, tiny_toml

from crane_dance_act import batch, batchconf, synth, train

HOME = "/fakehome/bobsmith"


class FakeRunners:
    """段階の実処理の偽物。calls に (段階, run, 引数) を残す。fail = {(run, 段階): 種類}。"""

    def __init__(self, fail=None, post_ng=()):
        self.calls = []
        self.fail = dict(fail or {})
        self.post_ng = set(post_ng)

    def _maybe(self, ctx, stage):
        kind = self.fail.get((ctx.name, stage))
        if kind == "raise":
            raise RuntimeError(f"想定外 {HOME}/x TOKEN=sekrit-value")
        if kind == "failure":
            raise batch.StageFailure(f"失敗 {HOME}/data API_KEY=abc123")

    def build_data(self, ctx):
        self.calls.append(("data", ctx.name))
        self._maybe(ctx, "data")
        if not ctx.npz_dir.exists():
            self._manifest(ctx.npz_dir, ctx)
        return {"npz": ctx.npz_dir.name}

    def build_dataset(self, ctx):
        self.calls.append(("dataset", ctx.name))
        self._maybe(ctx, "dataset")
        if not ctx.ds_root.exists():
            self._manifest(ctx.ds_root, ctx)
        return {"dataset": ctx.ds_root.name}

    @staticmethod
    def _manifest(d, ctx):
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({"data_config_sha256": ctx.data_hash, "choreography_sha256": synth.file_sha256(ctx.choreography_path)}), encoding="utf-8")

    def train(self, ctx):
        self.calls.append(("train", ctx.name))
        self._maybe(ctx, "train")
        kind = self.fail.get((ctx.name, "train-result"))
        steps = ctx.run["train"]["steps"]
        if kind == "preflight":
            return {"exit_code": 2, "preflight_errors": [f"装置が違う {HOME}/ds"], "post_check_errors": []}
        if kind == "crash-no-ckpt":
            ctx.train_dir.mkdir(parents=True)
            return {"exit_code": 1, "preflight_errors": [], "post_check_errors": [], "checkpoints": []}
        ck = ctx.checkpoint(steps)
        ck.mkdir(parents=True)
        if kind == "crash-with-ckpt":
            return {"exit_code": 137, "preflight_errors": [], "post_check_errors": [], "checkpoints": [ck.parent.name]}
        return {
            "exit_code": 0, "preflight_errors": [], "device_actual": ctx.device, "loss_first": 1.0, "loss_last": 0.1,
            "sec_per_step": 0.1, "sec_per_step_wall": 0.2, "total_sec": 12.0, "checkpoints": [ck.parent.name],
            "post_check_errors": ["lr が違う"] if ctx.name in self.post_ng else [],
        }  # fmt: skip

    def evaluate(self, ctx, step):
        self.calls.append(("eval", ctx.name, step))
        self._maybe(ctx, "eval")
        cols = {}
        for c in ctx.run["eval"]["columns"]:
            from crane_dance_act import evaluate

            cols[evaluate.column_name(c)] = {"spec": c, "summary": {"keyframe_max_rad": 0.0512, "timing_max_beats": 0.2, "unreached": 1, "all_pass": False}}
        p = ctx.eval_path(step)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"columns": cols, "eval_sec": 3.0}), encoding="utf-8")
        return p

    def export(self, ctx, step, mode):
        self.calls.append(("export", ctx.name, step, mode))
        self._maybe(ctx, "export")
        d = ctx.export_dir(step, mode)
        d.mkdir(parents=True)
        (d / "export.json").write_text("{}", encoding="utf-8")
        return d

    def stages(self):
        return [c[0] for c in self.calls]


class Log:
    def __init__(self):
        self.lines = []

    def line(self, msg):
        self.lines.append(msg)


class NoLog(Log):
    pass


def cfg_of(text, name="t"):
    import tomllib

    return batchconf.resolve_config(tomllib.loads(text), name)


def go(cfg, out, runners, runs=None, device="cpu"):
    masking = batch.Masking(out, home=HOME)
    log = batch.ProgressLog(out / "progress.log", masking, echo=None)
    rc = batch.run_batch(cfg, runs or cfg["runs"], out, device, runners, masking, log)
    return rc, masking


def status(out, name):
    return json.loads((out / "runs" / name / "status.json").read_text(encoding="utf-8"))


@pytest.fixture
def two():
    return cfg_of(tiny_toml(names=("r1", "r2")))


# ---------------------------------------------------------------- 計画と状態遷移
def test_full_pass_then_rerun_skips_everything(tmp_path, two):
    out = tmp_path / "o"
    fr = FakeRunners()
    rc, _ = go(two, out, fr)
    assert rc == 0
    assert fr.stages() == ["data", "dataset", "train", "eval", "export", "export"] * 1 + ["data", "dataset", "train", "eval", "export", "export"]
    st = status(out, "r1")
    assert [s for s in batch.STAGES] == list(st["stages"]) and all(r["state"] == "done" for r in st["stages"].values())
    assert st["last_attempt"]["label"] == "完了"
    assert (out / "runs" / "r1" / "resolved.json").is_file() and (out / "batch_meta.json").is_file()
    # 2 回目：全部飛ばす（実処理は 1 回も呼ばれない）
    before = {n: status(out, n)["stages"] for n in ("r1", "r2")}
    fr2 = FakeRunners()
    rc, _ = go(two, out, fr2)
    assert rc == 0 and fr2.calls == []
    assert {n: status(out, n)["stages"] for n in ("r1", "r2")} == before


def test_dataset_is_shared_by_data_hash(tmp_path):
    cfg = cfg_of(tiny_toml(names=("r1", "r2")))  # 同じデータ設定の 2 run
    out = tmp_path / "o"
    go(cfg, out, FakeRunners())
    assert sorted(p.name for p in (out / "datasets").iterdir()) == sorted([f"npz-{cfg_hash(cfg)}", f"ds-{cfg_hash(cfg)}"])


def cfg_hash(cfg):
    sha = synth.file_sha256(batch.DEFAULT_CHOREOGRAPHY)
    cver = synth.load_contract(batch.DEFAULT_CONTRACT)["contract_version"]
    return batchconf.data_hash(cfg["runs"][0], sha, cver)[:12]


def test_changed_hash_stops_without_overwriting(tmp_path, two):
    out = tmp_path / "o"
    go(two, out, FakeRunners())
    snap = (out / "runs" / "r1" / "status.json").read_text(encoding="utf-8")
    eval_before = (out / "runs" / "r1" / "eval" / "000002.json").read_text(encoding="utf-8")
    changed = cfg_of(tiny_toml(names=("r1", "r2"), steps=4, save_freq=2))  # steps が違う → 学習のハッシュが違う
    fr = FakeRunners()
    rc, _ = go(changed, out, fr)
    assert rc == 1 and fr.calls == []  # 何も実行しない
    st = status(out, "r1")
    assert st["last_attempt"]["label"] == "停止" and "ハッシュ" in st["last_attempt"]["reason"]
    assert {k: v for k, v in st.items() if k != "last_attempt"} == {k: v for k, v in json.loads(snap).items() if k != "last_attempt"}
    assert (out / "runs" / "r1" / "eval" / "000002.json").read_text(encoding="utf-8") == eval_before
    # 評価の設定だけ変えても、評価の段階で止まる（学習は飛ばされ、評価の結果は上書きされない）
    only_eval = cfg_of(tiny_toml(names=("r1", "r2")).replace("tau = 0.06", "tau = 0.08"))
    fr = FakeRunners()
    rc, _ = go(only_eval, out, fr)
    assert rc == 1 and fr.calls == [] and status(out, "r1")["last_attempt"]["label"] == "停止"


def test_failure_is_recorded_and_next_run_continues(tmp_path, two):
    out = tmp_path / "o"
    fr = FakeRunners(fail={("r1", "train-result"): "crash-no-ckpt"})
    rc, _ = go(two, out, fr)
    assert rc == 1
    a, b = status(out, "r1"), status(out, "r2")
    assert a["last_attempt"]["label"] == "失敗（学習）" and a["stages"]["train"]["state"] == "failed"
    assert "eval" not in a["stages"]
    assert b["last_attempt"]["label"] == "完了"  # 次の run は進む
    assert ("train", "r2") in fr.calls


def test_interrupted_training_is_distinguished_and_moves_on(tmp_path, two):
    out = tmp_path / "o"
    fr = FakeRunners(fail={("r1", "train-result"): "crash-with-ckpt"})
    rc, _ = go(two, out, fr)
    a = status(out, "r1")
    assert rc == 1 and a["last_attempt"]["label"] == "中断" and a["stages"]["train"]["state"] == "interrupted"
    assert status(out, "r2")["last_attempt"]["label"] == "完了"
    # 同じハッシュで再実行しても、残った学習の出力を上書き・削除せず止まる
    fr2 = FakeRunners()
    rc, _ = go(two, out, fr2, runs=[two["runs"][0]])
    assert rc == 1 and ("train", "r1") not in fr2.calls
    assert status(out, "r1")["last_attempt"]["label"] == "停止"
    assert (out / "runs" / "r1" / "train" / "checkpoints").is_dir()


def test_preflight_failure_of_training_and_unexpected_errors(tmp_path, two):
    out = tmp_path / "o"
    fr = FakeRunners(fail={("r1", "train-result"): "preflight", ("r2", "eval"): "raise"})
    rc, _ = go(two, out, fr)
    assert rc == 1
    assert status(out, "r1")["last_attempt"]["label"] == "失敗（学習）"
    assert status(out, "r2")["last_attempt"]["label"] == "失敗（評価）"
    # 学習前の検査で止まったときは出力先が作られていないので、直して同じハッシュで再開できる
    fr2 = FakeRunners()
    rc, _ = go(two, out, fr2)
    assert rc == 0 and ("train", "r1") in fr2.calls and ("data", "r1") not in fr2.calls


def test_post_check_ng_goes_on_to_eval_and_is_marked(tmp_path, two):
    out = tmp_path / "o"
    fr = FakeRunners(post_ng={"r1"})
    rc, _ = go(two, out, fr)
    a = status(out, "r1")
    assert rc == 1  # 学習の失敗ではないが、正常でもない
    assert a["last_attempt"]["label"] == "完了（post_check NG）" and a["stages"]["train"]["state"] == "done"
    assert ("eval", "r1", 2) in fr.calls and ("export", "r1", 2, "open") in fr.calls
    assert "post_check NG" in (out / "summary.md").read_text(encoding="utf-8")


def test_partial_eval_is_resumed_without_redoing_finished_ckpts(tmp_path):
    cfg = cfg_of(tiny_toml(names=("r1",), steps=4, save_freq=2).replace("ckpts = [4]\ncount", "ckpts = [2, 4]\ncount"))
    out = tmp_path / "o"
    fr = FakeRunners(fail={("r1", "eval"): "failure"})
    rc, _ = go(cfg, out, fr)
    assert rc == 1 and status(out, "r1")["stages"]["eval"]["state"] == "failed"
    fr2 = FakeRunners()
    rc, _ = go(cfg, out, fr2)
    assert rc == 0
    assert [c for c in fr2.calls if c[0] == "eval"] == [("eval", "r1", 2), ("eval", "r1", 4)]  # 偽の評価器は両方 失敗していた
    fr3 = FakeRunners()
    go(cfg, out, fr3)
    assert fr3.calls == []


# ---------------------------------------------------------------- 集計表
def test_summary_is_rebuilt_idempotently_from_results(tmp_path, two):
    out = tmp_path / "o"
    go(two, out, FakeRunners())
    s1 = (out / "summary.md").read_text(encoding="utf-8")
    go(two, out, FakeRunners())  # 2 回流す
    s2 = (out / "summary.md").read_text(encoding="utf-8")
    assert s1 == s2
    masking = batch.Masking(out, home=HOME)
    assert batch.build_summary(out, two, masking) == s1 and batch.build_summary(out, two, masking) == s1
    lines = s1.splitlines()
    header = next(ln for ln in lines if ln.startswith("| run |"))
    for col in ("観測の設計", "データ（本数・σ・切り詰め・τ）", "lr", "batch", "steps", "chunk", "use_vae", "開始", "終了", "所要（学習／評価）", "ckpt", "open_m10", "closed_m10_tau0.06", "書き出しの場所", "状態"):
        assert col in header
    row = next(ln for ln in lines if ln.startswith("| r1 |"))
    assert "0.051 / 0.20 未到達1" in row and "12s／3s" in row and "000002-open" in row and "000002-closed" in row and "完了" in row
    assert s1.count("| r1 |") == 1 and s1.count("| r2 |") == 1  # 追記しない


def test_summary_marks_unrun_runs(tmp_path, two):
    out = tmp_path / "o"
    out.mkdir()
    s = batch.build_summary(out, two, batch.Masking(out, home=HOME))
    assert s.count("未実行") == 2


def test_free_text_is_masked_in_status_summary_and_progress(tmp_path, two):
    out = tmp_path / "o"
    fr = FakeRunners(fail={("r1", "dataset"): "failure", ("r2", "eval"): "raise"})
    go(two, out, fr)
    texts = [p.read_text(encoding="utf-8") for p in out.rglob("*") if p.is_file() and p.suffix in (".json", ".md", ".log")]
    blob = "\n".join(texts)
    assert blob
    for secret in (HOME, "bobsmith", "abc123", "sekrit-value"):
        assert secret not in blob, secret
    assert "失敗（データセット）" in (out / "summary.md").read_text(encoding="utf-8")


def test_status_write_is_atomic(tmp_path, monkeypatch):
    p = tmp_path / "s.json"
    batch.write_json_atomic(p, {"a": 1})
    real = os.replace
    calls = []
    monkeypatch.setattr(os, "replace", lambda a, b: (calls.append((os.path.basename(a), os.path.basename(b))), real(a, b))[1])
    batch.write_json_atomic(p, {"a": 2})
    assert calls and calls[0][1] == "s.json" and calls[0][0].startswith(".s.json.tmp")
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 2}
    # 書き込みの途中で落ちても、元のファイルは壊れず、作りかけも残らない
    monkeypatch.setattr(os, "replace", lambda a, b: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        batch.write_json_atomic(p, {"a": 3})
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 2} and [x.name for x in tmp_path.iterdir()] == ["s.json"]


def test_progress_lines_for_stages_and_training(tmp_path, monkeypatch):
    cfg = cfg_of(tiny_toml(names=("r1",)))
    ctx_dir = tmp_path / "o"
    ctx_dir.mkdir()
    masking = batch.Masking(ctx_dir, home=HOME)
    log = Log()
    ctx = batch.RunCtx(run=cfg["runs"][0], out_root=ctx_dir, device="cpu", hashes={}, data_hash="a" * 64, log=log, masking=masking)

    def fake_child(argv, log_path, masker, on_line=None, env=None):
        for ln in ("PROGRESS 50 0.9", "PROGRESS 1950 0.5", "PROGRESS 2000 0.4", "PROGRESS 2050 0.3", "PROGRESS 4100 0.2", "noise"):
            on_line(ln)
        res = ctx.run_dir / "results" / "train.json"
        res.write_text(json.dumps({"exit_code": 0}), encoding="utf-8")
        return 0, ""

    monkeypatch.setattr(batch, "run_child", fake_child)
    assert batch.Runners(masking).train(ctx) == {"exit_code": 0}
    steps = [ln.split()[3] for ln in log.lines]
    assert steps == ["50", "2000", "4100"]  # 2000 step ごとの境目だけ
    # 実行の流れの行（段階の開始・終了）
    out = tmp_path / "o2"
    lg = Log()
    batch.run_batch(cfg, cfg["runs"], out, "cpu", FakeRunners(), masking, lg)
    assert any("r1 学習 開始" in ln for ln in lg.lines) and any("r1 学習 終了" in ln for ln in lg.lines)


# ---------------------------------------------------------------- 子プロセスの出力の伏せ字
def test_child_output_is_masked_before_saving(tmp_path):
    masking = batch.Masking(tmp_path, home=HOME)
    code = (
        "import sys;"
        f"print('path {HOME}/work/data.npz');"
        "print('API_KEY=top-secret-value');"
        f"sys.stderr.write('native: {HOME}/lib/x.so loaded\\n');"
        "sys.stderr.write('TOKEN=\"multi\\n'); sys.stderr.write('line secret\"\\n');"
        "print('PROGRESS 50 0.5')"
    )
    seen = []
    rc, last = batch.run_child([sys.executable, "-c", code], tmp_path / "logs" / "c.log", masking.log_masker(), on_line=seen.append)
    saved = (tmp_path / "logs" / "c.log").read_text(encoding="utf-8")
    assert rc == 0
    for secret in (HOME, "bobsmith", "top-secret-value", "multi", "line secret"):
        assert secret not in saved and all(secret not in s for s in seen) and secret not in last
    assert "PROGRESS 50 0.5" in seen


# ---------------------------------------------------------------- caffeinate
def test_caffeinate_is_started_for_own_pid_or_warns(tmp_path):
    masking = batch.Masking(tmp_path)
    log = Log()
    got = {}

    class P:
        def __init__(self, argv):
            got["argv"] = argv

    proc = batch.start_caffeinate(log, which=lambda n: "/usr/bin/caffeinate", popen=P, pid=4242)
    assert got["argv"] == ["/usr/bin/caffeinate", "-i", "-w", "4242"] and isinstance(proc, P)
    assert any("蓋" in ln for ln in log.lines)  # -i は蓋を閉じたときの眠りは止めない
    log2 = Log()
    assert batch.start_caffeinate(log2, which=lambda n: None) is None
    assert len(log2.lines) == 1 and "警告" in log2.lines[0]
    log3 = Log()

    def boom(argv):
        raise OSError("no")

    assert batch.start_caffeinate(log3, which=lambda n: "/x", popen=boom) is None and "警告" in log3.lines[0]


# ---------------------------------------------------------------- 全体の事前検査
PF = {"check_binary": False}


def pf(tmp_path, **kw):
    base = {"which": lambda n: "/usr/bin/caffeinate", "disk_usage": lambda p: type("U", (), {"free": 500e9})(), "env": {}, **PF}
    base.update(kw)
    return batch.global_preflight({}, kw.pop("device", "cpu") if "device" in kw else "cpu", tmp_path / "o", **base)


def test_preflight_cases(tmp_path):
    assert pf(tmp_path) == ([], [])
    errs, _ = batch.global_preflight({}, "mps", tmp_path / "o", mps_available=lambda: False, env={}, which=lambda n: "x", **PF)
    assert any("MPS" in e for e in errs)
    errs, _ = batch.global_preflight({}, "cpu", tmp_path / "o", mps_available=lambda: False, env={}, which=lambda n: "x", **PF)
    assert errs == []  # --device cpu では MPS の検査を外す
    errs, _ = batch.global_preflight({}, "mps", tmp_path / "o", mps_available=lambda: True, env={"ACCELERATE_USE_CPU": "1"}, which=lambda n: "x", **PF)
    assert any("ACCELERATE_USE_CPU" in e for e in errs)
    errs, _ = batch.global_preflight({}, "cpu", tmp_path / "o", env={"ACCELERATE_TORCH_DEVICE": "mps"}, which=lambda n: "x", **PF)
    assert any("ACCELERATE_TORCH_DEVICE" in e for e in errs)
    errs, _ = pf(tmp_path, disk_usage=lambda p: type("U", (), {"free": 1e9})())
    assert any("ディスク" in e for e in errs)
    errs, warns = pf(tmp_path, which=lambda n: None)
    assert errs == [] and any("caffeinate" in w for w in warns)  # caffeinate が無いのは警告だけ
    errs, _ = batch.global_preflight({}, "cpu", tmp_path / "o", env={}, which=lambda n: "x", lerobot_train=tmp_path / "nope", check_binary=True)
    assert any("lerobot-train" in e for e in errs)
    errs, _ = batch.global_preflight({}, "cpu", tmp_path / "o", env={}, which=lambda n: "x", choreography_path=tmp_path / "none.json", **PF)
    assert any("読めない" in e for e in errs)


def test_preflight_detects_changed_contract_or_choreography(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "batch_meta.json").write_text(json.dumps({"contract_version": "0.2.0", "choreography_sha256": "0" * 64}), encoding="utf-8")
    errs, _ = batch.global_preflight({}, "cpu", out, env={}, which=lambda n: "x", **PF)
    assert any("前回" in e for e in errs)


# ---------------------------------------------------------------- コマンド
def write_cfg(tmp_path, text, name="c"):
    p = tmp_path / f"{name}.toml"
    p.write_text(text, encoding="utf-8")
    return p


def cli(tmp_path, args, runners=None, **kw):
    out = []
    rc = batch.cli_main(args, runners=runners, echo=out.append, caffeinate=lambda log: None,
                        preflight_kwargs={"check_binary": False, "env": {}, "which": lambda n: "x", **kw})  # fmt: skip
    return rc, out


def test_dry_run_prints_plan_and_creates_nothing(tmp_path):
    cfgp = write_cfg(tmp_path, EXAMPLE_TOML, "example")
    root = tmp_path / "root"
    fr = FakeRunners()
    rc, out = cli(tmp_path, ["--config", str(cfgp), "--dry-run", "--device", "cpu", "--out-root", str(root)], runners=fr)
    text = "\n".join(out)
    assert rc == 0 and fr.calls == [] and not root.exists()
    assert [ln.split()[1].rstrip(":") for ln in text.splitlines() if ln.startswith("run ")] == ["b1", "b2", "b3", "b4", "b5"]
    assert "ds-" in text and "state constant" in text and "env no_phase" in text and "chunk 100" in text
    assert "段階: データ=新規" in text and "書き出し=新規" in text
    assert HOME not in text and os.path.expanduser("~") not in text  # 実パスを出さない


def test_dry_run_reads_existing_status_without_writing(tmp_path):
    cfgp = write_cfg(tmp_path, tiny_toml(names=("r1",)), "c")
    root = tmp_path / "root"
    cfg = batchconf.load_config(cfgp)
    go(cfg, root, FakeRunners())
    snap = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    rc, out = cli(tmp_path, ["--config", str(cfgp), "--dry-run", "--device", "cpu", "--out-root", str(root)])
    assert rc == 0 and "完了済み（飛ばす）" in "\n".join(out)
    assert snap == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_cli_exit_codes(tmp_path):
    cfgp = write_cfg(tmp_path, tiny_toml(names=("r1", "r2")), "c")
    root = tmp_path / "root"
    base = ["--config", str(cfgp), "--device", "cpu", "--out-root", str(root)]
    # 全部成功 → 0
    rc, _ = cli(tmp_path, base, runners=FakeRunners())
    assert rc == 0
    # どれか失敗 → 1
    rc, _ = cli(tmp_path, ["--config", str(cfgp), "--device", "cpu", "--out-root", str(tmp_path / "r2")], runners=FakeRunners(fail={("r2", "train-result"): "crash-no-ckpt"}))
    assert rc == 1
    # 設定の誤り・--only の誤り → 2
    rc, out = cli(tmp_path, ["--config", str(write_cfg(tmp_path, 'bogus = 1', "bad")), "--device", "cpu"])
    assert rc == 2 and "設定の誤り" in out[0]
    rc, _ = cli(tmp_path, [*base, "--only", "nope"])
    assert rc == 2
    # --only は選んだ run だけ
    fr = FakeRunners()
    rc, _ = cli(tmp_path, ["--config", str(cfgp), "--device", "cpu", "--out-root", str(tmp_path / "r3"), "--only", "r2"], runners=fr)
    assert rc == 0 and {c[1] for c in fr.calls} == {"r2"}


def test_preflight_failure_exits_2_before_any_run(tmp_path):
    cfgp = write_cfg(tmp_path, tiny_toml(names=("r1",)), "c")
    root = tmp_path / "root"
    fr = FakeRunners()
    rc, out = cli(tmp_path, ["--config", str(cfgp), "--device", "mps", "--out-root", str(root)], runners=fr, mps_available=lambda: False)
    assert rc == 2 and fr.calls == [] and any("PREFLIGHT NG" in ln and "MPS" in ln for ln in out)
    assert not root.exists()  # 始める前に止まるので何も作らない
    rc, out = cli(tmp_path, ["--config", str(cfgp), "--device", "cpu", "--out-root", str(root)], runners=fr, disk_usage=lambda p: type("U", (), {"free": 1e9})())
    assert rc == 2 and fr.calls == [] and not root.exists()


def test_config_path_is_resolved_from_act_dir_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert batch.resolve_config_path("batches/x.toml") == batch.ACT_DIR / "batches" / "x.toml"
    assert batch.resolve_config_path(str(tmp_path / "y.toml")) == tmp_path / "y.toml"
    assert batch.BATCH_STEP.is_file() and batch.DEFAULT_CONTRACT.is_file()


def test_train_module_is_unchanged_for_defaults():
    # train_act.py の既存の呼び方（policy なし）は従来の chunk 50 のまま
    cmd = train.build_command("/x/ds", "local/x", "/x/out", 10, 8, 1000, 10, 0, "mps")
    assert "--policy.chunk_size=50" in cmd and not any(a.startswith("--policy.optimizer_lr") for a in cmd)
