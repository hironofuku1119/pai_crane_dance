"""バッチ道具の本体。設定（TOML）の [[run]] を順に、データ → データセット → 学習 → 評価 → 書き出し → 集計 と進める。

- 出力の根: act/out/batch/<設定名>/ 。run ごとに runs/<name>/status.json（原子的に書く）。
- 同じハッシュで完了済みの段階は飛ばす。ハッシュが違えば止める（上書き・自動削除はしない）。
- 学習・評価・書き出し・データセット変換は子プロセス。出力は全部 LogMasker を通してから保存する。
- 各 run の失敗は記録して次へ。終了コード: 0＝全部成功 / 1＝どれか失敗（中断・停止・post_check NG を含む）/ 2＝始める前の検査で止めた。
- 閉ループは仮の機械（一次遅れ）であり Gazebo の閉ループではない。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from crane_dance_act import batchconf, state_mode, synth, train, variants

ACT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ACT_DIR.parent
DEFAULT_CONTRACT = REPO_ROOT / "contract" / "contract.json"
DEFAULT_CHOREOGRAPHY = REPO_ROOT / "contract" / "choreography.json"
BATCH_STEP = ACT_DIR / "scripts" / "batch_step.py"
DEFAULT_MIN_FREE_GB = 10.0
STAGES = ("data", "dataset", "train", "eval", "export")
STAGE_JA = {"data": "データ", "dataset": "データセット", "train": "学習", "eval": "評価", "export": "書き出し"}
LID_NOTE = "caffeinate -i は蓋を閉じたときの眠りは止めない（電源につないで蓋を開けたままにすること）"
CLOSED_NOTE = "閉ループは仮の機械（一次遅れ）の中で回した指令であり、Gazebo の閉ループではない"
OK = "完了"
OK_POST_NG = "完了（post_check NG）"


class StageFailure(Exception):
    """段階の失敗。interrupted=True は途中で落ちた（チェックポイントが残っている）。"""

    def __init__(self, reason, interrupted=False):
        super().__init__(reason)
        self.reason = reason
        self.interrupted = interrupted


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_text_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 伏せ字・進み具合
class Masking:
    """自由記述を保存・表示する前に通す。実パス・ホーム・利用者名・秘密を伏せる（mask_log と同じ規則）。"""

    def __init__(self, out_root=None, home=None):
        named = [(ACT_DIR, "<act>"), (REPO_ROOT, "<repo>")]
        if out_root is not None:
            named.append((Path(out_root).resolve(), "<batch_out>"))
        self.named = tuple(named)
        self.home = home

    def text(self, s):
        return train.mask_log(str(s), home=self.home, named_paths=self.named)

    def line(self, s, limit=200):
        one = " ".join(self.text(s).split())
        return one if len(one) <= limit else one[: limit - 1] + "…"

    def log_masker(self):
        return train.LogMasker(home=self.home, named_paths=self.named)


class ProgressLog:
    """進み具合を 1 行ずつ（段階の開始・終了・学習の 2000 step ごと・失敗）。ファイルへ追記し、echo にも出す。"""

    def __init__(self, path, masking, echo=print):
        self.path = Path(path) if path is not None else None
        self.masking = masking
        self.echo = echo

    def line(self, msg):
        text = f"[{time.strftime('%H:%M:%S')}] {self.masking.line(msg, limit=300)}"
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        if self.echo is not None:
            self.echo(text)


# ---------------------------------------------------------------- 文脈
@dataclass
class RunCtx:
    run: dict
    out_root: Path
    device: str
    hashes: dict
    data_hash: str
    contract_path: Path = DEFAULT_CONTRACT
    choreography_path: Path = DEFAULT_CHOREOGRAPHY
    log: object = None  # ProgressLog
    masking: object = None  # Masking
    extra: dict = field(default_factory=dict)

    @property
    def name(self):
        return self.run["name"]

    @property
    def run_dir(self):
        return self.out_root / "runs" / self.name

    @property
    def train_dir(self):
        return self.run_dir / "train"

    @property
    def npz_dir(self):
        return self.out_root / "datasets" / f"npz-{self.data_hash[:12]}"

    @property
    def ds_root(self):
        return self.out_root / "datasets" / f"ds-{self.data_hash[:12]}"

    @property
    def repo_id(self):
        return f"local/crane_dance_batch_{self.data_hash[:12]}"

    def eval_path(self, step):
        return self.run_dir / "eval" / f"{step:06d}.json"

    def export_dir(self, step, mode):
        return self.run_dir / "export" / f"{step:06d}-{mode}"

    def checkpoint(self, step):
        return self.train_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"


# ---------------------------------------------------------------- 子プロセス
def run_child(argv, log_path, masker, on_line=None, env=None):
    """子プロセスを走らせ、標準出力・標準エラーを全部 masker に通してから log_path に保存する。
    (終了コード, 最後の空でない行（伏せ済み）) を返す。"""
    last = ""

    def on_masked(masked):
        nonlocal last
        for ln in masked.splitlines():
            if ln.strip():
                last = ln.strip()
                if on_line is not None:
                    on_line(ln)

    rc = train.run_streamed(argv, log_path, masker, on_masked=on_masked, env=env)
    return rc, last


class Runners:
    """段階の実処理（本物）。テストでは同じ形の偽物に差し替える。"""

    def __init__(self, masking=None, python=None, env=None):
        self.masking = masking
        self.python = python or sys.executable
        self.env = env

    # --- データ（npz）：その場で作る（numpy のみ・ネイティブの出力なし）
    def build_data(self, ctx):
        d = ctx.run["data"]
        out = ctx.npz_dir
        contract = synth.load_contract(ctx.contract_path)
        choreo = synth.load_choreography(ctx.choreography_path)
        sha = synth.file_sha256(ctx.choreography_path)
        if out.exists():
            check_dataset_manifest(out, ctx.data_hash, sha)
            return {"npz": out.name, "reused": True}
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp_root = out.parent / f".{out.name}.tmp-{uuid.uuid4().hex[:8]}"
        try:
            tmp_root.mkdir()
            sm = state_mode.constant_from_choreography(choreo) if d["state_mode"] == "constant" else state_mode.lagged()
            base = tmp_root / "base"
            synth.write_varied_set(
                contract, choreo, sha, d["episodes"], d["seed"], base,
                tau_range_s=d["tau_range_s"], sigma_rad=d["sigma_rad"], clip_rad=d["clip_rad"], state_mode=sm,
            )  # fmt: skip
            mpath = base / "manifest.json"
            m = read_json(mpath)
            m["data_config_sha256"] = ctx.data_hash
            m["data_config"] = d
            write_json_atomic(mpath, m)
            final = base
            if d["env_variant"] != "full":
                final = tmp_root / "final"
                variants.make_variant_set(base, d["env_variant"], final, contract)
            os.rename(final, out)
        finally:
            if tmp_root.exists():
                shutil.rmtree(tmp_root)
        return {"npz": out.name, "reused": False}

    # --- データセット（LeRobotDataset への変換）：子プロセス
    def build_dataset(self, ctx):
        out = ctx.ds_root
        if out.exists():
            check_dataset_manifest(out, ctx.data_hash, synth.file_sha256(ctx.choreography_path))
            return {"dataset": out.name, "reused": True}
        spec = {
            "npz_dir": str(ctx.npz_dir), "out_root": str(out), "repo_id": ctx.repo_id,
            "contract_path": str(ctx.contract_path), "choreography_path": str(ctx.choreography_path),
        }  # fmt: skip
        rc, last = self._child(ctx, "dataset", spec, "dataset")
        if rc != 0 or not out.exists():
            raise StageFailure(f"データセットの変換に失敗（終了コード {rc}）: {self.masking.line(last)}")
        return {"dataset": out.name, "reused": False}

    def train(self, ctx):
        t = ctx.run["train"]
        res_path = ctx.run_dir / "results" / "train.json"
        res_path.parent.mkdir(parents=True, exist_ok=True)
        if res_path.exists():
            res_path.unlink()  # 同じ出力先の古い結果を読まない（出力先が既にあれば run_training が止める）
        spec = {
            "dataset_root": str(ctx.ds_root), "repo_id": ctx.repo_id, "output_dir": str(ctx.train_dir),
            "steps": t["steps"], "save_freq": t["save_freq"], "device": ctx.device, "batch_size": t["batch_size"],
            "seed": t["seed"], "num_workers": t["num_workers"], "policy": t["policy"],
            "contract_path": str(ctx.contract_path), "choreography_path": str(ctx.choreography_path),
            "result_path": str(res_path),
        }  # fmt: skip
        last_logged = {"bucket": -1}

        def on_line(line):
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "PROGRESS" and parts[1].isdigit():
                step = int(parts[1])
                loss = parts[2] if len(parts) > 2 else "?"
                if step // train_progress_every() > last_logged["bucket"]:
                    last_logged["bucket"] = step // train_progress_every()
                    ctx.log.line(f"{ctx.name} 学習 step {step} loss {loss}")

        rc, last = self._child(ctx, "train", spec, "train", on_line=on_line)
        if not res_path.exists():
            raise StageFailure(f"学習の子プロセスが結果を残さず終了（終了コード {rc}）: {self.masking.line(last)}")
        return read_json(res_path)

    def evaluate(self, ctx, step):
        out = ctx.eval_path(step)
        out.parent.mkdir(parents=True, exist_ok=True)
        spec = {
            "checkpoint": str(ctx.checkpoint(step)), "columns": ctx.run["eval"]["columns"], "device": ctx.device,
            "count": ctx.run["eval"]["count"], "env_variant": ctx.run["data"]["env_variant"],
            "contract_path": str(ctx.contract_path), "choreography_path": str(ctx.choreography_path),
            "result_path": str(out),
        }  # fmt: skip
        rc, last = self._child(ctx, "eval", spec, f"eval-{step:06d}")
        if rc != 0 or not out.exists():
            raise StageFailure(f"評価に失敗（ckpt {step}・終了コード {rc}）: {self.masking.line(last)}")
        return out

    def export(self, ctx, step, mode):
        x = ctx.run["export"]
        out = ctx.export_dir(step, mode)
        out.parent.mkdir(parents=True, exist_ok=True)
        spec = {
            "checkpoint": str(ctx.checkpoint(step)), "out_dir": str(out), "mode": mode, "m": x["m"], "tau": x["tau"],
            "formats": x["formats"], "device": ctx.device, "index": x["index"], "env_variant": ctx.run["data"]["env_variant"],
            "contract_path": str(ctx.contract_path), "choreography_path": str(ctx.choreography_path),
        }  # fmt: skip
        rc, last = self._child(ctx, "export", spec, f"export-{step:06d}-{mode}")
        if rc != 0 or not out.exists():
            raise StageFailure(f"書き出しに失敗（ckpt {step}・{mode}・終了コード {rc}）: {self.masking.line(last)}")
        return out

    def _child(self, ctx, kind, spec, tag, on_line=None):
        masker = self.masking.log_masker()
        with tempfile.TemporaryDirectory(dir=ctx.out_root, prefix=".spec-") as td:  # 実パスを含む入力は残さない
            sp = Path(td) / f"{tag}.json"
            sp.write_text(json.dumps(spec), encoding="utf-8")
            env = train.build_env(self.env if self.env is not None else os.environ)
            return run_child(
                [self.python, str(BATCH_STEP), kind, str(sp)],
                ctx.run_dir / "logs" / f"{tag}.child.log", masker, on_line=on_line, env=env,
            )  # fmt: skip


def train_progress_every():
    from crane_dance_act import batch_steps

    return batch_steps.PROGRESS_EVERY


def check_dataset_manifest(root, data_hash, choreography_sha256):
    """名前がハッシュで決まるデータ（npz・データセット）を再利用してよいか。違えば StageFailure（上書き・削除しない）。"""
    p = Path(root) / "manifest.json"
    if not p.is_file():
        raise StageFailure(f"{Path(root).name} に manifest.json が無い（作りかけか別物。手で確かめる）")
    m = read_json(p)
    if m.get("data_config_sha256") != data_hash or m.get("choreography_sha256") != choreography_sha256:
        raise StageFailure(f"{Path(root).name} の manifest がいまの設定・振付と合わない（上書きしない。手で確かめる）")


# ---------------------------------------------------------------- 事前検査（S8）
def global_preflight(
    cfg, device, out_root, contract_path=DEFAULT_CONTRACT, choreography_path=DEFAULT_CHOREOGRAPHY, env=None,
    mps_available=None, disk_usage=shutil.disk_usage, min_free_gb=DEFAULT_MIN_FREE_GB, which=shutil.which,
    lerobot_train=None, check_binary=True,
):  # fmt: skip
    """全体の事前検査を 1 回。(エラーのリスト, 警告のリスト)。エラーがあれば run を始めない（終了コード 2）。"""
    errs, warns = [], []
    env = dict(os.environ if env is None else env)
    if device not in ("mps", "cpu"):
        errs.append(f"device は mps か cpu: {device}")
    if device == "mps":
        avail = train._mps_available if mps_available is None else mps_available
        if not avail():
            errs.append("MPS が使えない（試験は --device cpu）")
    errs += train.check_device_env(env, device)
    try:
        contract = synth.load_contract(contract_path)
        choreo = synth.load_choreography(choreography_path)
        sha = synth.file_sha256(choreography_path)
        cver = str(contract["contract_version"])
        fps = int(contract["recording"]["fps"])
        if fps < 1 or 10**9 % fps:
            errs.append(f"契約の fps {fps} が 1e9 を割り切らない")
        if not choreo.get("keyframes"):
            errs.append("振付に keyframes が無い")
        meta_path = Path(out_root) / "batch_meta.json"
        if meta_path.is_file():
            meta = read_json(meta_path)
            if meta.get("contract_version") != cver or meta.get("choreography_sha256") != sha:
                errs.append("このバッチの前回と契約の版か振付の sha256 が違う（出力先を変えるか、手で確かめる）")
    except (OSError, ValueError, KeyError, TypeError) as e:
        errs.append(f"契約か振付を読めない: {type(e).__name__}")
    probe = Path(out_root)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_gb = disk_usage(probe).free / 1e9
    if free_gb < min_free_gb:
        errs.append(f"ディスクの空きが {free_gb:.1f} GB（下限 {min_free_gb} GB）")
    if check_binary:
        exe = Path(train.LEROBOT_TRAIN if lerobot_train is None else lerobot_train)
        if not exe.exists():
            errs.append("lerobot-train が見つからない（act/.venv を作る）")
    if which("caffeinate") is None:
        warns.append("caffeinate が見つからない（眠りを止められない。続ける）")
    return errs, warns


def start_caffeinate(log, which=shutil.which, popen=subprocess.Popen, pid=None):
    """自分の pid に対して caffeinate -i を張る。使えなければ警告 1 行で続ける（None を返す）。"""
    exe = which("caffeinate")
    if exe is None:
        log.line("警告: caffeinate が使えない。眠りを止めずに続ける")
        return None
    try:
        proc = popen([exe, "-i", "-w", str(os.getpid() if pid is None else pid)])
    except OSError as e:
        log.line(f"警告: caffeinate を起動できない（{type(e).__name__}）。続ける")
        return None
    log.line(f"caffeinate -i を張った。{LID_NOTE}")
    return proc


# ---------------------------------------------------------------- 状態遷移（S5）
def new_status(ctx):
    return {"name": ctx.name, "config_sha256": ctx.run["config_sha256"], "stages": {}, "last_attempt": None}


def load_status(path, ctx):
    p = Path(path)
    return read_json(p) if p.is_file() else new_status(ctx)


def execute_run(ctx, runners, only_stages=None):
    """1 本の run を最後まで（または失敗・停止・中断まで）進める。ラベル（"完了" など）を返す。status.json は段階ごとに原子的に書く。"""
    mk, log = ctx.masking, ctx.log
    sp = ctx.run_dir / "status.json"
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    status = load_status(sp, ctx)
    plan = [s for s in STAGES if s in ctx.hashes]

    def attempt(label, reason=""):
        status["last_attempt"] = {"label": label, "reason": mk.line(reason), "at": now_str()}
        write_json_atomic(sp, status)
        log.line(f"{ctx.name} {label}" + (f": {reason}" if reason else ""))
        return label

    # 停止の判定は何も書く前に（ハッシュが違えば上書き・自動削除しない）
    for stage in plan:
        rec = status["stages"].get(stage)
        if rec and rec["hash"] != ctx.hashes[stage]:
            return attempt("停止", f"{STAGE_JA[stage]}のハッシュが前回と違う（設定が変わった。上書き・削除はしない。別の出力先にするか手で確かめる）")
    write_json_atomic(ctx.run_dir / "resolved.json", {"sha256": ctx.run["config_sha256"], "stage_hashes": ctx.hashes, "run": {**ctx.run, "description": mk.text(ctx.run["description"])}})
    status["config_sha256"] = ctx.run["config_sha256"]

    for stage in plan:
        if only_stages is not None and stage not in only_stages:
            continue
        rec = status["stages"].get(stage)
        if rec and rec["state"] == "done":
            log.line(f"{ctx.name} {STAGE_JA[stage]} 飛ばす（同じハッシュで完了済み）")
            continue
        if stage == "train" and ctx.train_dir.exists():
            return attempt("停止", "学習の出力先が既にある（途中までの学習が残っている。上書き・削除はしない。手で確かめる）")
        status["stages"][stage] = {"hash": ctx.hashes[stage], "state": "running", "started": now_str(), "ended": None, "detail": {}}
        write_json_atomic(sp, status)
        log.line(f"{ctx.name} {STAGE_JA[stage]} 開始")
        try:
            detail = DO_STAGE[stage](ctx, runners)
        except StageFailure as e:
            state = "interrupted" if e.interrupted else "failed"
            status["stages"][stage].update(state=state, ended=now_str(), detail={"reason": mk.line(e.reason)})
            return attempt("中断" if e.interrupted else f"失敗（{STAGE_JA[stage]}）", e.reason)
        except KeyboardInterrupt:
            status["stages"][stage].update(state="interrupted", ended=now_str(), detail={"reason": "キーボード割り込み"})
            attempt("中断", "キーボード割り込み")
            raise
        except Exception as e:  # 想定外もその run の失敗として記録して次へ
            status["stages"][stage].update(state="failed", ended=now_str(), detail={"reason": mk.line(f"{type(e).__name__}: {e}")})
            return attempt(f"失敗（{STAGE_JA[stage]}）", f"{type(e).__name__}: {e}")
        status["stages"][stage].update(state="done", ended=now_str(), detail=detail)
        write_json_atomic(sp, status)
        log.line(f"{ctx.name} {STAGE_JA[stage]} 終了")
    post = status["stages"].get("train", {}).get("detail", {}).get("post_check_errors") or []
    return attempt(OK_POST_NG if post else OK, "; ".join(post))


def _stage_data(ctx, runners):
    return runners.build_data(ctx)


def _stage_dataset(ctx, runners):
    return runners.build_dataset(ctx)


def _stage_train(ctx, runners):
    mk = ctx.masking
    res = runners.train(ctx)
    pre = res.get("preflight_errors") or []
    if pre:
        raise StageFailure("学習前の検査で止まった: " + "; ".join(mk.line(e) for e in pre))
    ckpts = list(res.get("checkpoints") or [])
    if res.get("exit_code", 1) != 0:
        # 学習の失敗と、途中で落ちた中断（チェックポイントが残っている）を分ける
        raise StageFailure(f"学習が終了コード {res.get('exit_code')} で終わった（残った ckpt: {len(ckpts)}）", interrupted=bool(ckpts))
    keys = ("exit_code", "device_actual", "loss_first", "loss_last", "sec_per_step", "sec_per_step_wall", "total_sec")
    detail = {k: res.get(k) for k in keys}
    detail["post_check_errors"] = [mk.line(e) for e in (res.get("post_check_errors") or [])]
    detail["checkpoints"] = ckpts
    return detail


def _stage_eval(ctx, runners):
    done, failed = [], []
    for step in ctx.run["eval"]["ckpts"]:
        if ctx.eval_path(step).exists():
            done.append(step)
            continue
        try:
            runners.evaluate(ctx, step)
            done.append(step)
        except StageFailure as e:
            failed.append(e.reason)
            ctx.log.line(f"{ctx.name} 評価 失敗 ckpt {step}: {e.reason}")
    if failed:
        raise StageFailure(f"{len(failed)} 件の評価が失敗: {failed[0]}")
    return {"ckpts": done}


def _stage_export(ctx, runners):
    x = ctx.run["export"]
    done, failed = [], []
    for step in x["ckpts"]:
        for mode in x["modes"]:
            if ctx.export_dir(step, mode).exists():
                done.append(f"{step:06d}-{mode}")
                continue
            try:
                runners.export(ctx, step, mode)
                done.append(f"{step:06d}-{mode}")
            except StageFailure as e:
                failed.append(e.reason)
                ctx.log.line(f"{ctx.name} 書き出し 失敗 ckpt {step} {mode}: {e.reason}")
    if failed:
        raise StageFailure(f"{len(failed)} 件の書き出しが失敗: {failed[0]}")
    return {"exports": done}


DO_STAGE = {"data": _stage_data, "dataset": _stage_dataset, "train": _stage_train, "eval": _stage_eval, "export": _stage_export}


# ---------------------------------------------------------------- 集計表（S7）
def _cell(s, closed):
    t = "nan" if s["timing_max_beats"] is None else f"{s['timing_max_beats']:.2f}"
    base = f"{s['keyframe_max_rad']:.3f} / {t}"
    return base + (f" 未到達{s['unreached']}" if closed else "") + (" ✓" if s["all_pass"] else "")


def _md(s):
    return str(s).replace("|", "/").replace("\n", " ")


def build_summary(out_root, cfg, masking):
    """全 run の status.json・resolved.json・評価の JSON から毎回作り直す markdown。同じ入力なら同じ出力（冪等）。"""
    out_root = Path(out_root)
    rows, col_names = [], []
    for cfg_run in cfg["runs"]:
        rd = out_root / "runs" / cfg_run["name"]
        run = cfg_run
        if (rd / "resolved.json").is_file():
            run = read_json(rd / "resolved.json")["run"]
        status = read_json(rd / "status.json") if (rd / "status.json").is_file() else None
        d, tr, ev = run["data"], run["train"], run["eval"]
        pol = tr["policy"]
        obs = "state 定数（keyframes[0]）" if d["state_mode"] == "constant" else "state あり（一次遅れ＋ノイズ）"
        if d["env_variant"] != "full":
            obs += f"・環境状態 {d['env_variant']}"
        data = f"{d['episodes']} 本・σ{d['sigma_rad']}・±{d['clip_rad']}・τ{d['tau_range_s'][0]}〜{d['tau_range_s'][1]}"
        stages = status["stages"] if status else {}
        trec = stages.get("train", {})
        start = trec.get("started") or "-"
        ends = [s["ended"] for s in stages.values() if s.get("ended")]
        end = max(ends) if ends else "-"
        train_sec = trec.get("detail", {}).get("total_sec")
        la = status["last_attempt"] if status and status.get("last_attempt") else None
        state = "未実行" if la is None else la["label"] + (f": {la['reason']}" if la["reason"] and la["label"] != OK else "")
        exports = sorted(p.name for p in (rd / "export").iterdir() if p.is_dir() and not p.name.startswith(".")) if (rd / "export").is_dir() else []
        results = {}
        for step in ev["ckpts"]:
            p = rd / "eval" / f"{step:06d}.json"
            if p.is_file():
                results[step] = read_json(p)
                for name in results[step]["columns"]:
                    if name not in col_names:
                        col_names.append(name)
        eval_sec = round(sum(r.get("eval_sec", 0) or 0 for r in results.values()), 1) if results else None
        elapsed = f"{'-' if train_sec is None else f'{train_sec:.0f}s'}／{'-' if eval_sec is None else f'{eval_sec:.0f}s'}"
        base = [run["name"], obs, data, pol["lr"], tr["batch_size"], tr["steps"], pol["chunk_size"], pol["use_vae"], start, end, elapsed]
        for step in ev["ckpts"]:
            rows.append((base, step, results.get(step), "<br>".join(exports) if exports else "-", state))
    head = ["run", "観測の設計", "データ（本数・σ・切り詰め・τ）", "lr", "batch", "steps", "chunk", "use_vae", "開始", "終了", "所要（学習／評価）", "ckpt"]
    lines = [
        f"# バッチ {cfg['name']} の結果",
        "",
        "この表は status.json・resolved.json・評価の JSON から毎回作り直す（追記しない・2 回作っても同じ）。",
        "セル＝キーフレーム誤差の最大 [rad] / タイミングのずれの最大 [拍]（閉ループは未到達の区間の数も）。検証データの最大。"
        "合否の目安 0.035 rad・0.1 拍（✓＝全部合格）は仮。" + CLOSED_NOTE + "。",
        "",
        "| " + " | ".join(head + col_names + ["書き出しの場所", "状態"]) + " |",
        "|" + "---|" * (len(head) + len(col_names) + 2),
    ]
    for base, step, res, exports, state in rows:
        cells = [*base, step]
        for name in col_names:
            if res is None or name not in res["columns"]:
                cells.append("-")
            else:
                cells.append(_cell(res["columns"][name]["summary"], res["columns"][name]["spec"]["kind"] != "open"))
        cells += [exports, state]
        lines.append("| " + " | ".join(_md(masking.text(c)) for c in cells) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 計画（--dry-run）
def plan_text(cfg, runs, device, out_root, contract_path, choreography_path):
    """計画だけ。何も作らない・書かない（既存の status.json は読むだけ）。"""
    contract = synth.load_contract(contract_path)
    sha = synth.file_sha256(choreography_path)
    cver = contract["contract_version"]
    out = [f"バッチ {cfg['name']}（run {len(runs)} 本・device {device}）", f"出力の根: {Path(out_root).name} 以下（act/out/batch/）", f"契約 {cver}・振付 sha256 {sha[:12]}", ""]
    for run in runs:
        dh = batchconf.data_hash(run, sha, cver)
        hs = batchconf.stage_hashes(run, dh, device)
        d, t, e, x = run["data"], run["train"], run["eval"], run["export"]
        p = t["policy"]
        status_p = Path(out_root) / "runs" / run["name"] / "status.json"
        st = read_json(status_p)["stages"] if status_p.is_file() else {}

        def mark(stage):
            if stage not in hs:
                return "なし"
            rec = st.get(stage)
            if rec is None:
                return "新規"
            return "完了済み（飛ばす）" if rec["hash"] == hs[stage] and rec["state"] == "done" else ("止まる（ハッシュ違い）" if rec["hash"] != hs[stage] else "再開")

        out.append(f"run {run['name']}: {run['description']}")
        out.append(f"  データ: ds-{dh[:12]}  {d['episodes']} 本・σ{d['sigma_rad']}・±{d['clip_rad']}・τ{d['tau_range_s'][0]}〜{d['tau_range_s'][1]}・state {d['state_mode']}・env {d['env_variant']}・seed {d['seed']}")
        out.append(f"  学習: steps {t['steps']}・batch {t['batch_size']}・lr {p['lr']}・lr_backbone {p['lr_backbone']}・chunk {p['chunk_size']}・n_action_steps {p['n_action_steps']}・use_vae {p['use_vae']}・kl {p['kl_weight']}・dropout {p['dropout']}・save_freq {t['save_freq']}・seed {t['seed']}")
        out.append(f"  評価: ckpt {','.join(str(c) for c in e['ckpts'])}・検証 {e['count']} 本・列 {', '.join(evaluate_column_names(e['columns']))}")
        out.append("  書き出し: " + ("なし" if x is None else f"ckpt {','.join(str(c) for c in x['ckpts'])}・mode {','.join(x['modes'])}・形式 {','.join(x['formats'])}（m {x['m']}・τ {x['tau']}）"))
        out.append("  段階: " + " → ".join(f"{STAGE_JA[s]}={mark(s)}" for s in STAGES if s in hs) + " → 集計")
    return "\n".join(out) + "\n"


def evaluate_column_names(columns):
    from crane_dance_act import evaluate

    return [evaluate.column_name(c) for c in columns]


# ---------------------------------------------------------------- 全体
def run_batch(cfg, runs, out_root, device, runners, masking, log, contract_path=DEFAULT_CONTRACT, choreography_path=DEFAULT_CHOREOGRAPHY):
    """選んだ run を順に実行して集計表を作る。終了コード（0 / 1）を返す。"""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    contract = synth.load_contract(contract_path)
    sha = synth.file_sha256(choreography_path)
    cver = contract["contract_version"]
    meta_path = out_root / "batch_meta.json"
    if not meta_path.exists():
        write_json_atomic(meta_path, {"config": cfg["name"], "contract_version": cver, "choreography_sha256": sha, "created": now_str()})
    labels = {}
    interrupted = False
    for run in runs:
        dh = batchconf.data_hash(run, sha, cver)
        ctx = RunCtx(run=run, out_root=out_root, device=device, hashes=batchconf.stage_hashes(run, dh, device), data_hash=dh,
                     contract_path=contract_path, choreography_path=choreography_path, log=log, masking=masking)  # fmt: skip
        try:
            labels[run["name"]] = execute_run(ctx, runners)
        except KeyboardInterrupt:
            labels[run["name"]] = "中断"
            interrupted = True
        write_text_atomic(out_root / "summary.md", build_summary(out_root, cfg, masking))
        if interrupted:
            break
    log.line("集計表を作り直した（summary.md）")
    log.line("結果: " + "・".join(f"{n}={l}" for n, l in labels.items()))
    if interrupted:
        return 130
    return 0 if all(v == OK for v in labels.values()) else 1


# ---------------------------------------------------------------- コマンド
def resolve_config_path(arg):
    """相対パスは cwd でなく act/ からの位置で解く（どこから呼んでも同じ）。"""
    p = Path(arg)
    return p if p.is_absolute() else ACT_DIR / p


def cli_main(argv=None, runners=None, echo=print, preflight_kwargs=None, caffeinate=start_caffeinate):
    import argparse

    ap = argparse.ArgumentParser(description="バッチ道具（設定の [[run]] を順に：データ → 学習 → 評価 → 書き出し → 集計）")
    ap.add_argument("--config", required=True, help="batches/<名前>.toml（相対は act/ から）")
    ap.add_argument("--only", nargs="+", metavar="RUN", help="この run だけ実行する")
    ap.add_argument("--device", choices=["mps", "cpu"], default="mps")
    ap.add_argument("--dry-run", action="store_true", help="計画だけ出して何も作らない")
    ap.add_argument("--out-root", type=Path, help="出力の根（既定: act/out/batch/<設定名>）")
    ap.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    a = ap.parse_args(argv)

    try:
        cfg = batchconf.load_config(resolve_config_path(a.config))
    except batchconf.ConfigError as e:
        echo(f"設定の誤り: {e}")
        return 2
    runs = cfg["runs"]
    if a.only:
        names = [r["name"] for r in runs]
        unknown = [n for n in a.only if n not in names]
        if unknown:
            echo(f"--only に設定に無い run: {unknown}（ある run: {names}）")
            return 2
        runs = [r for r in runs if r["name"] in a.only]
    out_root = (a.out_root or ACT_DIR / "out" / "batch" / cfg["name"]).resolve()
    kw = dict(preflight_kwargs or {})
    masking = Masking(out_root)

    if a.dry_run:
        echo(plan_text(cfg, runs, a.device, out_root, DEFAULT_CONTRACT, DEFAULT_CHOREOGRAPHY).rstrip("\n"))
        errs, warns = global_preflight(cfg, a.device, out_root, min_free_gb=a.min_free_gb, **kw)
        echo("事前検査（参考・dry-run では止めない）: " + ("OK" if not errs else "NG: " + "; ".join(masking.line(e) for e in errs)))
        for w in warns:
            echo(f"警告: {w}")
        echo(CLOSED_NOTE)
        return 0

    errs, warns = global_preflight(cfg, a.device, out_root, min_free_gb=a.min_free_gb, **kw)
    if errs:
        for e in errs:
            echo(f"PREFLIGHT NG: {masking.line(e)}")
        return 2
    out_root.mkdir(parents=True, exist_ok=True)
    log = ProgressLog(out_root / "progress.log", masking, echo=echo)
    log.line(f"バッチ {cfg['name']} 開始（run {len(runs)} 本・device {a.device}）")
    for w in warns:
        log.line(f"警告: {w}")
    proc = caffeinate(log)
    try:
        return run_batch(cfg, runs, out_root, a.device, runners or Runners(masking), masking, log)
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
