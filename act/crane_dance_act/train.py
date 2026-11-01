"""ACT の学習の入口（純関数）。lerobot 0.6.1 の既製 lerobot-train を、検査つきで起動する材料を作る。

引数の書式（--policy.chunk_size=50 など）は draccus で、lerobot/scripts/lerobot_train.py と
configs/train.py・policies/act/configuration_act.py のソースに合わせてある。
"""
import os

# lerobot・huggingface_hub を import する前に固定する。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json  # noqa: E402
import re  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from crane_dance_act import state_mode as state_mode_mod  # noqa: E402
from crane_dance_act.synth import file_sha256, load_contract  # noqa: E402
from crane_dance_act.to_lerobot import STD_FLOOR_RAD  # noqa: E402

# 1 秒。1 小節（約 1.8 秒）より短い。
CHUNK_SIZE = 50
# 推論時だけ使う値。chunk_size 以下でないと ACTConfig が止める（configuration_act.py の検査）ので 50 を明示。
N_ACTION_STEPS = 50
ALLOWED_ENV_KEYS = ("HF_HUB_OFFLINE", "PYTORCH_ENABLE_MPS_FALLBACK")
FLOOR_KEYS = ("observation.state", "action")
LEROBOT_TRAIN = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "lerobot-train"


def _mps_available():
    import torch

    return bool(torch.backends.mps.is_available())


def preflight(dataset_root, contract, choreography_path, device, mps_available=_mps_available):
    """学習前の検査。不一致のメッセージのリストを返す（空なら通る）。"""
    errs = []
    dataset_root = Path(dataset_root)
    if device == "mps":
        if not mps_available():
            errs.append("MPS が使えない（止める。試験は --device cpu）")
    elif device != "cpu":
        errs.append(f"device は mps か cpu: {device}")

    from lerobot.datasets.io_utils import load_stats

    stats = load_stats(dataset_root)
    if stats is None:
        errs.append("meta/stats.json が無い")
    else:
        for key in FLOOR_KEYS:
            std = np.asarray(stats[key]["std"], dtype=np.float64)
            # 有限性を先に見る（NaN は大小比較をすり抜け、+inf は下限を満たしてしまう）
            if not np.all(np.isfinite(std)):
                errs.append(f"{key} の std に有限でない値（NaN・inf）がある")
            elif not np.all(std >= STD_FLOOR_RAD - 1e-12):
                errs.append(f"{key} の std が下限 {STD_FLOOR_RAD} を下回る（最小 {float(std.min()):.6f}）")

    # state_mode（manifest）と stats の照合。定数の入れ忘れ・旧形式の黙った通過を止める
    try:
        sm = state_mode_mod.read_manifest_state_mode(dataset_root)
    except state_mode_mod.StateModeError as e:
        errs.append(f"state_mode: {e}")
    else:
        if stats is not None and "observation.state" in stats:
            st = stats["observation.state"]
            errs += [f"state_mode: {x}" for x in state_mode_mod.check_against_stats(sm, st["mean"], st["std"], STD_FLOOR_RAD)]

    from lerobot.configs.types import NormalizationMode
    from lerobot.policies.act.configuration_act import ACTConfig

    nm = ACTConfig().normalization_mapping
    for k in ("STATE", "ACTION"):
        if nm.get(k) != NormalizationMode.MEAN_STD:
            errs.append(f"ACTConfig の既定の正規化 {k} が MEAN_STD でない: {nm.get(k)}")

    mpath = dataset_root / "manifest.json"
    if not mpath.exists():
        errs.append("データセットに manifest.json が無い")
    else:
        m = json.loads(mpath.read_text(encoding="utf-8"))
        now = file_sha256(choreography_path)
        if m.get("choreography_sha256") != now:
            errs.append("manifest の振付 sha256 が、いまの choreography.json と一致しない")
    return errs


def build_command(dataset_root, repo_id, output_dir, steps, batch_size, seed, save_freq, num_workers, device, log_freq=50, policy=None):
    """policy は resolve_policy の結果（省くと従来どおり chunk 50 だけを渡す）。"""
    chunk = CHUNK_SIZE if policy is None else policy["chunk_size"]
    n_act = N_ACTION_STEPS if policy is None else policy["n_action_steps"]
    cmd = [
        str(LEROBOT_TRAIN),
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root}",
        "--policy.type=act",
        f"--policy.chunk_size={chunk}",
        f"--policy.n_action_steps={n_act}",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
        f"--output_dir={output_dir}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--seed={seed}",
        f"--save_freq={save_freq}",
        f"--num_workers={num_workers}",
        f"--log_freq={log_freq}",
    ]
    if policy is not None:
        cmd += [
            f"--policy.optimizer_lr={policy['lr']!r}",
            f"--policy.optimizer_lr_backbone={policy['lr_backbone']!r}",
            f"--policy.use_vae={'true' if policy['use_vae'] else 'false'}",
            f"--policy.kl_weight={policy['kl_weight']!r}",
            f"--policy.dropout={policy['dropout']!r}",
        ]
    if device == "cpu":
        cmd.append("--policy.device=cpu")
    return cmd


# 学習の上書きの許可リスト（S2）。lerobot 0.6.1 の ACTConfig の既定（configuration_act.py）と同じ値を既定にし、
# 保存された config.json・optimizer_param_groups.json で照合する（既定が食い違えば post_check が止める）。
POLICY_DEFAULTS = {"lr": 1e-5, "lr_backbone": 1e-5, "use_vae": True, "kl_weight": 10.0, "dropout": 0.1, "chunk_size": CHUNK_SIZE}
POLICY_KEYS = ("lr", "lr_backbone", "use_vae", "kl_weight", "dropout", "chunk_size", "n_action_steps")
TRAIN_TOP_KEYS = ("seed", "steps", "batch_size", "save_freq", "num_workers")
ALLOWED_TRAIN_KEYS = POLICY_KEYS + TRAIN_TOP_KEYS


def resolve_policy(overrides=None, min_n_action_steps=1):
    """許可リストの上書きを検査し、既定を埋めた辞書を返す。外れたら ValueError。
    n_action_steps は chunk_size 以下・min_n_action_steps（評価の最大 m）以上。省くと chunk_size。"""
    ov = dict(overrides or {})
    bad = sorted(set(ov) - set(POLICY_KEYS))
    if bad:
        raise ValueError(f"許可リストにない学習の上書き: {bad}（許可: {list(ALLOWED_TRAIN_KEYS)}）")
    out = dict(POLICY_DEFAULTS)
    out.update(ov)

    def num(key, lo, hi=None, lo_open=False):
        v = out[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
            raise ValueError(f"{key} は有限な数: {v!r}")
        too_low = v <= lo if lo_open else v < lo
        if too_low or (hi is not None and v > hi):
            raise ValueError(f"{key} が範囲外: {v!r}")
        out[key] = float(v)

    num("lr", 0.0, lo_open=True)
    num("lr_backbone", 0.0, lo_open=True)
    num("kl_weight", 0.0)
    num("dropout", 0.0, 1.0)
    if not isinstance(out["use_vae"], bool):
        raise ValueError(f"use_vae は真偽値: {out['use_vae']!r}")
    for key in ("chunk_size",):
        if isinstance(out[key], bool) or not isinstance(out[key], int) or out[key] < 1:
            raise ValueError(f"{key} は 1 以上の整数: {out[key]!r}")
    n_act = out.get("n_action_steps", out["chunk_size"])
    if isinstance(n_act, bool) or not isinstance(n_act, int) or n_act < 1:
        raise ValueError(f"n_action_steps は 1 以上の整数: {n_act!r}")
    if n_act > out["chunk_size"]:
        raise ValueError(f"n_action_steps {n_act} が chunk_size {out['chunk_size']} を超える")
    if n_act < min_n_action_steps:
        raise ValueError(f"n_action_steps {n_act} が評価の最大 m {min_n_action_steps} より小さい")
    out["n_action_steps"] = n_act
    return out


def build_env(base_env):
    env = dict(base_env)
    env["HF_HUB_OFFLINE"] = "1"
    return env


# Accelerate が学習の装置を決めるときに読む環境変数（継承すると要求と違う装置で学習しうる）
DEVICE_OVERRIDE_ENV_KEYS = ("ACCELERATE_USE_CPU", "ACCELERATE_TORCH_DEVICE")


def check_device_env(env, device):
    """要求した装置を上書きする環境変数が無いか。不一致のメッセージのリストを返す。
    --device cpu のときは CPU 強制の変数があっても要求と同じなので通す。"""
    errs = []
    use_cpu = str(env.get("ACCELERATE_USE_CPU", "")).strip().lower()
    torch_dev = str(env.get("ACCELERATE_TORCH_DEVICE", "")).strip().lower()
    if device == "mps":
        if use_cpu not in ("", "0", "false", "no"):
            errs.append("環境変数 ACCELERATE_USE_CPU が CPU を強制している（要求は mps）")
        if torch_dev and not torch_dev.startswith("mps"):
            errs.append(f"環境変数 ACCELERATE_TORCH_DEVICE が {torch_dev} を指定している（要求は mps）")
    elif device == "cpu":
        # Accelerate 1.15.0 では ACCELERATE_TORCH_DEVICE が Accelerator(cpu=True) より優先される（実測で mps を返す）
        if torch_dev and not torch_dev.startswith("cpu"):
            errs.append(f"環境変数 ACCELERATE_TORCH_DEVICE が {torch_dev} を指定している（要求は cpu）")
    return errs


def probe_accelerate_device(env, cpu=False):
    """学習と同じ env で Accelerate が選ぶ装置を子プロセスで調べる（lerobot_train.py は accelerator.device を学習に使う）。
    cpu=True は --policy.device=cpu のときの lerobot と同じ Accelerator(cpu=True)（lerobot_train.py:245 の force_cpu）。
    読めなければ "unknown"。"""
    import subprocess
    import sys

    code = f"from accelerate import Accelerator; print(Accelerator(cpu={bool(cpu)}).device.type)"
    try:
        r = subprocess.run([sys.executable, "-c", code], env=dict(env), capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    out = r.stdout.strip().splitlines()
    return out[-1].strip() if r.returncode == 0 and out else "unknown"


def resolve_actual_device(requested, probed, logged):
    """(実際の装置, 不一致のメッセージのリスト)。probe と学習ログの読み取りを照合する。
    要求と違う装置が 1 つでも示されたら不一致。どちらも読めなければ unknown で不一致。"""
    errs = []
    seen = {d for d in (probed, logged) if d and d != "unknown"}
    if not seen:
        return "unknown", [f"学習の装置を確かめられない（要求は {requested}）"]
    if len(seen) > 1:
        errs.append(f"Accelerate の装置 {probed} と学習ログの装置 {logged} が食い違う")
    actual = probed if probed != "unknown" else logged
    if actual != requested:
        errs.append(f"学習の装置 {actual} が要求 {requested} と違う")
    return actual, errs


# 保存するログの伏せ字（秘密 → 実パス → ダッシュ形 → 利用者名 の順）
_VALUE = r"""("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\S+)"""
_REDACTIONS = (
    (re.compile(r"(?i)\b([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)[A-Z0-9_]*)\s*=\s*" + _VALUE), r"\1=<redacted>"),
    (re.compile(r"(?i)bearer\s+" + _VALUE), "bearer <redacted>"),
    (re.compile(r"(?i)(--token|--password)\s+" + _VALUE), r"\1 <redacted>"),
)
_SENTINELS = (("<user>", "\x00U\x00"), ("<redacted>", "\x00R\x00"))


_OPEN_QUOTED_SECRET = re.compile(
    r"(?i)(?:\b[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)[A-Z0-9_]*\s*=\s*|bearer\s+|--token\s+|--password\s+)([\"'])"
)


def _has_unclosed_secret_quote(text):
    """秘密の値が引用符で始まり、まだ閉じていないか（\\ のエスケープを考える）。閉じていなければその開始位置、閉じていれば -1。"""
    pos = 0
    while True:
        m = _OPEN_QUOTED_SECRET.search(text, pos)
        if not m:
            return -1
        q = m.group(1)
        i = m.end()
        while i < len(text):
            if text[i] == "\\":
                i += 2
                continue
            if text[i] == q:
                break
            i += 1
        if i >= len(text):
            return m.start(1)
        pos = i + 1


class LogMasker:
    """子プロセスの出力を 1 行ずつ受け取り、伏せてから返す。引用符つきの秘密値が改行をまたぐ間は行をためて、
    値全体がそろってから伏せる（物理行ごとに伏せると後半が残る）。ためすぎ（MAX_HOLD）や最後まで
    閉じない引用は、開始の引用符より後ろを全部伏せる。"""

    MAX_HOLD = 64 * 1024

    def __init__(self, home=None, named_paths=()):
        self.home = home
        self.named_paths = named_paths
        self.pending = ""
        self.open_quote = None  # ためすぎで先に伏せた引用がまだ閉じていない間の引用符

    def _mask(self, text):
        return mask_log(text, home=self.home, named_paths=self.named_paths)

    def _cut_open(self, text, start):
        self.open_quote = text[start]
        return self._mask(text[:start]) + "<redacted>\n"

    def feed(self, line):
        if self.open_quote is not None:
            # 先に伏せた引用の続き：閉じる引用符までは捨てる（伏せ済み）、その後ろは通常どおり
            i = 0
            while i < len(line):
                if line[i] == "\\":
                    i += 2
                    continue
                if line[i] == self.open_quote:
                    break
                i += 1
            if i >= len(line):
                return ""
            self.open_quote = None
            line = line[i + 1 :]
            if not line.strip():
                return ""
        self.pending += line
        start = _has_unclosed_secret_quote(self.pending)
        if start < 0:
            out, self.pending = self._mask(self.pending), ""
            return out
        if len(self.pending) > self.MAX_HOLD:
            out, self.pending = self._cut_open(self.pending, start), ""
            return out
        return ""

    def close(self):
        if not self.pending:
            return ""
        start = _has_unclosed_secret_quote(self.pending)
        out = self._cut_open(self.pending, start) if start >= 0 else self._mask(self.pending)
        self.pending = ""
        self.open_quote = None
        return out


def mask_log(text, home=None, named_paths=()):
    """学習ログを保存する前に伏せる。named_paths は (実パス, 置き換えの名前) の組（データセットと出力先）。
    何度通しても結果は同じ。loss・updt_s・装置の表示は変えない。"""
    for pat, rep in _REDACTIONS:
        text = pat.sub(rep, text)
    for real, name in sorted(named_paths, key=lambda p: -len(str(p[0]))):
        real = str(real)
        if real and real != "/":
            text = text.replace(real, name)
    home = os.path.expanduser("~") if home is None else home
    if home and home != "/":
        text = text.replace(home, "~")
        text = text.replace(home.replace("/", "-"), "~")
        user = os.path.basename(home.rstrip("/"))
        if len(user) >= 3:
            for real, tok in _SENTINELS:
                text = text.replace(real, tok)
            text = text.replace(user, "<user>")
            for real, tok in _SENTINELS:
                text = text.replace(tok, real)
    return text


def detect_device(log_text):
    """学習ログから装置を読む。読めなければ "unknown"（推測しない）。

    lerobot-train は実機の装置を 1 行では出さない。ログで読める根拠は次の 2 つだけ。
    ① configs/policies.py の警告 "Device 'X' is not available. Switching to 'Y'"（Y が自動選択された装置。
       学習の装置は Accelerate が自動で決める＝lerobot_train.py:243-245）
    ② 設定の表示が device: cpu（CPU 指定のときだけ Accelerate が従う＝同:245）
    """
    m = re.search(r"Switching to '(mps|cpu|cuda)'", log_text)
    if m:
        return m.group(1)
    if re.search(r"'device':\s*'cpu'", log_text):
        return "cpu"
    return "unknown"


def parse_log(log_text):
    """(最初の loss, 最後の loss, 1 ステップあたり秒(updt_s+data_s の平均)) をログから読む。読めなければ None。"""
    losses = [float(x) for x in re.findall(r"\bloss:(-?[0-9.]+(?:e-?\d+)?)", log_text)]
    upd = [float(x) for x in re.findall(r"\bupdt_s:([0-9.]+)", log_text)]
    dat = [float(x) for x in re.findall(r"\bdata_s:([0-9.]+)", log_text)]
    sec = None
    if upd:
        sec = float(np.mean(upd)) + (float(np.mean(dat)) if dat else 0.0)
    return (losses[0] if losses else None, losses[-1] if losses else None, sec)


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def run_record(cmd, seed, choreography_sha256, dataset_root, repo_id, device, actual_device, env, steps, batch_size):
    """run.json の辞書。実パスは書かない（フォルダ名だけ）。env は ALLOWED_ENV_KEYS の鍵だけ。"""

    def scrub(a):
        # --key=/abs/path/name → --key=name（実パスを残さない）
        k, sep, v = a.partition("=")
        if sep and v.startswith("/"):
            return f"{k}={Path(v).name}"
        return Path(a).name if a.startswith("/") else a

    return {
        "command": [scrub(a) for a in cmd],
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "choreography_sha256": choreography_sha256,
        "dataset_repo_id": repo_id,
        "dataset_root_name": Path(dataset_root).name,
        "lerobot_version": _version("lerobot"),
        "torch_version": _version("torch"),
        "device_requested": device,
        "device_actual": actual_device,
        "env": {k: env[k] for k in ALLOWED_ENV_KEYS if k in env},
        "note": "seed は記録するが MPS でのビット単位の再現は主張しない",
    }


def find_pretrained_dirs(output_dir):
    root = Path(output_dir) / "checkpoints"
    if not root.exists():
        return []
    return sorted(p / "pretrained_model" for p in root.iterdir() if p.is_dir() and not p.is_symlink() and (p / "pretrained_model").is_dir())


def _lrs_from_param_groups(path):
    groups = json.loads(Path(path).read_text(encoding="utf-8"))
    return [g.get("lr") for g in groups]


def post_check(output_dir, expected=None):
    """保存された config.json の正規化方式と chunk_size を確かめる。不一致のリストを返す。
    expected（resolve_policy の結果＋任意の "state_mode"）を渡すと、config.json の chunk_size・n_action_steps・use_vae・
    kl_weight・dropout と、training_state/optimizer_param_groups.json の lr、チェックポイントの state_mode.json も照合する。"""
    errs = []
    dirs = find_pretrained_dirs(output_dir)
    if not dirs:
        return ["チェックポイント（pretrained_model）が無い"]
    want_chunk = CHUNK_SIZE if expected is None else expected["chunk_size"]
    for d in dirs:
        name = d.parent.name
        cp = d / "config.json"
        if not cp.exists():
            errs.append(f"{name}: config.json が無い")
            continue
        cfg = json.loads(cp.read_text(encoding="utf-8"))
        nm = cfg.get("normalization_mapping", {})
        for k in ("STATE", "ACTION"):
            if nm.get(k) != "MEAN_STD":
                errs.append(f"{name}: normalization_mapping.{k} が MEAN_STD でない: {nm.get(k)}")
        if cfg.get("chunk_size") != want_chunk:
            errs.append(f"{name}: chunk_size が {want_chunk} でない: {cfg.get('chunk_size')}")
        if expected is None:
            continue
        for key in ("n_action_steps", "use_vae"):
            if cfg.get(key) != expected[key]:
                errs.append(f"{name}: {key} が {expected[key]!r} でない: {cfg.get(key)!r}")
        for key in ("kl_weight", "dropout"):
            got = cfg.get(key)
            if not isinstance(got, (int, float)) or abs(got - expected[key]) > 1e-9 * max(1.0, abs(expected[key])):
                errs.append(f"{name}: {key} が {expected[key]!r} でない: {got!r}")
        pg = d.parent / "training_state" / "optimizer_param_groups.json"
        if not pg.exists():
            errs.append(f"{name}: training_state/optimizer_param_groups.json が無い（lr を照合できない）")
        else:
            lrs = _lrs_from_param_groups(pg)
            want_lrs = {expected["lr"], expected["lr_backbone"]}
            if not lrs or any(not isinstance(x, (int, float)) or not any(abs(x - w) <= 1e-12 + 1e-9 * abs(w) for w in want_lrs) for x in lrs):
                errs.append(f"{name}: optimizer_param_groups の lr {lrs} が期待 {sorted(want_lrs)} と合わない")
            elif not any(abs(x - expected["lr"]) <= 1e-12 + 1e-9 * expected["lr"] for x in lrs):
                errs.append(f"{name}: optimizer_param_groups に lr={expected['lr']!r} の群が無い: {lrs}")
        if expected.get("state_mode") is not None:
            try:
                got_sm = state_mode_mod.read_checkpoint(d)
            except state_mode_mod.StateModeError as e:
                errs.append(f"{name}: {e}")
            else:
                if got_sm != expected["state_mode"]:
                    errs.append(f"{name}: state_mode.json が学習時の state_mode と違う")
    return errs


# ---------------------------------------------------------------- 学習の中核（S1）
_PACT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = _PACT.parent / "contract" / "contract.json"
DEFAULT_CHOREOGRAPHY = _PACT.parent / "contract" / "choreography.json"
_STEP_RE = re.compile(r"\bstep:(\d+)(K?)\s")
_LOSS_RE = re.compile(r"\bloss:(-?[0-9.]+(?:e-?\d+)?)")


@dataclass
class TrainSpec:
    dataset_root: Path
    repo_id: str
    output_dir: Path
    steps: int
    save_freq: int
    device: str = "mps"
    batch_size: int = 8
    seed: int = 1000
    num_workers: int = 0
    policy: dict = field(default_factory=dict)  # resolve_policy を通した辞書（省くと既定）
    contract_path: Path = DEFAULT_CONTRACT
    choreography_path: Path = DEFAULT_CHOREOGRAPHY
    on_progress: object = None  # (step:int, loss:float|None) を受ける関数。伏せたログの行から呼ぶ


STOP_WAIT_SEC = 5
# 一番外側の run_streamed が子を新しいプロセスグループで起動したことを、子と孫に知らせる目印。
# これが立っているプロセスの中の run_streamed は、新しいグループを作らない（孫が外側のグループから逃げないように）。
PGROUP_ENV = "CRANE_DANCE_PGROUP_MANAGED"


def _group_alive(pgid, proc):
    proc.poll()  # 直接の子を回収しておく（ゾンビが残るとグループが生きて見える）
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _stop_group(pgid, proc):
    """プロセスグループ全体（子・孫・その先）を SIGTERM → 待つ → SIGKILL → 待つ。"""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        end = time.monotonic() + STOP_WAIT_SEC
        while time.monotonic() < end:
            if not _group_alive(pgid, proc):
                return
            time.sleep(0.05)


def _stop_child(proc, group=False):
    """子がまだ生きていれば terminate → 待つ → kill。停止処理の失敗は握りつぶす（元の例外を隠さない）。
    group=True（子を新しいプロセスグループで起動したとき）は、先にグループ全体を止めて孫まで回収する。"""
    if group:
        try:
            _stop_group(proc.pid, proc)
        except BaseException:  # noqa: BLE001  停止の失敗で元の例外を上書きしない
            pass
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=STOP_WAIT_SEC)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    except BaseException:  # noqa: BLE001  停止の失敗で元の例外を上書きしない
        try:
            proc.kill()
            proc.wait(timeout=STOP_WAIT_SEC)
        except BaseException:  # noqa: BLE001
            pass
    try:
        if proc.stdout is not None:
            proc.stdout.close()
    except BaseException:  # noqa: BLE001
        pass


def run_streamed(argv, log_path, masker, on_masked=None, env=None):
    """子を走らせ、出力を masker で伏せて log_path に保存し、終了コードを返す。
    起動した子の所有は、ログ・コールバックの成否から独立させる（Popen 直後から try/finally で覆い、
    例外・割込みでも子の終了を確認してから投げ直す）。例外の経路では masker の未確定の行は書かない。
    一番外側（PGROUP_ENV が立っていないプロセス）では子を新しいプロセスグループで起動し、終わるとき・止めるときは
    グループごと止める（バッチ → batch_step → lerobot-train の孫まで残さない）。内側では新しいグループを作らない。"""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    own_group = os.environ.get(PGROUP_ENV) != "1"
    child_env = dict(os.environ if env is None else env)
    child_env[PGROUP_ENV] = "1"
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(argv, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=own_group)
        try:

            def handle(masked):
                lf.write(masked)
                lf.flush()
                if on_masked is not None:
                    on_masked(masked)

            for raw in proc.stdout:
                handle(masker.feed(raw.decode("utf-8", errors="replace")))
            rc = proc.wait()
            handle(masker.close())
            return rc
        finally:
            _stop_child(proc, group=own_group)


def run_training(spec):
    """検査 → lerobot-train（子プロセス・出力は LogMasker を通して保存）→ 保存物の検査 → run.json。
    scripts/train_act.py・バッチから同じものを呼ぶ。結果の辞書を返す（"exit_code": 2 は学習前の検査で止めた）。
    出力先が既にあれば止める（上書き・削除しない）。"""
    out = Path(spec.output_dir).resolve()
    if out.exists():
        return {"exit_code": 2, "preflight_errors": [f"出力先が既にある: {out.name}"]}
    contract = load_contract(spec.contract_path)
    policy = resolve_policy(spec.policy)
    env = build_env(os.environ)
    errs = preflight(spec.dataset_root, contract, spec.choreography_path, spec.device)
    errs += check_device_env(env, spec.device)
    # 学習と同じ env で Accelerate が選ぶ装置を調べる。cpu は lerobot と同じ Accelerator(cpu=True)
    probed = probe_accelerate_device(env, cpu=(spec.device == "cpu")) if not errs else "unknown"
    if not errs and probed != "unknown" and probed != spec.device:
        errs.append(f"学習と同じ環境で Accelerate が選ぶ装置が {probed}（要求は {spec.device}）")
    if errs:
        return {"exit_code": 2, "preflight_errors": errs}

    root = Path(spec.dataset_root).resolve()
    sm = state_mode_mod.read_manifest_state_mode(root)
    cmd = build_command(root, spec.repo_id, out, spec.steps, spec.batch_size, spec.seed, spec.save_freq, spec.num_workers, spec.device, policy=policy)
    log_path = out.parent / f"{out.name}.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    masker = LogMasker(named_paths=((root, f"<dataset_root:{root.name}>"), (out, f"<output_dir:{out.name}>")))

    def emit(text):
        if spec.on_progress is None:
            return
        for line in text.splitlines():
            for chunk in line.replace("\r", "\n").splitlines():
                ms = _STEP_RE.search(chunk)
                if ms:
                    ml = _LOSS_RE.search(chunk)
                    spec.on_progress(int(ms.group(1)) * (1000 if ms.group(2) else 1), float(ml.group(1)) if ml else None)

    t0 = time.perf_counter()
    rc = run_streamed(cmd, log_path, masker, on_masked=emit, env=env)
    total = time.perf_counter() - t0
    text = log_path.read_text(encoding="utf-8", errors="replace")
    actual, dev_errs = resolve_actual_device(spec.device, probed, detect_device(text))
    first, last, sec = parse_log(text)

    if rc == 0:
        for d in find_pretrained_dirs(out):
            state_mode_mod.write_atomic(state_mode_mod.checkpoint_file(d), sm)
    expected = dict(policy, state_mode=sm)
    post = (post_check(out, expected) if rc == 0 else ["学習が失敗したので検査せず"]) + dev_errs
    post = [mask_log(e, named_paths=masker.named_paths) for e in post]  # run.json に実パスを残さない
    rec = run_record(cmd, spec.seed, file_sha256(spec.choreography_path), spec.dataset_root, spec.repo_id, spec.device, actual, env, spec.steps, spec.batch_size)
    rec.update({"exit_code": rc, "post_check_errors": post, "state_mode": sm, "policy": policy, "total_sec": round(total, 1)})
    run_json = out / "run.json"
    if out.exists():
        run_json.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "exit_code": rc, "device_actual": actual, "loss_first": first, "loss_last": last,
        "sec_per_step": sec if sec is not None else total / spec.steps,
        "sec_per_step_from_log": sec is not None, "sec_per_step_wall": total / spec.steps,
        "total_sec": total, "post_check_errors": post, "preflight_errors": [],
        "log_path": log_path, "run_json": run_json, "state_mode": sm, "policy": policy,
    }
