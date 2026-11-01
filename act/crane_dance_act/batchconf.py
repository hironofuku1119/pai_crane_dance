"""バッチ道具の設定（TOML・stdlib の tomllib）。未知の鍵は誤り・型と範囲を検査し、既定値を埋めた解決済みの設定と sha256 を作る。

    [[run]]
    name = "b1"                      # 英小文字・数字・_・-（先頭は英小文字か数字）
    description = "自由記述"
    [run.data]    seed / episodes / sigma_rad / clip_rad / tau_range_s / state_mode / env_variant
    [run.train]   steps（必須）/ batch_size / seed / save_freq / num_workers / lr / lr_backbone / use_vae / kl_weight / dropout /
                  chunk_size / n_action_steps
    [run.eval]    ckpts / count / columns（[[run.eval.columns]] に kind = open|closed|ensemble, m, tau）
    [run.export]  ckpts / modes / formats / m / tau / index
"""
import hashlib
import json
import math
import re
import tomllib
from pathlib import Path

from crane_dance_act import evaluate, train, variants

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
STATE_MODES = ("lagged", "constant")
EXPORT_MODES = ("open", "closed")
EXPORT_FORMATS = ("npz", "yaml")
MAX_TRAIN_EPISODES = 20  # 検証データは ep020〜（学習は ep000〜ep019 まで）
DEFAULT_COLUMNS = (
    {"kind": "open", "m": 10},
    {"kind": "closed", "m": 10, "tau": 0.06},
    {"kind": "ensemble", "m": 1, "tau": 0.06},
)
DATA_DEFAULTS = {
    "seed": 0, "episodes": 20, "sigma_rad": 0.0175, "clip_rad": 0.035, "tau_range_s": [0.02, 0.10],
    "state_mode": "lagged", "env_variant": "full",
}


class ConfigError(ValueError):
    """設定の誤り（未知の鍵・型・範囲）。"""


def _unknown(table, allowed, where):
    extra = sorted(set(table) - set(allowed))
    if extra:
        raise ConfigError(f"{where}: 未知の鍵 {extra}（許可: {sorted(allowed)}）")


def _int(table, key, default, lo, hi, where, required=False):
    if key not in table:
        if required:
            raise ConfigError(f"{where}: {key} は必須")
        return default
    v = table[key]
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise ConfigError(f"{where}.{key}: {lo}〜{hi} の整数: {v!r}")
    return v


def _float(table, key, default, lo, hi, where):
    if key not in table:
        return default
    v = table[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
        raise ConfigError(f"{where}.{key}: {lo}〜{hi} の数: {v!r}")
    return float(v)


def _choice(table, key, default, choices, where):
    v = table.get(key, default)
    if v not in choices:
        raise ConfigError(f"{where}.{key}: {list(choices)} のどれか: {v!r}")
    return v


def _int_list(table, key, default, where, lo=1, hi=10**8):
    v = table.get(key, default)
    if not isinstance(v, list) or not v:
        raise ConfigError(f"{where}.{key}: 空でない整数のリスト: {v!r}")
    for x in v:
        if isinstance(x, bool) or not isinstance(x, int) or not lo <= x <= hi:
            raise ConfigError(f"{where}.{key}: {lo}〜{hi} の整数のリスト: {v!r}")
    if len(set(v)) != len(v):
        raise ConfigError(f"{where}.{key}: 重複がある: {v!r}")
    return sorted(v)


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _resolve_data(t, where):
    _unknown(t, DATA_DEFAULTS, where)
    d = {
        "seed": _int(t, "seed", 0, 0, 2**31 - 1, where),
        "episodes": _int(t, "episodes", 20, 1, MAX_TRAIN_EPISODES, where),
        "sigma_rad": _float(t, "sigma_rad", 0.0175, 0.0, 1.0, where),
        "clip_rad": _float(t, "clip_rad", 0.035, 0.0, 1.0, where),
        "state_mode": _choice(t, "state_mode", "lagged", STATE_MODES, where),
        "env_variant": _choice(t, "env_variant", "full", variants.KINDS, where),
    }
    tr = t.get("tau_range_s", DATA_DEFAULTS["tau_range_s"])
    if (
        not isinstance(tr, list) or len(tr) != 2
        or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in tr)
        or not 0 < tr[0] <= tr[1] <= 1.0
    ):
        raise ConfigError(f"{where}.tau_range_s: 0 < 下限 ≤ 上限 ≤ 1 の 2 要素: {tr!r}")
    d["tau_range_s"] = [float(tr[0]), float(tr[1])]
    return d


def _resolve_columns(raw, where):
    if raw is None:
        raw = [dict(c) for c in DEFAULT_COLUMNS]
    if not isinstance(raw, list):
        raise ConfigError(f"{where}.columns: 表の配列")
    try:
        return evaluate.check_columns(raw)
    except ValueError as e:
        raise ConfigError(f"{where}.columns: {e}") from e


def _resolve_run(r, index):
    where = f"run[{index}]"
    if not isinstance(r, dict):
        raise ConfigError(f"{where}: 表でない")
    _unknown(r, ("name", "description", "data", "train", "eval", "export"), where)
    name = r.get("name")
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ConfigError(f"{where}.name: 英小文字・数字・_・- で 1〜40 文字（先頭は英小文字か数字）: {name!r}")
    where = f"run.{name}"
    desc = r.get("description", "")
    if not isinstance(desc, str) or len(desc) > 500:
        raise ConfigError(f"{where}.description: 500 文字以内の文字列")

    data = _resolve_data(dict(r.get("data", {})), f"{where}.data")

    t = dict(r.get("train", {}))
    _unknown(t, train.ALLOWED_TRAIN_KEYS, f"{where}.train")
    tw = f"{where}.train"
    steps = _int(t, "steps", None, 1, 10**7, tw, required=True)
    top = {
        "steps": steps,
        "batch_size": _int(t, "batch_size", 8, 1, 4096, tw),
        "seed": _int(t, "seed", 1000, 0, 2**31 - 1, tw),
        "save_freq": _int(t, "save_freq", 1000, 1, 10**7, tw),
        "num_workers": _int(t, "num_workers", 0, 0, 16, tw),
    }

    e = dict(r.get("eval", {}))
    ew = f"{where}.eval"
    _unknown(e, ("ckpts", "count", "columns"), ew)
    columns = _resolve_columns(e.get("columns"), ew)
    ev = {
        "ckpts": _int_list(e, "ckpts", [steps], ew),
        "count": _int(e, "count", 4, 1, 20, ew),
        "columns": columns,
    }

    ex = None
    if "export" in r:
        xt = dict(r["export"])
        xw = f"{where}.export"
        _unknown(xt, ("ckpts", "modes", "formats", "m", "tau", "index"), xw)
        modes = xt.get("modes", ["open"])
        formats = xt.get("formats", list(EXPORT_FORMATS))
        for key, val, ok in (("modes", modes, EXPORT_MODES), ("formats", formats, EXPORT_FORMATS)):
            if not isinstance(val, list) or not val or any(x not in ok for x in val) or len(set(val)) != len(val):
                raise ConfigError(f"{xw}.{key}: {list(ok)} から選んだ重複のないリスト: {val!r}")
        ex = {
            "ckpts": _int_list(xt, "ckpts", [steps], xw),
            "modes": sorted(modes),
            "formats": sorted(formats),
            "m": _int(xt, "m", 10, 1, 1000, xw),
            "tau": _float(xt, "tau", 0.06, 1e-6, 10.0, xw),
            "index": _int(xt, "index", 20, 20, 10**6, xw),
        }

    # 評価・書き出しの ckpt は保存される step（save_freq の倍数か最後）で、steps 以下
    for where_, steps_list in ((ew, ev["ckpts"]), (f"{where}.export", ex["ckpts"] if ex else [])):
        for c in steps_list:
            if c > steps or not (c % top["save_freq"] == 0 or c == steps):
                raise ConfigError(f"{where_}.ckpts: {c} は保存される step（save_freq={top['save_freq']} の倍数か steps={steps}）ではない")
    max_m = max([c["m"] for c in columns] + ([ex["m"]] if ex else []))
    pol_over = {k: v for k, v in t.items() if k in train.POLICY_KEYS}
    try:
        policy = train.resolve_policy(pol_over, min_n_action_steps=max_m)
    except ValueError as err:
        raise ConfigError(f"{tw}: {err}") from err
    return {"name": name, "description": desc, "data": data, "train": {**top, "policy": policy}, "eval": ev, "export": ex}


def resolve_config(raw, config_name):
    """TOML を読んだ辞書 → 解決済みの設定。{"name", "runs": [...]}（各 run に "config_sha256"）。"""
    if not isinstance(raw, dict):
        raise ConfigError("設定が表でない")
    _unknown(raw, ("batch", "run"), "トップ")
    batch = raw.get("batch", {})
    _unknown(batch, ("description",), "batch")
    runs_raw = raw.get("run")
    if not isinstance(runs_raw, list) or not runs_raw:
        raise ConfigError("[[run]] が 1 つも無い")
    runs = [_resolve_run(r, i) for i, r in enumerate(runs_raw)]
    names = [r["name"] for r in runs]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise ConfigError(f"run の名前が重複: {dup}")
    for r in runs:
        r["config_sha256"] = _sha({k: v for k, v in r.items() if k != "description"})
    return {"name": config_name, "description": str(batch.get("description", "")), "runs": runs}


def load_config(path):
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(f"設定を読めない: {path.name}: {e}") from e
    if not re.match(r"^[A-Za-z0-9_.-]+$", path.stem):
        raise ConfigError(f"設定のファイル名に使えない文字がある: {path.stem!r}")
    return resolve_config(raw, path.stem)


# ---------------------------------------------------------------- 段階ごとのハッシュ（S5・S6）
def data_hash(run, choreography_sha256, contract_version):
    """データ（npz とデータセット）の名前とハッシュ。データの部分の設定＋振付と契約の版だけで決まる。"""
    return _sha({"data": run["data"], "choreography_sha256": choreography_sha256, "contract_version": contract_version, "gen": 1})


def stage_hashes(run, data_h, device):
    """段階ごとのハッシュ。後の段階は前の段階のハッシュを含む（上流が変われば下流も変わる）。"""
    train_h = _sha({"data": data_h, "train": run["train"], "device": device})
    eval_h = _sha({"train": train_h, "eval": run["eval"]})
    out = {"data": data_h, "dataset": data_h, "train": train_h, "eval": eval_h}
    if run["export"] is not None:
        out["export"] = _sha({"train": train_h, "export": run["export"]})
    return out
