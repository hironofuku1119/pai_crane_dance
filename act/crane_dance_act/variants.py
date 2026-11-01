"""環境状態の一部を定数にした変種（progress だけで覚えていないかの診断）。学習と評価で同じ関数を使う。

データの形（N, 3）は変えない（契約は変わらない）。列の並びは sin_phase, cos_phase, progress。
- full: そのまま（コピー）
- no_phase: sin・cos の列を 0・1 に固定（拍位相の情報を抜く）
- no_progress: progress の列を 0 に固定（曲の進行度の情報を抜く）
"""
import json
import os
import shutil
import uuid
from pathlib import Path

import numpy as np

from crane_dance_act import synth

KINDS = ("full", "no_phase", "no_progress")


def apply_env_variant(env, kind):
    """環境状態 (N, 3) の変種のコピーを返す。元の配列は変えない。"""
    if kind not in KINDS:
        raise ValueError(f"未知の変種: {kind}（{', '.join(KINDS)} のどれか）")
    out = np.array(env, copy=True)
    if out.ndim != 2 or out.shape[1] != 3:
        raise ValueError(f"環境状態の形が (N, 3) でない: {out.shape}")
    if kind == "no_phase":
        out[:, 0] = 0.0
        out[:, 1] = 1.0
    elif kind == "no_progress":
        out[:, 2] = 0.0
    return out


def make_variant_set(in_dir, kind, out_dir, contract):
    """in_dir の ep*.npz の observation_environment_state だけを変えて out_dir に書く。
    manifest.json を写して "env_variant" を足す。out_dir が既にあれば止まる。"""
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    if kind not in KINDS:
        raise ValueError(f"未知の変種: {kind}（{', '.join(KINDS)} のどれか）")
    if out_dir.exists():
        raise FileExistsError(f"出力先が既にある: {out_dir.name}")
    manifest_path = in_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("入力に manifest.json が無い")
    paths = sorted(in_dir.glob("ep*.npz"))
    if not paths:
        raise ValueError("入力に ep*.npz が 0 本")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tmp = out_dir.parent / f".{out_dir.name}.tmp-{uuid.uuid4().hex[:8]}"
    try:
        tmp.mkdir(parents=True)
        for p in paths:
            with np.load(p, allow_pickle=False) as z:
                ep = {k: z[k] for k in z.files}
            ep["observation_environment_state"] = apply_env_variant(ep["observation_environment_state"], kind)
            synth.save_npz(tmp / p.name, ep, contract)
        manifest["env_variant"] = kind
        (tmp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.rename(tmp, out_dir)
    except BaseException:
        if tmp.exists():
            shutil.rmtree(tmp)
        raise
    return manifest
