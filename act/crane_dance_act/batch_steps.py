"""バッチ道具の子プロセス側（学習・評価・書き出しを 1 段ずつ走らせる）。親（batch.py）が出力を全部 LogMasker に通して保存する。

使い方（親が呼ぶ）: python scripts/batch_step.py <train|eval|export> <spec.json>
終了コード: 結果を書けたら 0（学習が失敗しても結果の中の exit_code に残す）、結果を作れなかったら 1（理由は標準エラーに 1 行）。
標準出力の "PROGRESS <step> <loss>" の行を親が拾う（学習の 2000 step ごと）。
"""
import json
import math
import os
import sys
import uuid
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402

from crane_dance_act import evaluate, export, state_mode, train  # noqa: E402

PROGRESS_EVERY = 2000


def jsonable(x):
    """json に書ける形にする（NaN・inf は null・numpy は組み込み型・Path は文字列にせず落とす）。"""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items() if not isinstance(v, Path)}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def write_json_atomic(path, obj):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    try:
        tmp.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def step_train(spec):
    last = {"bucket": -1}

    def on_progress(step, loss):
        bucket = step // PROGRESS_EVERY
        if bucket > last["bucket"]:
            last["bucket"] = bucket
            print(f"PROGRESS {step} {'' if loss is None else loss}", flush=True)

    ts = train.TrainSpec(
        dataset_root=Path(spec["dataset_root"]), repo_id=spec["repo_id"], output_dir=Path(spec["output_dir"]),
        steps=spec["steps"], save_freq=spec["save_freq"], device=spec["device"], batch_size=spec["batch_size"],
        seed=spec["seed"], num_workers=spec["num_workers"], policy=spec["policy"],
        contract_path=Path(spec["contract_path"]), choreography_path=Path(spec["choreography_path"]),
        on_progress=on_progress,
    )  # fmt: skip
    r = train.run_training(ts)
    r["checkpoints"] = [d.parent.name for d in train.find_pretrained_dirs(ts.output_dir)]
    write_json_atomic(spec["result_path"], r)
    return 0


def step_eval(spec):
    es = evaluate.EvalSpec(
        checkpoint=Path(spec["checkpoint"]), columns=spec["columns"], device=spec["device"], count=spec["count"],
        env_variant=spec["env_variant"], contract_path=Path(spec["contract_path"]),
        choreography_path=Path(spec["choreography_path"]),
    )  # fmt: skip
    try:
        r = evaluate.evaluate_checkpoint(es)
    except (evaluate.EvalError, state_mode.StateModeError) as e:
        print(f"評価できない: {e}", file=sys.stderr)
        return 1
    write_json_atomic(spec["result_path"], r)
    return 0


def step_export(spec):
    xs = export.ExportSpec(
        checkpoint=Path(spec["checkpoint"]), out_dir=Path(spec["out_dir"]), mode=spec["mode"], m=spec["m"], tau=spec["tau"],
        formats=tuple(spec["formats"]), device=spec["device"], index=spec["index"], env_variant=spec["env_variant"],
        contract_path=Path(spec["contract_path"]), choreography_path=Path(spec["choreography_path"]),
    )  # fmt: skip
    try:
        export.write_export(xs)
    except (export.ExportError, state_mode.StateModeError) as e:
        print(f"書き出せない: {e}", file=sys.stderr)
        return 1
    return 0


def step_dataset(spec):
    """npz のフォルダ → LeRobotDataset（to_lerobot.convert が一時フォルダ → rename で原子的に作る）。"""
    from crane_dance_act import synth, to_lerobot

    contract = synth.load_contract(spec["contract_path"])
    npz_dir = Path(spec["npz_dir"])
    paths = sorted(npz_dir.glob("ep*.npz"))
    manifest = npz_dir / "manifest.json"
    try:
        to_lerobot.convert(
            paths, contract, spec["choreography_path"], Path(spec["out_root"]), spec["repo_id"],
            manifest if manifest.is_file() else None,
        )  # fmt: skip
    except (ValueError, OSError) as e:
        print(f"変換できない: {e}", file=sys.stderr)
        return 1
    return 0


STEPS = {"train": step_train, "eval": step_eval, "export": step_export, "dataset": step_dataset}


def main(argv):
    if len(argv) != 2 or argv[0] not in STEPS:
        print(f"使い方: batch_step.py <{'|'.join(STEPS)}> <spec.json>", file=sys.stderr)
        return 2
    spec = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    return STEPS[argv[0]](spec)
