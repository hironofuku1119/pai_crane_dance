"""ACT のチェックポイントを検証データで評価する（開ループ＋合成の閉ループ・シミュレータなし）。cwd に依存しない。
中身は crane_dance_act.evaluate.evaluate_checkpoint（バッチ道具と同じもの）。

閉ループは Gazebo の代わりではなく、synth.py の一次遅れの追従モデルを「仮の機械」にしたもの。
"""
import os

# lerobot・huggingface_hub を import する前に固定する（値は import 時に 1 回だけ読まれる）。
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

ACT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACT_DIR))

from crane_dance_act import evaluate, state_mode, variants  # noqa: E402


def _clean(x):
    """json に書ける形にする（NaN は null・numpy は組み込み型・水準のキーは文字列）。"""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", type=pathlib.Path, default=ACT_DIR.parent / "contract" / "contract.json")
    ap.add_argument("--choreography", type=pathlib.Path, default=ACT_DIR.parent / "contract" / "choreography.json")
    ap.add_argument("--checkpoint", type=pathlib.Path, required=True, help="pretrained_model のフォルダ")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=20, help="検証データの先頭の番号（学習は 0〜19）")
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--m", default="1,10", help="方策の呼び出し間隔（フレーム）。カンマ区切り")
    ap.add_argument("--taus", default="0.02,0.06,0.10,0.15", help="閉ループの時定数[s]。カンマ区切り")
    ap.add_argument("--env-variant", choices=variants.KINDS, default="full")
    ap.add_argument("--device", choices=("mps", "cpu"), default="mps")
    ap.add_argument("--assume-legacy-lagged", action="store_true", help="state_mode.json が無い旧形式のチェックポイントを lagged として読む（明示したときだけ）")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()

    if a.out.exists():
        print(f"出力先が既にある: {a.out.name}", file=sys.stderr)
        return 1
    ms = [int(x) for x in a.m.split(",")]
    taus = [float(x) for x in a.taus.split(",")]
    columns = []
    for m in ms:
        columns.append({"kind": "open", "m": m})
        columns += [{"kind": "closed", "m": m, "tau": tau} for tau in taus]
    spec = evaluate.EvalSpec(
        checkpoint=a.checkpoint, columns=columns, device=a.device, count=a.count, start=a.start, seed=a.seed,
        env_variant=a.env_variant, contract_path=a.contract, choreography_path=a.choreography,
        assume_legacy_lagged=a.assume_legacy_lagged,
    )  # fmt: skip
    try:
        r = evaluate.evaluate_checkpoint(spec)
    except (evaluate.EvalError, state_mode.StateModeError) as e:
        print(str(e), file=sys.stderr)
        return 1

    result = {
        "checkpoint": r["checkpoint"],
        "choreography_sha256": r["choreography_sha256"],
        "contract_version": r["contract_version"],
        "seed": a.seed,
        "episodes": r["episodes"],
        "env_variant": a.env_variant,
        "device": a.device,
        "state_mode": r["state_mode"],
        "m_list": ms,
        "tau_list_s": taus,
        "note": r["note"],
        "open_loop": {},
        "closed_loop": {},
    }
    for m in ms:
        c = r["columns"][evaluate.column_name({"kind": "open", "m": m})]
        result["open_loop"][f"m{m}"] = {"episodes": c["episodes"], "all_pass": c["all_pass"]}
        for tau in taus:
            c = r["columns"][evaluate.column_name({"kind": "closed", "m": m, "tau": tau})]
            result["closed_loop"][f"m{m}_tau{tau}"] = {"episodes": c["episodes"], "all_pass": c["all_pass"]}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "x", encoding="utf-8") as f:
        f.write(json.dumps(_clean(result), ensure_ascii=False, indent=2) + "\n")

    print(f"checkpoint: {result['checkpoint']}  variant: {a.env_variant}  device: {a.device}")
    for kind in ("open_loop", "closed_loop"):
        for name, res in result[kind].items():
            jmax = max(e["judge"]["keyframe_max_judge_rad"] for e in res["episodes"])
            tmax = max(
                (e["judge"]["timing_max_abs_dt_beats"] for e in res["episodes"] if not math.isnan(e["judge"]["timing_max_abs_dt_beats"])),
                default=math.nan,
            )
            un = sum(e["judge"]["unreached_at_0.5"] for e in res["episodes"])
            print(f"{kind} {name}: keyframe_max={jmax:.4f} rad  timing_max={tmax:.3f} beats  unreached={un}  pass={res['all_pass']}")
    print(f"out: {a.out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
