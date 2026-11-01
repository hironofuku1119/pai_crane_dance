"""lerobot-train を検査つきで起動する。cwd に依存しない。中身は crane_dance_act.train.run_training（バッチ道具と同じもの）。"""
import argparse
import pathlib
import sys

ACT_DIR = pathlib.Path(__file__).resolve().parents[1]
# スクリプト実行では sys.path[0] がスクリプトのフォルダになる（Python 3.12 docs, sys_path_init）ので act/ を足す。
sys.path.insert(0, str(ACT_DIR))

from crane_dance_act import train  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=pathlib.Path, default=ACT_DIR / "out" / "lerobot" / "crane_dance_synth_v1")
    ap.add_argument("--repo-id", default="local/crane_dance_synth_v1")
    ap.add_argument("--output-dir", type=pathlib.Path, required=True, help="out/train/<run>")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--save-freq", type=int, required=True)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", choices=["mps", "cpu"], default="mps")
    ap.add_argument("--contract", type=pathlib.Path, default=ACT_DIR.parent / "contract" / "contract.json")
    ap.add_argument("--choreography", type=pathlib.Path, default=ACT_DIR.parent / "contract" / "choreography.json")
    a = ap.parse_args(argv)

    spec = train.TrainSpec(
        dataset_root=a.dataset_root, repo_id=a.repo_id, output_dir=a.output_dir, steps=a.steps, save_freq=a.save_freq,
        device=a.device, batch_size=a.batch_size, seed=a.seed, num_workers=a.num_workers,
        contract_path=a.contract, choreography_path=a.choreography,
    )  # fmt: skip
    r = train.run_training(spec)
    if r["preflight_errors"]:
        for e in r["preflight_errors"]:
            print(e if e.startswith("出力先が既にある") else f"PREFLIGHT NG: {e}", file=sys.stderr)
        return 2

    rc, post = r["exit_code"], r["post_check_errors"]
    print(f"exit_code: {rc}")
    print(f"device_actual: {r['device_actual']}")
    print(f"loss_first: {r['loss_first'] if r['loss_first'] is not None else 'unknown'}")
    print(f"loss_last: {r['loss_last'] if r['loss_last'] is not None else 'unknown'}")
    if r["sec_per_step_from_log"]:
        print(f"sec_per_step: {r['sec_per_step']:.4f} (ログの updt_s+data_s の平均)")
    else:
        print(f"sec_per_step: {r['sec_per_step']:.4f} (ログから読めず・全体の秒数 ÷ steps)")
    print(f"sec_per_step_wall: {r['sec_per_step_wall']:.4f} (全体の秒数 ÷ steps)")
    print(f"total_sec: {r['total_sec']:.1f}")
    for e in post:
        print(f"POST_CHECK NG: {e}")
    log_path, run_json = r["log_path"], r["run_json"]
    print(f"log: {log_path.relative_to(ACT_DIR) if log_path.is_relative_to(ACT_DIR) else log_path.name}")
    print(f"run_json: {run_json.relative_to(ACT_DIR) if run_json.is_relative_to(ACT_DIR) else run_json.name}")
    return rc if rc != 0 else (1 if post else 0)


if __name__ == "__main__":
    sys.exit(main())
