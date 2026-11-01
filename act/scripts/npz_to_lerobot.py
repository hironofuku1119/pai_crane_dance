"""合成 npz のフォルダを LeRobotDataset に変換する。cwd に依存しない。"""
import os

# lerobot を import する前にオフライン固定（import 時に 1 回だけ読まれるため）。
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

ACT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACT_DIR))

from crane_dance_act import synth, to_lerobot  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", type=pathlib.Path, default=ACT_DIR.parent / "contract" / "contract.json")
    ap.add_argument("--choreography", type=pathlib.Path, default=ACT_DIR.parent / "contract" / "choreography.json")
    ap.add_argument("--in-dir", type=pathlib.Path, default=ACT_DIR / "out" / "synth")
    ap.add_argument("--out-root", type=pathlib.Path, default=ACT_DIR / "out" / "lerobot" / "crane_dance_synth_v1")
    ap.add_argument("--repo-id", default=to_lerobot.DEFAULT_REPO_ID)
    a = ap.parse_args()

    contract = synth.load_contract(a.contract)
    paths = sorted(a.in_dir.glob("ep*.npz"))
    manifest = a.in_dir / "manifest.json"
    out = to_lerobot.convert(
        paths, contract, a.choreography, a.out_root, a.repo_id, manifest if manifest.is_file() else None
    )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(a.repo_id, root=out)
    print(f"episodes: {ds.num_episodes}")
    print(f"frames: {ds.num_frames}")
    print(f"fps: {ds.fps}")
    print(f"std_floor_rad: {to_lerobot.STD_FLOOR_RAD}")
    print(f"out: {a.out_root.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
