"""バッチ道具の子プロセス（学習・評価・書き出しの 1 段）。親は crane_dance_act.batch。cwd に依存しない。"""
import os
import pathlib
import sys

os.environ["HF_HUB_OFFLINE"] = "1"

ACT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACT_DIR))

from crane_dance_act import batch_steps  # noqa: E402

if __name__ == "__main__":
    sys.exit(batch_steps.main(sys.argv[1:]))
