"""バッチ道具の入口。設定（batches/<名前>.toml）の [[run]] を順に、データ → データセット → 学習 → 評価 → 書き出し → 集計 と進める。cwd に依存しない。

    uv run python scripts/run_batch.py --config batches/<名前>.toml [--only <run>...] [--device mps|cpu] [--dry-run]

出力の根は act/out/batch/<設定名>/ 。終了コード: 0＝全部成功 / 1＝どれか失敗 / 2＝始める前の検査で止めた。
"""
import os
import pathlib
import sys

os.environ["HF_HUB_OFFLINE"] = "1"

ACT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACT_DIR))

from crane_dance_act import batch  # noqa: E402

if __name__ == "__main__":
    sys.exit(batch.cli_main())
