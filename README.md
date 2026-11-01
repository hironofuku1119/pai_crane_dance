# crane_dance

物理シミュレータ Gazebo の中のロボットアーム CRANE+ V2 を、自作の曲に合わせて自作の振付で踊らせるコードです。振付から合成したお手本で模倣学習の手法 ACT（Action Chunking with Transformers）を学習させ、その出力を Gazebo で再生し、シミュレーションの時計と現実の時計のずれを記録から測ります。

東京大学 松尾・岩澤研究室「Physical AI 講座」応用編1 の最終課題の提出物です。結果と考察は、別に提出したレポートにあります。

## フォルダの構成

```
act/        Mac 側：お手本の合成・学習・評価・npz の書き出し（Python 3.12・uv）
ros/        コンテナ側：再生ノード player（ROS 2 のパッケージ）
scripts/    Gazebo での再生（show.sh）・動作確認（local_check.sh）・約束の検査・時間のずれの解析
contract/   振付（choreography.json）と、Mac 側とコンテナ側の約束（contract.json）
tests/      単体テスト（act/tests は学習側のテスト）
docs/       再現の手順の詳細・用語・引用と参考・振付の姿勢の記録
compose.yaml  講座の Docker イメージを起動する設定
```

## 始め方

### 必要なもの

- Apple Silicon の Mac（学習に Apple の GPU〈MPS〉を使う）
- [uv](https://docs.astral.sh/uv/)・ffmpeg（`brew install ffmpeg`）
- Docker Desktop（講座の Docker イメージ `airobotbook/ros2-desktop-ai-robot-book-humble` を、`compose.yaml` で digest を固定して使う）
- Google Chrome（コンテナの画面を noVNC で見る）

### 手順

コマンドは repo の根から打ちます。

1. 学習環境を作り、テストと約束の検査を通す。

   ```bash
   cd act && uv sync && uv run pytest -q && cd ..
   python3 -m unittest discover -s tests
   python3 scripts/check_contract.py          # 最終行が CONTRACT: PASS
   ```

2. 本番の方策 B5 を学習し、関節の目標の列（npz）を書き出す。

   ```bash
   cd act
   caffeinate -i uv run python scripts/run_batch.py --config batches/b5.toml
   cd ..
   ```

   書き出し先は `act/out/batch/b5/runs/b5/export/020000-open/trajectory.npz` です。

3. コンテナを起動し、Gazebo で踊らせる。

   ```bash
   docker compose up -d
   bash scripts/show.sh npz act/out/batch/b5/runs/b5/export/020000-open/trajectory.npz
   ```

   Chrome で `http://localhost:6081/vnc.html?resize=scale` を開くと、Gazebo の画面が見えます（パスワードの確かめ方は `docs/reproduce.md`）。止めるときは `docker compose down`。

4. シミュレーションの時計と現実の時計のずれを測る：`docs/reproduce.md` の「時間のずれを測る」の手順で記録し、`scripts/analyze_timing.py` で解析します。

細かい手順（画面へのつなぎ方・比較用のパターン・計測の 3 つのターミナル・困ったとき）は [`docs/reproduce.md`](docs/reproduce.md) にあります。

### 入っていないもの

- 学習した重み（約 460 MB で、GitHub の 1 ファイルの上限を超えるため）。手順 2 で学習し直せます。
- 曲のファイル。拍の解析結果（`act/measurements/beats.json`）と、曲のファイルの sha256 だけが入っています。
- 計測の記録の生のファイル。

## レポートとの対応

| レポートでの名前 | この repo の場所 |
|---|---|
| 振付ファイル・約束・検査器 | `contract/choreography.json`・`contract/contract.json`・`scripts/check_contract.py` |
| 拍の時刻表 | `act/measurements/beats.json` |
| 合成のお手本 | `act/crane_dance_act/synth.py` |
| B5 の設定 | `act/batches/b5.toml` |
| 比較した 20 通りの条件 | `act/batches/`（`example.toml`・`extra-c1-c5.toml`・`v2-lr1e-5.toml`） |
| `b5_ckpt20000_open.npz` | 手順 2 で書き出す `trajectory.npz` と同じ形のもの |
| 再生ノード player | `ros/src/crane_dance/crane_dance/player.py` |
| 振付の 4 つの姿勢の記録と画像 | `docs/keyframes/` |
| 解析の道具 | `scripts/analyze_timing.py`・`scripts/timing_core.py` |
| 動作確認の道具 | `scripts/local_check.sh` |

レポートに出てくる略語・名称は [`docs/glossary.md`](docs/glossary.md)、引用と参考は [`docs/REFERENCES.md`](docs/REFERENCES.md) にあります。
