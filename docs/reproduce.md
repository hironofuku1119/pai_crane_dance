# 再現の手順（詳細）

README の「始め方」の細かい版です。コマンドは、特に書いていなければ Mac のターミナルで、repo の根から打ちます。

## 1. 確かめた版

| 道具 | 版 |
|---|---|
| uv | 0.13.5 |
| Python（Mac・学習） | 3.12（`act/.python-version`。依存は `act/uv.lock` で固定） |
| lerobot / torch / librosa | 0.6.1 / 2.11.0 / 1.0.0 |
| ffmpeg | 8.1.2（Homebrew） |
| Docker Desktop | 29.8.2 |
| Python（Mac・検査と解析） | 3.10 以上（標準ライブラリだけ） |
| 講座の Docker イメージ | `airobotbook/ros2-desktop-ai-robot-book-humble@sha256:b58ec74c019aacd9ce4818fa25fc5a9e0a1f22f8c9cd1f5260994bf15cd9eb63`（amd64） |
| イメージの中身 | Ubuntu 22.04.5・ROS 2 Humble・Gazebo Sim 6.16.0（Fortress）・crane_plus 2.0.1・Python 3.10.12 |

講座のイメージは Intel（amd64）向けなので、Apple Silicon の Mac ではエミュレーションで動きます。

## 2. 学習環境

```bash
cd act
uv python install 3.12        # 初回だけ
uv sync                       # act/uv.lock どおりに環境を作る
uv run pytest -q              # 最後に passed と出れば通過
uv run python -c "import torch; print(torch.backends.mps.is_available())"   # True なら Apple の GPU が使える
cd ..
```

## 3. 曲の拍の解析（曲のファイルがあるときだけ）

拍の解析結果 `act/measurements/beats.json` は入っているので、学習と再生にはこの手順は要りません。曲のファイルを `data/PAImusic.wav` に置いたときだけ、解析し直せます。`beats.json` には曲のファイルの sha256 が入っていて、同じ曲なら同じ時刻表になります。`data/` は git の対象外です。

```bash
cd act
uv run python scripts/analyze_song.py --dry-run   # 書き込まずに表示だけ
uv run python scripts/analyze_song.py             # measurements/beats.json を書き直す
CRANE_SONG_WAV=../data/PAImusic.wav uv run pytest -q   # 実際の曲を使うテストも走らせる（未設定なら飛ばす）
cd ..
```

## 4. 学習と書き出し

### 4.1 本番の方策 B5

```bash
cd act
uv run python scripts/run_batch.py --config batches/b5.toml --dry-run   # 計画だけ見る
caffeinate -i uv run python scripts/run_batch.py --config batches/b5.toml
cd ..
```

- 1 本の設定ファイルで、データの合成（20 本）→ LeRobotDataset への変換 → 学習（20,000 step・バッチ 32・学習率 1e-4）→ 評価 → npz の書き出しまで進みます。`caffeinate -i` は学習中に Mac がスリープしないようにします。
- 出力は `act/out/batch/b5/`（git の対象外）。書き出した npz は `runs/b5/export/020000-open/trajectory.npz` です。
- 途中で止まったら、同じコマンドをもう一度打つと、終わった段を飛ばして続きから進みます。
- レポートの B5 は、このバッチ道具ができる前の道具で、同じデータ・同じ設定で学習しました。学習データの配列が一致することは確かめていますが、学習し直したときに重みまで一致するかは確かめていません（Apple の GPU での学習は完全な決定性を保証しません）。

### 4.2 比較用のパターン

| 設定 | 中身 |
|---|---|
| `act/batches/example.toml` | B1〜B5（観測のばらつき・関節の角度を定数・chunk 100・進行度だけ） |
| `act/batches/extra-c1-c5.toml` | c1〜c5（乱数の種・VAE なし・学習率 3e-4） |
| `act/batches/v2-lr1e-5.toml` | v2 の 3 本と E1（学習率 1e-5・10,000 step） |

```bash
cd act
caffeinate -i uv run python scripts/run_batch.py --config batches/example.toml
cd ..
```

### 4.3 学習データだけを作る

```bash
cd act
uv run python scripts/make_synth_npz.py --episodes 20 --seed 0 --out-dir out/synth-v2
uv run python scripts/npz_to_lerobot.py --in-dir out/synth-v2 \
  --out-root out/lerobot/crane_dance_synth_v2 --repo-id local/crane_dance_synth_v2
cd ..
```

1 行目で `out/synth-v2/` に `ep000.npz`〜`ep019.npz` と `manifest.json` ができ、2 行目で `out/lerobot/crane_dance_synth_v2/` に LeRobotDataset ができます。

## 5. Gazebo で踊らせる

講座では、Docker イメージを次のように取得・起動するよう示されています。

```bash
docker pull airobotbook/ros2-desktop-ai-robot-book-humble:latest
docker run -p 6080:80 --shm-size=512m airobotbook/ros2-desktop-ai-robot-book-humble:latest
```

この repo では同じイメージを `compose.yaml` から起動します。違いは 2 点です：①`latest`（後から中身が変わりうる名札）ではなく、取得した時点の中身を指す digest で固定しています。②講義用のコンテナとぶつからないよう、画面のポートを 6081 にし、この repo をコンテナの `/workspace` に見せています。共有メモリは講座と同じ 512 MB です。コンテナの中で書き込めるのは `/workspace/ros` の下だけです。

1. コンテナを起動します（コンテナ名 `pai_fin2_ros`）。

   ```bash
   docker compose up -d
   ```

2. 画面の係が動いているか確かめます。`novnc` と `vnc` の 2 行が両方 `RUNNING` なら次へ。`STARTING` なら数秒待って打ち直します。

   ```bash
   docker compose exec ros supervisorctl status
   ```

3. パスワードを確かめます。表示された 1 行の、`${PASSWORD:-` と `}` のあいだの文字がパスワードです（講座のイメージの既定の値）。

   ```bash
   docker compose exec ros grep -m1 '^VNC_PASSWORD=' /entrypoint.sh
   ```

4. Chrome で `http://localhost:6081/vnc.html?resize=scale` を開き、「接続」（Connect）を押します。「パスワード:」の欄に 3 の文字を入れ、「資格情報を送信」（Send Credentials）を押します。
5. お手本（振付そのもの）を踊らせます。初回は Gazebo の立ち上がりと ROS パッケージのビルドで数分かかります。

   ```bash
   bash scripts/show.sh choreography
   ```

6. 学習した方策の出力（4.1 で書き出した npz）を踊らせます。npz は repo の中に置いたものしか渡せません。

   ```bash
   bash scripts/show.sh npz act/out/batch/b5/runs/b5/export/020000-open/trajectory.npz
   ```

- Ctrl+C で再生だけが止まります。全部止めるときは `docker compose down`。
- つながらないとき：2 で `RUNNING` でない行があれば `docker compose down` → `docker compose up -d` からやり直します。パスワードが違うと言われたら 3 をもう一度見ます。
- `show.sh` は、動作計画の MoveIt を立てない起動ファイル（`no_moveit_crane_plus_with_table.launch.py`）を使います。6 の計測は、講座と同じ MoveIt ありの起動ファイルで行います。
- 録画するときは、踊らせながら macOS の画面収録（Cmd+Shift+5）で Chrome の画面を収録します。Gazebo は音を出さないので、編集で曲を重ねます（再生ノードがログに出す開始時刻 t0 で合わせる）。

## 6. 時間のずれを測る

同じ 1 回の再生で、シム時刻と実時間を両方記録して解析します。5 の 1〜4 で画面につないでから、コンテナのデスクトップでターミナルを 3 つ開いて打ちます。

**ターミナル 1：Gazebo を立てる（開いたまま）**

```bash
source /opt/ros/humble/setup.bash
source ~/airobot_ws/install/setup.bash
mkdir -p /workspace/ros/bags/timing
ros2 launch crane_plus_gazebo crane_plus_with_table.launch.py 2>&1 | tee /workspace/ros/bags/timing/gazebo_launch.log
```

Gazebo の窓にロボットが出るまで待ちます。

**ターミナル 2：制御器を確かめて、記録を始める（開いたまま）**

```bash
source /opt/ros/humble/setup.bash
source ~/airobot_ws/install/setup.bash
ros2 topic list --no-daemon | grep joint_trajectory
```

`/crane_plus_arm_controller/joint_trajectory` と `/crane_plus_gripper_controller/joint_trajectory` の 2 行が出たら、記録を始めます。出なければ 10 秒待って打ち直します。

```bash
cd /workspace/ros/bags/timing
ros2 bag record -o run1 /clock /joint_states /crane_dance/joint_target
```

このターミナルでスペースキーを押さないでください（記録が一時停止します）。

**ターミナル 3：再生ノードを動かす**

```bash
source /opt/ros/humble/setup.bash
source ~/airobot_ws/install/setup.bash
source /workspace/ros/install/setup.bash
```

最後の行で `No such file or directory` が出たときだけ、再生ノードをビルドしてから、もう一度読み込みます。

```bash
cd /workspace/ros && colcon build && source /workspace/ros/install/setup.bash
```

再生します。各行の頭に、その行が出た実時間（壁時計と、戻らない時計の 2 つ）を付けて、ファイルにも残します。1 行で打ちます。

```bash
cd /workspace/ros/bags/timing && ros2 run crane_dance player --ros-args -p use_sim_time:=true -p record:=true -p source:=npz -p npz:=/workspace/act/out/batch/b5/runs/b5/export/020000-open/trajectory.npz 2>&1 | python3 -u -c 'import sys,time; [print("%.3f %.3f %s" % (time.time(), time.monotonic(), l), end="", flush=True) for l in iter(sys.stdin.readline, "")]' | tee run1_player.txt
```

`終わりの姿勢と初期姿勢の差の最大 = … rad` の行が出たら終わりです。再生ノードは、記録側が `/crane_dance/joint_target` をつかんだのを確かめてから踊り始めます。

**記録を止めて書き出す**：ターミナル 2 で Ctrl+C を 1 回押して記録を止め、ターミナル 3 で JSONL に書き出します。

```bash
ros2 bag info run1      # /crane_dance/joint_target の Count が 1560 なら 1 回分そろっている
python3 /workspace/scripts/analyze_timing.py extract run1 run1.jsonl
```

**Mac 側で解析する**（コンテナの `/workspace/ros` は Mac の `ros/` と同じ場所です）。要約 `summary.md`・表・グラフが `--out` のフォルダにできます。

```bash
python3 scripts/analyze_timing.py analyze ros/bags/timing/run1.jsonl \
  --npz act/out/batch/b5/runs/b5/export/020000-open/trajectory.npz \
  --player-log ros/bags/timing/run1_player.txt --out ros/bags/timing/run1_analysis
```

- 最終行に `TIMING: PASS` か `TIMING: FAIL (n)` が出ます。検査が 1 つでも通らないと、要約の先頭に「無効」と出て、終了コード 1 で終わります。
- 記録の生のファイルは `ros/bags/`（git の対象外）に置きます。
- 2 回目は、名前を `run2` などに変えて、ターミナル 2 の記録からやり直します。止めるときは、Mac で `docker compose down`。

## 7. 検査

| 何を | コマンド | 通過の印 |
|---|---|---|
| 単体テスト | `python3 -m unittest discover -s tests` | `OK` |
| 約束（振付）の検査 | `python3 scripts/check_contract.py` | `CONTRACT: PASS` |
| npz の検査 | `python3 scripts/check_contract.py --npz <npz のパス>` | `NPZ: PASS` |
| 学習側のテスト | `cd act && uv run pytest -q` | `passed` |
| コンテナの動作確認（起動・公開ポート・Gazebo のシム時刻） | `bash scripts/local_check.sh` | `LOCAL_CHECK: PASS` |

`local_check.sh` は、このプロジェクトのコンテナ（`pai_fin2_ros`）が動いていると最初に止まるので、先に `docker compose down` しておきます。GitHub Actions（`.github/workflows/check.yml`）は、秘密の混入の検査と単体テストを回します。

## 8. 振付の 4 つの姿勢の記録（`docs/keyframes/`）

| 姿勢 | 角度の記録 | 画像 |
|---|---|---|
| 直立・グリッパ閉 | `standing_closed.txt` | `standing_closed.png` |
| 直立・グリッパ開 | `standing_opened.txt` | `standing_opened.png` |
| おじぎ・グリッパ閉 | `bowing_closed.txt` | `bowing_closed.png` |
| おじぎ・グリッパ開 | `bowing_opened.txt` | `bowing_opened.png` |

角度の記録は `/joint_states` を 1 回分取り出した出力そのままです。関節の並びは `joint1, joint2, joint4, joint3, hand` で、振付ファイルの並び（`joint1, joint2, joint3, joint4, hand`）と 3 番目と 4 番目が入れ替わっているので、`name` の並びに合わせて読みます。
