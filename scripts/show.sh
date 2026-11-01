#!/usr/bin/env bash
set -euo pipefail
# show.sh — Gazebo を立ち上げ、お手本（choreography）か npz を再生ノード player で踊らせる（host で動かす）。
# 使い方: bash scripts/show.sh choreography
#         bash scripts/show.sh npz <repo の中の npz のパス>
# 失敗はすべて「何が・どこで → どうすればいいか」の 1 行を出して終了コード 1。
# 止めるのは Ctrl+C（再生だけ止まる）。コンテナと Gazebo は残る。全部止めるのは docker compose down。
cd "$(dirname "$0")/.."

USAGE="使い方が違う（show.sh の引数）→ bash scripts/show.sh choreography か bash scripts/show.sh npz <repo の中の npz のパス>"

die() {
  echo "$1"
  exit 1
}

MODE="${1:-}"
NPZ_REL=""
case "$MODE" in
  choreography)
    if [ "$#" -ne 1 ]; then die "$USAGE"; fi
    ;;
  npz)
    if [ "$#" -ne 2 ]; then die "$USAGE"; fi
    ;;
  *)
    die "$USAGE"
    ;;
esac

ROOT="$(pwd)"

# 1. npz は repo の中にあること。コンテナ内のパスは /workspace/<repo からの相対パス
if [ "$MODE" = "npz" ]; then
  NPZ_ARG="$2"
  if [ ! -f "$NPZ_ARG" ]; then
    die "npz が見つからない（${NPZ_ARG}）→ repo の中の npz のパスを渡す"
  fi
  NPZ_REL="$(python3 -I -c '
import os, sys
root = os.path.realpath(sys.argv[1])
path = os.path.realpath(sys.argv[2])
if os.path.commonpath([root, path]) != root:
    print("OUTSIDE")
else:
    print(os.path.relpath(path, root))
' "$ROOT" "$NPZ_ARG")" || die "npz のパスを調べられない（${NPZ_ARG}）→ パスを確かめてからもう一度"
  if [ "$NPZ_REL" = "OUTSIDE" ]; then
    die "npz が repo の外にある（${NPZ_ARG}）→ npz は repo の中に置く（ros/ の下など）"
  fi
  case "$NPZ_REL" in
    *[!A-Za-z0-9._/-]*)
      die "npz のパスに使えない文字がある（${NPZ_REL}）→ 英数字と . _ - / だけのパスに置き直す"
      ;;
  esac
fi

# 2. docker が使えること
if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
  die "docker または docker compose が使えない（host）→ Docker Desktop を起動してからもう一度"
fi

# 3. 同じ名前のコンテナが別の場所から動いていたら止めない
OTHER_DIR="$(docker inspect -f '{{index .Config.Labels "com.docker.compose.project.working_dir"}}' pai_fin2_ros 2>/dev/null || true)"
if [ -n "$OTHER_DIR" ] && [ "$OTHER_DIR" != "$ROOT" ]; then
  die "別の場所（$(basename "$OTHER_DIR")）から起動した pai_fin2_ros が動いている → そちらで docker compose down してからもう一度"
fi

# 3b. 起動の排他（原子的）：host のロックファイルを flock で取り、再生が終わるまで持つ
#     fd 9 を開いたまま player まで持つ。show.sh と、そこから起動した docker compose exec が終われば外れる
#     （取り残しの掃除は要らない）。取れなければ、何も起動せず・何も止めずに 1 行で知らせて終わる。
#     ロックはコンテナ名ごとに 1 つ（同じ名前のコンテナは 1 つしか動かないため）
LOCK_FILE="${SHOW_LOCK_FILE:-${TMPDIR:-/tmp}/pai_fin2_ros.show.lock}"
if ! exec 9>>"$LOCK_FILE"; then
  die "排他のロックファイルを開けない（${LOCK_FILE}）→ TMPDIR を確かめてからもう一度"
fi
if ! python3 -I -c '
import fcntl, sys
try:
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    sys.exit(1)
'; then
  die "別の show.sh が動いている（コンテナ pai_fin2_ros）→ その端末で Ctrl+C して止めてからもう一度"
fi

# コンテナの中で ROS を使う。$HOME はコンテナの中で展開させる
rexec() {
  docker compose exec -T -u ubuntu -e DISPLAY=:0 ros bash -c "source /opt/ros/humble/setup.bash && source \"\$HOME/airobot_ws/install/setup.bash\" && $1"
}

# 4. 起動して X の準備を待つ
if ! docker compose up -d >/dev/null 2>&1; then
  die "コンテナを起動できない（docker compose up -d）→ docker compose ps と docker compose logs で確かめる"
fi
X_OK=0
for _ in $(seq 1 30); do
  if docker compose exec -T ros test -S /tmp/.X11-unix/X0 >/dev/null 2>&1; then
    X_OK=1
    break
  fi
  sleep 2
done
if [ "$X_OK" != "1" ]; then
  die "画面サーバが立たない（コンテナ pai_fin2_ros）→ docker compose down してからもう一度"
fi

# 5. Gazebo（動いていれば二重に起動しない）
if ! docker compose exec -T ros pgrep -f "ign gazebo" >/dev/null 2>&1; then
  if ! docker compose exec -d -u ubuntu -e DISPLAY=:0 ros bash -c 'source /opt/ros/humble/setup.bash && source "$HOME/airobot_ws/install/setup.bash" && ros2 launch crane_plus_gazebo no_moveit_crane_plus_with_table.launch.py > /tmp/show_gz.log 2>&1' >/dev/null 2>&1; then
    die "Gazebo を起動できない（コンテナ pai_fin2_ros）→ docker compose down してからもう一度"
  fi
fi

# 6. /joint_states を待つ（壁時計で合計 180 秒・試すたびに 5 秒あける）
#    ros2 topic echo は使わない：CLI が初回に起動する常駐プロセス（ros2 daemon）が、timeout の時間切れで
#    道連れに止められて壊れ、以後の ros2 topic echo が全部失敗することがあった（実機で起きた）。
#    代わりに rclpy で /joint_states を 1 本受けるだけの小さな Python で待つ（daemon を使わない）
JS_PROBE="python3 - <<'PY'
import sys, time, rclpy
from sensor_msgs.msg import JointState
rclpy.init()
n = rclpy.create_node('show_js_probe')
got = []
n.create_subscription(JointState, '/joint_states', lambda m: got.append(1), 10)
end = time.monotonic() + 8.0
while not got and time.monotonic() < end:
    rclpy.spin_once(n, timeout_sec=0.1)
n.destroy_node(); rclpy.shutdown()
sys.exit(0 if got else 1)
PY"
JS_OK=0
JS_START="$(date +%s)"
while :; do
  if rexec "timeout 20 $JS_PROBE" >/dev/null 2>&1; then
    JS_OK=1
    break
  fi
  if [ $(( $(date +%s) - JS_START )) -ge "${SHOW_JS_DEADLINE_S:-180}" ]; then
    break
  fi
  sleep "${SHOW_JS_INTERVAL_S:-5}"
done
if [ "$JS_OK" != "1" ]; then
  die "Gazebo が 180 秒以内に立ち上がらない（コンテナ pai_fin2_ros）→ docker compose down してからもう一度"
fi

# 7. 再生ノードをビルド
if ! docker compose exec -T -u ubuntu ros bash -c 'source /opt/ros/humble/setup.bash && cd /workspace/ros && colcon build'; then
  die "再生ノードのビルドに失敗（/workspace/ros・ログは ros/log）→ ros/log の中の最新のログを見て直してからもう一度"
fi

# 8. 起動の排他（もう一段）：show.sh を通さずに動かした player が既にいたら、止めずに知らせて終わる
#    show.sh どうしの同時起動は 3b のロックが防ぐ。pgrep の出力は変数に受けてから見る（パイプに grep -q を置かない）
RUNNING="$(docker compose exec -T ros pgrep -f "crane_dance/lib/crane_dance/player|ros2 run crane_dance player" 2>/dev/null || true)"
if [ -n "$RUNNING" ]; then
  die "別の再生（player）が動いている（コンテナ pai_fin2_ros）→ その端末で Ctrl+C して止めてからもう一度"
fi

# 10. Ctrl+C・終了信号では、自分が起動した player だけを止めてから終わる（コンテナと Gazebo は残す）
#     自分の player は run_id で見分ける（同じ checkout の別の再生は止めない）
RUN_ID="show$$x$(date +%s)"
stop_player() {
  docker compose exec -T ros pkill -INT -f "run_id:=${RUN_ID}" >/dev/null 2>&1 || true
}
on_int() {
  trap - INT TERM
  stop_player
  exit 130
}
on_term() {
  trap - INT TERM
  docker compose exec -T ros pkill -TERM -f "run_id:=${RUN_ID}" >/dev/null 2>&1 || true
  exit 143
}
trap on_int INT
trap on_term TERM

# 9. player を前で動かす
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || true)"
PARAMS="-p use_sim_time:=true -p source:=$MODE -p run_id:=$RUN_ID"
if [ "$MODE" = "npz" ]; then
  PARAMS="$PARAMS -p npz:=/workspace/$NPZ_REL"
fi
if [ -n "$GIT_SHA" ]; then
  PARAMS="$PARAMS -p git_sha:=$GIT_SHA"
fi
TTY_FLAG=()
if [ ! -t 0 ]; then
  TTY_FLAG=(-T)
fi
RC=0
docker compose exec ${TTY_FLAG[@]+"${TTY_FLAG[@]}"} -u ubuntu -e DISPLAY=:0 ros bash -c "source /opt/ros/humble/setup.bash && source \"\$HOME/airobot_ws/install/setup.bash\" && source /workspace/ros/install/setup.bash && ros2 run crane_dance player --ros-args $PARAMS" || RC=$?
exit "$RC"
