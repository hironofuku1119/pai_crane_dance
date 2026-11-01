#!/usr/bin/env bash
set -euo pipefail
# local_check.sh — compose の設定・起動・公開ポート・Gazebo のシム時刻を確かめる。
# 最終行は LOCAL_CHECK: PASS か LOCAL_CHECK: FAIL (<理由>)。FAIL のときは終了コード 1。
# 途中でコマンドが異常終了したとき・中断されたとき・停止（docker compose down）に失敗したときも FAIL。
cd "$(dirname "$0")/.."

STARTED=0
FAIL_REASON=""
LECTURE_BEFORE=""
# /clock の判定（壁時計の秒）：この秒数のあいだ受け続け、受信の間隔がこれより空いたら「止まった」
CLOCK_WINDOW_S=10
CLOCK_MAX_GAP_S=0.5
# 講座イメージの digest（compose.yaml の image と、実際に動いているイメージの両方をこの値と照合する）
EXPECTED_DIGEST="@sha256:b58ec74c019aacd9ce4818fa25fc5a9e0a1f22f8c9cd1f5260994bf15cd9eb63"
# /joint_states を待つ：Gazebo が立ち上がるまで、この間隔（秒）で、この合計秒数まで試す（テストでは環境変数で短くする）
JS_INTERVAL_S="${LOCAL_CHECK_JS_INTERVAL_S:-5}"
JS_DEADLINE_S="${LOCAL_CHECK_JS_DEADLINE_S:-180}"

fail() {
  if [ -z "$FAIL_REASON" ]; then
    FAIL_REASON="$1"
  fi
}

on_signal() {
  fail "中断された"
  exit 130
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if [ "$rc" -ne 0 ]; then
    fail "途中で異常終了した（終了コード $rc）"
  fi
  if [ "$STARTED" = "1" ]; then
    if ! docker compose down >/dev/null 2>&1; then
      fail "停止（docker compose down）に失敗した"
    fi
  fi
  if [ -z "$FAIL_REASON" ]; then
    echo "LOCAL_CHECK: PASS"
    exit 0
  fi
  echo "LOCAL_CHECK: FAIL ($FAIL_REASON)"
  exit 1
}
trap cleanup EXIT
trap on_signal INT TERM HUP

# コンテナの中で ROS を使う。\$HOME はコンテナの中で展開させる（host 側では展開しない）。
rexec() {
  docker compose exec -T -u ubuntu -e DISPLAY=:0 ros bash -c "source /opt/ros/humble/setup.bash && source \"\$HOME/airobot_ws/install/setup.bash\" && $1"
}

step1() {
  if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
    fail "docker または docker compose が使えない"
    return
  fi
  LECTURE_BEFORE="$(docker inspect -f '{{.State.Status}} {{.State.StartedAt}}' pai_ros2 2>/dev/null || echo absent)"
  local running
  running="$(docker inspect -f '{{.State.Running}}' pai_fin2_ros 2>/dev/null || echo false)"
  if [ "$running" = "true" ]; then
    fail "既に起動している"
  fi
}

# 正規化された設定の全サービスを照合する。サービスは ros だけ・公開は 127.0.0.1 の 6081 だけ。
step2() {
  local cfg verdict
  # --no-interpolate：.env などの変数を展開しない（秘密を平文にしない）。ポートの正規化はされる
  if ! cfg="$(docker compose config --no-interpolate --format json 2>/dev/null)"; then
    fail "compose の設定を取得できない"
    return
  fi
  verdict="$(python3 -I -c '
import json, sys
try:
    services = json.load(sys.stdin)["services"]
except Exception:
    print("compose の設定を読めない"); sys.exit(0)
if set(services) != {"ros"}:
    print("ros 以外のサービスがある"); sys.exit(0)
if "network_mode" in services["ros"]:
    print("network_mode がある"); sys.exit(0)
ports = services["ros"].get("ports") or []
if not all(isinstance(p, dict) and p.get("host_ip") == "127.0.0.1" for p in ports):
    print("ポートが loopback だけではない"); sys.exit(0)
if len(ports) != 1 or {str(p.get("published")) for p in ports} != {"6081"}:
    print("公開するポートが 6081 の 1 つではない"); sys.exit(0)
print("OK")
' <<<"$cfg")" || verdict="compose の設定の照合に失敗した"
  if [ "$verdict" != "OK" ]; then
    fail "$verdict"
  fi
}

step4() {
  local img digest repo
  if ! img="$(docker inspect -f '{{.Image}}' pai_fin2_ros 2>/dev/null)"; then
    fail "コンテナのイメージを調べられない"
    return
  fi
  digest="$(python3 -I -c '
import re
m = re.search(r"@sha256:[0-9a-f]{64}", open("compose.yaml", encoding="utf-8").read())
print(m.group(0) if m else "")
')"
  if [ "$digest" != "$EXPECTED_DIGEST" ]; then
    fail "digest が宣言と違う"
    return
  fi
  if [ -z "$digest" ] || ! repo="$(docker image inspect --format '{{json .RepoDigests}}' "$img" 2>/dev/null)"; then
    fail "digest が宣言と違う"
    return
  fi
  case "$repo" in
    *"$digest"*) ;;
    *) fail "digest が宣言と違う" ;;
  esac
}

step5() {
  local i ok=0
  for i in $(seq 1 30); do
    if curl -sf -o /dev/null http://127.0.0.1:6081/; then
      ok=1
      break
    fi
    sleep 2
  done
  if [ "$ok" != "1" ]; then
    fail "noVNC が応答しない"
    return
  fi
  ok=0
  for i in $(seq 1 30); do
    if docker compose exec -T ros test -S /tmp/.X11-unix/X0 >/dev/null 2>&1; then
      ok=1
      break
    fi
    sleep 2
  done
  if [ "$ok" != "1" ]; then
    fail "画面サーバが立たない"
  fi
}

# compose が立てた全コンテナの公開ポートを照合する（コンテナは 1 つだけのはず）。
step6() {
  local ids id out line n=0
  if ! ids="$(docker compose ps -q 2>/dev/null)"; then
    fail "コンテナの一覧を取得できない"
    return
  fi
  while IFS= read -r id; do
    if [ -z "$id" ]; then
      continue
    fi
    n=$((n + 1))
    if ! out="$(docker port "$id" 2>/dev/null)"; then
      fail "公開ポートを取得できない"
      return
    fi
    while IFS= read -r line; do
      if [ -z "$line" ]; then
        continue
      fi
      case "$line" in
        *"127.0.0.1:"*) ;;
        *) fail "公開が loopback だけではない"; return ;;
      esac
    done <<<"$out"
    case "$out" in
      *"0.0.0.0"* | *"[::]"* | *"5900"*) fail "公開が loopback だけではない"; return ;;
    esac
  done <<<"$ids"
  if [ "$n" -ne 1 ]; then
    fail "コンテナが 1 つではない（$n 個）"
  fi
}

step7() {
  if ! docker compose exec -d -u ubuntu -e DISPLAY=:0 ros bash -c 'source /opt/ros/humble/setup.bash && source "$HOME/airobot_ws/install/setup.bash" && ros2 launch crane_plus_gazebo crane_plus_with_table.launch.py > /tmp/local_check_gz.log 2>&1'; then
    fail "Gazebo を起動できない"
  fi
}

# /joint_states を壁時計の締め切りで待ち、試すたびに JS_INTERVAL_S 秒あける。
# ros2 topic echo は使わない（show.sh と同じ理由）：CLI が初回に起動する常駐プロセス
# （ros2 daemon）が timeout の時間切れで道連れに止められて壊れ、以後の ros2 コマンドが全部失敗することがあった
# （実機で起きた）。代わりに rclpy で /joint_states を 1 本受けるだけの小さな Python で待つ（daemon を使わない）。
JS_PROBE="python3 - <<'PY'
import sys, time, rclpy
from sensor_msgs.msg import JointState
rclpy.init()
n = rclpy.create_node('local_check_js_probe')
got = []
n.create_subscription(JointState, '/joint_states', lambda m: got.append(1), 10)
end = time.monotonic() + 8.0
while not got and time.monotonic() < end:
    rclpy.spin_once(n, timeout_sec=0.1)
n.destroy_node(); rclpy.shutdown()
sys.exit(0 if got else 1)
PY"
step8() {
  local ok=0 start now
  start="$(date +%s)"
  while :; do
    if rexec "timeout 20 $JS_PROBE" >/dev/null 2>&1; then
      ok=1
      break
    fi
    now="$(date +%s)"
    if [ $((now - start)) -ge "$JS_DEADLINE_S" ]; then
      break
    fi
    sleep "$JS_INTERVAL_S"
  done
  if [ "$ok" != "1" ]; then
    fail "/joint_states が ${JS_DEADLINE_S} 秒以内に届かない"
  fi
}

# /clock を壁時計 CLOCK_WINDOW_S 秒のあいだ受け続け、コンテナからは生の数値だけを受け取り、判定は host 側でする。
# 途中で止まった（最後の受信から測定終了まで、または受信の間隔が CLOCK_MAX_GAP_S を超えた）ら FAIL。
# 続けて届いたときは、測定区間全体の受信周期と実時間比を出す。
step9() {
  local raw line verdict
  raw="$(rexec "timeout $((CLOCK_WINDOW_S + 20)) python3 - <<'PY'
import time
import rclpy
from rosgraph_msgs.msg import Clock

WINDOW = $CLOCK_WINDOW_S
rclpy.init()
node = rclpy.create_node('local_check_clock')
st = {'n': 0, 'w_first': None, 'w_last': None, 's_first': None, 's_last': None, 'max_gap': 0.0}

def cb(m):
    w = time.monotonic()
    s = m.clock.sec + m.clock.nanosec * 1e-9
    if st['w_first'] is None:
        st['w_first'], st['s_first'] = w, s
    else:
        st['max_gap'] = max(st['max_gap'], w - st['w_last'])
    st['w_last'], st['s_last'] = w, s
    st['n'] += 1

node.create_subscription(Clock, '/clock', cb, 100)
t0 = time.monotonic()
while st['n'] == 0 and time.monotonic() - t0 < 5.0:
    rclpy.spin_once(node, timeout_sec=0.05)
if st['n'] == 0:
    print('CLOCKRAW n=0')
else:
    while time.monotonic() - st['w_first'] < WINDOW:
        rclpy.spin_once(node, timeout_sec=0.05)
    w_end = time.monotonic()
    print('CLOCKRAW n=%d w_first=%.6f w_last=%.6f w_end=%.6f s_first=%.9f s_last=%.9f max_gap=%.6f' % (
        st['n'], st['w_first'], st['w_last'], w_end, st['s_first'], st['s_last'], st['max_gap']))
node.destroy_node()
rclpy.shutdown()
PY" 2>/dev/null)" || raw=""
  line="$(grep -E '^CLOCKRAW ' <<<"$raw" | tail -n 1 || true)"
  verdict="$(python3 -I -c '
import sys
window, gap_limit, line = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
if not line.startswith("CLOCKRAW "):
    print("FAIL シム時刻を測れない"); sys.exit(0)
v = dict(kv.split("=", 1) for kv in line.split()[1:])
if int(v["n"]) < 2:
    print("FAIL /clock が届かない"); sys.exit(0)
n = int(v["n"])
wf, wl, we = float(v["w_first"]), float(v["w_last"]), float(v["w_end"])
sf, sl, gap = float(v["s_first"]), float(v["s_last"]), float(v["max_gap"])
if we - wf < window - 0.5:
    print("FAIL 測定が途中で終わった"); sys.exit(0)
if we - wl > gap_limit or gap > gap_limit:
    print("FAIL /clock が途中で止まった"); sys.exit(0)
dw = wl - wf
rtf = (sl - sf) / dw
print("clock_hz=%.1f rtf=%.3f" % ((n - 1) / dw, rtf))
if rtf < 0.05:
    print("FAIL シム時刻が進まない")
' "$CLOCK_WINDOW_S" "$CLOCK_MAX_GAP_S" "$line")" || verdict="FAIL シム時刻の判定に失敗した"
  while IFS= read -r line; do
    case "$line" in
      clock_hz=*) echo "$line" ;;
      "FAIL "*) fail "${line#FAIL }" ;;
    esac
  done <<<"$verdict"
}

# 測定のあいだに Gazebo が落ちていないか。
step9b() {
  if ! rexec 'pgrep -f "ign gazebo" >/dev/null' >/dev/null 2>&1; then
    fail "Gazebo が途中で落ちた"
  fi
}

step10() {
  local after
  after="$(docker inspect -f '{{.State.Status}} {{.State.StartedAt}}' pai_ros2 2>/dev/null || echo absent)"
  if [ "$after" != "$LECTURE_BEFORE" ]; then
    fail "pai_ros2 の状態が変わった"
  fi
}

step1
if [ -z "$FAIL_REASON" ]; then step2; fi
if [ -z "$FAIL_REASON" ]; then
  STARTED=1
  docker compose up -d >/dev/null 2>&1 || fail "compose up に失敗"
fi
if [ -z "$FAIL_REASON" ]; then step4; fi
if [ -z "$FAIL_REASON" ]; then step5; fi
if [ -z "$FAIL_REASON" ]; then step6; fi
if [ -z "$FAIL_REASON" ]; then step7; fi
if [ -z "$FAIL_REASON" ]; then step8; fi
if [ -z "$FAIL_REASON" ]; then step9; fi
if [ -z "$FAIL_REASON" ]; then step9b; fi
if [ -z "$FAIL_REASON" ]; then step10; fi
