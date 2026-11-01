"""compose.yaml と scripts/local_check.sh の検査（標準ライブラリの unittest・PyYAML なし）。

compose.yaml：コメントと空行を除いた行が、許可した形（サービスは ros だけ）とちょうど一致すること。
知らない書式・別のサービス・インラインの書式・ホスト IP の省略は、どれも問題として報告する。

local_check.sh：本物の docker の代わりに、PATH の先頭に置いた偽の docker・curl で動かし、
失敗を 1 つずつ注入して、最終行が FAIL・終了コードが 0 以外になることを確かめる（Docker は起動しない）。
"""
import json
import os
import re
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ROOT / "compose.yaml"
# 講座イメージの digest（compose.yaml の書き間違いを、ここの定数との照合で捕まえる。local_check.sh の EXPECTED_DIGEST と同じ値）
EXPECTED_DIGEST = "b58ec74c019aacd9ce4818fa25fc5a9e0a1f22f8c9cd1f5260994bf15cd9eb63"
LOCAL_CHECK = ROOT / "scripts" / "local_check.sh"

PORT_RE = re.compile(r'^\s*-\s+"127\.0\.0\.1:(\d+):(\d+)"\s*$')


def declared_digest() -> str:
    return EXPECTED_DIGEST


def allowed_lines(digest: str) -> list[str]:
    """compose.yaml として許す行（コメントと空行を除く）。これ以外の行は 1 行でも問題。"""
    return [
        "services:",
        "  ros:",
        f"    image: airobotbook/ros2-desktop-ai-robot-book-humble@sha256:{digest}",
        "    platform: linux/amd64",
        "    container_name: pai_fin2_ros",
        '    shm_size: "512m"',
        "    environment:",
        '      RESOLUTION: "1920x1080"',
        '      ROS_LOCALHOST_ONLY: "1"',
        "    ports:",
        '      - "127.0.0.1:6081:80"',
        "    volumes:",
        "      - ./:/workspace:ro",
        "      - ./ros:/workspace/ros",
    ]


def check(text: str, digest: str) -> list[str]:
    """問題の一覧を返す。空なら合格。"""
    problems: list[str] = []
    body = [ln.rstrip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    want = allowed_lines(digest)
    if body != want:
        extra = [ln for ln in body if ln not in want]
        missing = [ln for ln in want if ln not in body]
        problems.append(f"許可した形と違う（余分 {extra[:3]}・不足 {missing[:3]}）")
    # 読み手に理由が分かるよう、よくある崩れ方は名前をつけて報告する
    if "5900" in text:
        problems.append("5900 が書かれている")
    if "0.0.0.0" in text or "[::]" in text:
        problems.append("0.0.0.0 または [::] がある")
    if len(re.findall(r"^\s*ports\s*:", text, re.M)) != 1:
        problems.append("ports: がちょうど 1 つではない（別のサービスに公開がある）")
    if re.search(r"^[ \t]*ports[ \t]*:[ \t]*[^ \t#\n]", text, re.M):
        problems.append("ports: のインラインの書式は許さない")
    for line in body:
        s = line.strip()
        if s.startswith("- ") and ":" in s and "/workspace" not in s and not PORT_RE.match(line):
            problems.append(f"知らないポート書式: {s}")
    return problems


class ComposeStaticTest(unittest.TestCase):
    def setUp(self):
        self.text = COMPOSE.read_text(encoding="utf-8")
        self.digest = declared_digest()

    def test_actual_passes(self):
        self.assertEqual(check(self.text, self.digest), [])

    def refute(self, new: str):
        self.assertNotEqual(new, self.text, "置き換えが効いていない")
        self.assertTrue(check(new, self.digest), "問題が報告されなかった")

    def test_all_interfaces(self):
        self.refute(self.text.replace("127.0.0.1:6081", "0.0.0.0:6081"))

    def test_no_host_ip(self):
        # ホスト IP を省くと全インターフェースに公開される
        self.refute(self.text.replace('"127.0.0.1:6081:80"', '"6081:80"'))

    def test_long_form(self):
        old = '      - "127.0.0.1:6081:80"\n'
        new = '      - target: 80\n        published: "6081"\n        host_ip: 127.0.0.1\n'
        self.refute(self.text.replace(old, new))

    def test_inline_ports(self):
        old = '    ports:\n      - "127.0.0.1:6081:80"\n'
        self.refute(self.text.replace(old, '    ports: ["127.0.0.1:6081:80"]\n'))

    def test_other_service_inline_ports(self):
        self.refute(self.text + '  web:\n    image: nginx\n    ports: ["8080:80"]\n')

    def test_other_service_block_ports(self):
        self.refute(self.text + '  web:\n    image: nginx\n    ports:\n      - "127.0.0.1:8080:80"\n')

    def test_other_service_without_ports(self):
        self.refute(self.text + "  web:\n    image: nginx\n")

    def test_vnc_port_added(self):
        old = '      - "127.0.0.1:6081:80"\n'
        self.refute(self.text.replace(old, old + '      - "127.0.0.1:5900:5900"\n'))

    def test_privileged(self):
        self.refute(self.text.replace("    shm_size:", "    privileged: true\n    shm_size:"))

    def test_network_mode_host(self):
        self.refute(self.text.replace("    shm_size:", "    network_mode: host\n    shm_size:"))

    def test_digest_changed(self):
        i = self.text.index("@sha256:") + len("@sha256:")
        c = "0" if self.text[i] != "0" else "1"
        self.refute(self.text[:i] + c + self.text[i + 1 :])

    def test_ro_removed(self):
        self.refute(self.text.replace("./:/workspace:ro", "./:/workspace"))


# ---------------------------------------------------------------- local_check.sh（偽の docker で動かす）

FAKE_DOCKER = r"""#!/usr/bin/env bash
# 偽の docker。STUB_FAIL に書いた名前の操作だけ失敗させる。
args="$*"
has() { case " ${STUB_FAIL:-} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
case "$args" in
  "compose version") has version && exit 1; exit 0 ;;
  "inspect -f {{.State.Status}} {{.State.StartedAt}} pai_ros2") echo "exited 2000-01-01T00:00:00Z"; exit 0 ;;
  "inspect -f {{.State.Running}} pai_fin2_ros") [ -n "${STUB_RUNNING:-}" ] && echo true && exit 0; exit 1 ;;
  "compose config --no-interpolate --format json") has config && exit 1; cat "$STUB_CONFIG"; exit 0 ;;
  "compose up -d") has up && exit 1; exit 0 ;;
  "compose down") has down && exit 1; exit 0 ;;
  "inspect -f {{.Image}} pai_fin2_ros") echo "sha256:fakeimage"; exit 0 ;;
  "image inspect --format {{json .RepoDigests}} sha256:fakeimage") echo "[\"airobotbook/x@sha256:${STUB_DIGEST}\"]"; exit 0 ;;
  "compose ps -q") printf '%s\n' ${STUB_IDS:-c1}; exit 0 ;;
  "compose exec -T ros ss -ltn")
    echo "State Recv-Q Send-Q Local Address:Port Peer Address:Port"
    echo "LISTEN 0 5 127.0.0.1:6081 0.0.0.0:*"
    exit 0 ;;
  port\ *) has port && echo "80/tcp -> 0.0.0.0:6081"; echo "80/tcp -> 127.0.0.1:6081"; exit 0 ;;
esac
case "$args" in
  *"test -S /tmp/.X11-unix/X0"*) exit 0 ;;
  "compose exec -d "*) has gazebo && exit 1; exit 0 ;;
  *"local_check_js_probe"*)  # /joint_states を rclpy で 1 本待つ小さな Python（ros2 topic echo は使わない）
    has joint_states && exit 1
    # STUB_JS_FAIL_FIRST 回目までは「トピックがまだ無い」としてすぐ失敗する（Gazebo の立ち上がりが遅い場合）
    if [ -n "${STUB_JS_FAIL_FIRST:-}" ]; then
      c=$(cat "$STUB_JS_COUNTER" 2>/dev/null || echo 0); c=$((c + 1)); echo "$c" > "$STUB_JS_COUNTER"
      [ "$c" -le "$STUB_JS_FAIL_FIRST" ] && exit 1
    fi
    exit 0 ;;
  *"python3 -"*) [ -n "${STUB_CLOCK:-}" ] && echo "$STUB_CLOCK"; exit 0 ;;
  *"pgrep -f"*) has gzdead && exit 1; exit 0 ;;
esac
echo "fake docker: 知らない呼び出し: $args" >&2
exit 99
"""

GOOD_CONFIG = {"services": {"ros": {"ports": [
    {"mode": "ingress", "host_ip": "127.0.0.1", "target": 80, "published": "6081", "protocol": "tcp"},
]}}}
GOOD_CLOCK = ("CLOCKRAW n=9771 w_first=100.000000 w_last=110.000000 w_end=110.001000 "
              "s_first=60.000000000 s_last=69.770000000 max_gap=0.004000")


class LocalCheckStubTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.d = Path(self._tmp.name)
        for name, body in (("docker", FAKE_DOCKER), ("curl", "#!/usr/bin/env bash\nexit 0\n")):
            p = self.d / name
            p.write_text(body, encoding="utf-8")
            p.chmod(p.stat().st_mode | stat.S_IXUSR)
        self.config = self.d / "config.json"
        self.write_config(GOOD_CONFIG)

    def tearDown(self):
        self._tmp.cleanup()

    def write_config(self, cfg):
        self.config.write_text(json.dumps(cfg), encoding="utf-8")

    def run_check(self, **env):
        e = dict(os.environ)
        e["PATH"] = f"{self.d}{os.pathsep}{e['PATH']}"
        e.update({"STUB_CONFIG": str(self.config), "STUB_DIGEST": declared_digest(),
                  "STUB_CLOCK": GOOD_CLOCK,
                  "STUB_JS_COUNTER": str(self.d / "js_counter"),
                  "LOCAL_CHECK_JS_INTERVAL_S": "0.1", "LOCAL_CHECK_JS_DEADLINE_S": "3"})
        e.update(env)
        r = subprocess.run(["bash", str(LOCAL_CHECK)], capture_output=True, text=True, env=e, timeout=60)
        lines = r.stdout.strip().splitlines()
        return r.returncode, (lines[-1] if lines else ""), r.stdout + r.stderr

    def assert_fail(self, rc, last, out, reason=None):
        self.assertTrue(last.startswith("LOCAL_CHECK: FAIL"), out)
        self.assertNotEqual(rc, 0, out)
        if reason:
            self.assertIn(reason, last, out)

    def test_pass_and_whole_window_rates(self):
        rc, last, out = self.run_check()
        self.assertEqual((rc, last), (0, "LOCAL_CHECK: PASS"), out)
        # 測定区間全体（10 秒）の受信周期と実時間比
        self.assertIn("clock_hz=977.0 rtf=0.977", out)

    # ① 全サービスの公開ポート
    def test_other_service_with_ports(self):
        cfg = json.loads(json.dumps(GOOD_CONFIG))
        cfg["services"]["web"] = {"ports": [{"target": 80, "published": "8080", "protocol": "tcp"}]}
        self.write_config(cfg)
        self.assert_fail(*self.run_check(), reason="ros 以外のサービス")

    def test_host_ip_omitted(self):
        cfg = json.loads(json.dumps(GOOD_CONFIG))
        del cfg["services"]["ros"]["ports"][0]["host_ip"]
        self.write_config(cfg)
        self.assert_fail(*self.run_check(), reason="loopback")

    def test_extra_port(self):
        cfg = json.loads(json.dumps(GOOD_CONFIG))
        cfg["services"]["ros"]["ports"].append(
            {"host_ip": "127.0.0.1", "target": 5900, "published": "5900", "protocol": "tcp"})
        self.write_config(cfg)
        self.assert_fail(*self.run_check())

    def test_runtime_port_all_interfaces(self):
        self.assert_fail(*self.run_check(STUB_FAIL="port"), reason="loopback")

    def test_runtime_two_containers(self):
        self.assert_fail(*self.run_check(STUB_IDS="c1 c2"), reason="1 つではない")

    # ② 異常終了・停止の失敗
    def test_config_command_fails(self):
        self.assert_fail(*self.run_check(STUB_FAIL="config"), reason="設定を取得できない")

    def test_config_unreadable(self):
        self.config.write_text("not json", encoding="utf-8")
        self.assert_fail(*self.run_check())

    def test_up_fails(self):
        self.assert_fail(*self.run_check(STUB_FAIL="up"))

    def test_gazebo_launch_fails(self):
        self.assert_fail(*self.run_check(STUB_FAIL="gazebo"), reason="Gazebo を起動できない")

    def test_gazebo_dies(self):
        self.assert_fail(*self.run_check(STUB_FAIL="gzdead"), reason="Gazebo が途中で落ちた")

    def test_down_fails(self):
        self.assert_fail(*self.run_check(STUB_FAIL="down"), reason="停止")

    def test_digest_mismatch(self):
        self.assert_fail(*self.run_check(STUB_DIGEST="0" * 64), reason="digest")

    def test_already_running_does_not_stop(self):
        rc, last, out = self.run_check(STUB_RUNNING="1", STUB_FAIL="down")
        # 既に動いていたら止めない（down を呼ばない）ので、理由は「既に起動している」だけ
        self.assert_fail(rc, last, out, reason="既に起動している")
        self.assertNotIn("停止", last)

    # ③ /clock の鮮度
    def test_clock_stops_midway(self):
        # 最後の 9.2 秒は受信が無い（最後に受けた区間だけなら rtf は 1 に見える）
        stale = ("CLOCKRAW n=801 w_first=100.000000 w_last=100.800000 w_end=110.000000 "
                 "s_first=60.000000000 s_last=60.800000000 max_gap=0.002000")
        self.assert_fail(*self.run_check(STUB_CLOCK=stale), reason="途中で止まった")

    def test_clock_gap_in_middle(self):
        gap = ("CLOCKRAW n=5000 w_first=100.000000 w_last=110.000000 w_end=110.001000 "
               "s_first=60.000000000 s_last=65.000000000 max_gap=5.000000")
        self.assert_fail(*self.run_check(STUB_CLOCK=gap), reason="途中で止まった")

    def test_clock_window_too_short(self):
        short = ("CLOCKRAW n=3000 w_first=100.000000 w_last=103.000000 w_end=103.001000 "
                 "s_first=60.000000000 s_last=63.000000000 max_gap=0.002000")
        self.assert_fail(*self.run_check(STUB_CLOCK=short), reason="途中で終わった")

    def test_clock_none(self):
        self.assert_fail(*self.run_check(STUB_CLOCK="CLOCKRAW n=0"), reason="届かない")

    def test_clock_no_output(self):
        self.assert_fail(*self.run_check(STUB_CLOCK=""), reason="測れない")

    def test_sim_time_not_advancing(self):
        frozen = ("CLOCKRAW n=9771 w_first=100.000000 w_last=110.000000 w_end=110.001000 "
                  "s_first=60.000000000 s_last=60.000000000 max_gap=0.004000")
        self.assert_fail(*self.run_check(STUB_CLOCK=frozen), reason="進まない")

    def test_joint_states_missing(self):
        self.assert_fail(*self.run_check(STUB_FAIL="joint_states"), reason="/joint_states")

    def test_joint_states_late_still_passes(self):
        # Gazebo の立ち上がりが遅く、最初の 15 回は「トピックがまだ無い」ですぐ失敗しても、待ち続けて PASS する
        # （以前は 12 回を間をあけずに試すだけだったので、ここで FAIL していた）
        rc, last, out = self.run_check(STUB_JS_FAIL_FIRST="15")
        self.assertEqual((rc, last), (0, "LOCAL_CHECK: PASS"), out)


if __name__ == "__main__":
    unittest.main()
