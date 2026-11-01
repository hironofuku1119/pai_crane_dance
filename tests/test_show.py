"""scripts/show.sh の検査。偽の docker を PATH の先頭に置いて動かす（Docker は起動しない）。"""
import fcntl
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOW = ROOT / "scripts" / "show.sh"

FAKE_DOCKER = r"""#!/usr/bin/env bash
echo "$*" >> "$FAKE_DOCKER_LOG"
case "$*" in
  *"pgrep -f crane_dance"*) [ -n "${FAKE_PLAYER_RUNNING:-}" ] && echo 4242; exit 0 ;;
  *"--ros-args"*) sleep "${FAKE_PLAYER_SLEEP:-0}"; exit 0 ;;
esac
case "$1" in
  inspect) echo "$FAKE_WORKING_DIR"; exit 0 ;;
  compose) exit 0 ;;
esac
exit 0
"""


class ShowTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        bindir = self.tmp / "bin"
        bindir.mkdir()
        fake = bindir / "docker"
        fake.write_text(FAKE_DOCKER, encoding="utf-8")
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        self.log = self.tmp / "docker.log"
        self.log.write_text("", encoding="utf-8")
        self.env = dict(os.environ)
        self.env["PATH"] = f"{bindir}{os.pathsep}{self.env['PATH']}"
        self.env["FAKE_DOCKER_LOG"] = str(self.log)
        self.env["FAKE_WORKING_DIR"] = str(self.tmp / "elsewhere" / "other_folder")
        self.lock = self.tmp / "show.lock"
        self.env["SHOW_LOCK_FILE"] = str(self.lock)

    def run_show(self, *args):
        return subprocess.run(["bash", str(SHOW), *args], capture_output=True, text=True,
                              env=self.env, timeout=60, stdin=subprocess.DEVNULL)

    def calls(self):
        return self.log.read_text(encoding="utf-8")

    def test_syntax(self):
        r = subprocess.run(["bash", "-n", str(SHOW)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_no_args(self):
        r = self.run_show()
        self.assertEqual(r.returncode, 1)
        self.assertEqual(len(r.stdout.strip().splitlines()), 1)
        self.assertIn("使い方", r.stdout)
        self.assertNotIn("compose up", self.calls())

    def test_npz_outside_repo(self):
        outside = self.tmp / "out.npz"
        outside.write_bytes(b"x")
        r = self.run_show("npz", str(outside))
        self.assertEqual(r.returncode, 1)
        self.assertEqual(len(r.stdout.strip().splitlines()), 1)
        self.assertIn("repo の外", r.stdout)
        self.assertNotIn("compose up", self.calls())

    def test_other_working_dir(self):
        r = self.run_show("choreography")
        self.assertEqual(r.returncode, 1)
        self.assertEqual(len(r.stdout.strip().splitlines()), 1)
        self.assertIn("別の場所", r.stdout)
        self.assertIn("other_folder", r.stdout)
        self.assertNotIn(str(self.tmp), r.stdout)
        self.assertIn("inspect", self.calls())
        self.assertNotIn("compose up", self.calls())

    def test_refuses_when_another_player_runs(self):
        # 同じ checkout の別の再生が動いていたら、止めずに 1 行で知らせて終わる（pkill を呼ばない）
        self.env["FAKE_WORKING_DIR"] = str(ROOT)
        self.env["FAKE_PLAYER_RUNNING"] = "1"
        self.env["SHOW_JS_INTERVAL_S"] = "0"
        r = self.run_show("choreography")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("別の再生", r.stdout.strip().splitlines()[-1])
        self.assertNotIn("pkill", self.calls())
        self.assertNotIn("--ros-args", self.calls())  # player を起動していない

    def test_starts_own_player_with_run_id_and_kills_nothing(self):
        self.env["FAKE_WORKING_DIR"] = str(ROOT)
        self.env["SHOW_JS_INTERVAL_S"] = "0"
        r = self.run_show("choreography")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        calls = self.calls()
        self.assertIn("--ros-args", calls)
        self.assertIn("run_id:=show", calls)
        self.assertNotIn("pkill", calls)


    def test_refuses_while_lock_is_held(self):
        # 別の show.sh がロックを持っている間は、何も起動せず・何も止めずに 1 行で終わる
        self.env["FAKE_WORKING_DIR"] = str(ROOT)
        with open(self.lock, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            r = self.run_show("choreography")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertEqual(len(r.stdout.strip().splitlines()), 1)
        self.assertIn("別の show.sh", r.stdout)
        calls = self.calls()
        self.assertNotIn("compose up", calls)
        self.assertNotIn("colcon build", calls)
        self.assertNotIn("--ros-args", calls)
        self.assertNotIn("pkill", calls)

    def test_two_at_once_start_only_one_player(self):
        # 同時に二つ起動しても player を起動するのは一つだけ。拒否された側は所有者を止めない
        self.env["FAKE_WORKING_DIR"] = str(ROOT)
        self.env["SHOW_JS_INTERVAL_S"] = "0"
        self.env["FAKE_PLAYER_SLEEP"] = "3"
        procs = [subprocess.Popen(["bash", str(SHOW), "choreography"], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, env=self.env, stdin=subprocess.DEVNULL)
                 for _ in range(2)]
        outs = [p.communicate(timeout=60) for p in procs]
        codes = sorted(p.returncode for p in procs)
        self.assertEqual(codes, [0, 1], outs)
        refused = [o for p, o in zip(procs, outs) if p.returncode == 1][0][0]
        self.assertIn("別の show.sh", refused)
        calls = self.calls()
        self.assertEqual(calls.count("--ros-args"), 1)
        self.assertNotIn("pkill", calls)

    def test_lock_is_released_when_show_ends(self):
        # 再生が終われば外れる（取り残しの掃除は要らない）：続けて 2 回動かせる
        self.env["FAKE_WORKING_DIR"] = str(ROOT)
        self.env["SHOW_JS_INTERVAL_S"] = "0"
        for _ in range(2):
            r = self.run_show("choreography")
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.calls().count("--ros-args"), 2)


if __name__ == "__main__":
    unittest.main()
