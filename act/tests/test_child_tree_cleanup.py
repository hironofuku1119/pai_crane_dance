"""子プロセスの木の所有と回収。

本番と同じ二段の起動（外側の run_streamed → 子の Python → 子の中の run_streamed → 孫）を本物のプロセスで作り、
外側が止まったときに子と孫の両方が終わってから呼び出し元へ戻ることを確かめる。学習は走らせない。
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from crane_dance_act import train

ACT_DIR = Path(train.__file__).resolve().parents[1]
LIMIT_SEC = 20


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_gone(pid, sec=5.0):
    end = time.monotonic() + sec
    while time.monotonic() < end:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


def _child_argv(tmp_path, ignore_term=False, child_exits=False):
    """子：孫（pid をファイルに書いて眠る）を run_streamed で起動し、孫の出力を自分の標準出力へ流す。"""
    gc_pid = tmp_path / "gc.pid"
    trap = "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_term else ""
    grandchild = f"{trap}import os,time; open({str(gc_pid)!r},'w').write(str(os.getpid())); print('gc-ready', flush=True); time.sleep(60)"
    if child_exits:
        grandchild = f"import os; open({str(gc_pid)!r},'w').write(str(os.getpid())); print('gc-ready', flush=True)"
    child = (
        "import sys; sys.path.insert(0, {act!r}); from crane_dance_act import train;"
        "rc = train.run_streamed([sys.executable, '-c', {gc!r}], {log!r}, train.LogMasker(home='/fakehome/x'),"
        " on_masked=lambda m: print(m, end='', flush=True)); print('child-done', rc, flush=True)"
    ).format(act=str(ACT_DIR), gc=grandchild, log=str(tmp_path / "inner.log"))
    return [sys.executable, "-c", child], gc_pid


@pytest.fixture
def procs(monkeypatch):
    got = []
    real = subprocess.Popen

    def rec(*a, **k):
        p = real(*a, **k)
        got.append(p)
        return p

    monkeypatch.setattr(subprocess, "Popen", rec)
    monkeypatch.setattr(train, "STOP_WAIT_SEC", 1)
    monkeypatch.delenv(train.PGROUP_ENV, raising=False)  # このテスト自体は一番外側として動く
    yield got
    for p in got:
        if p.poll() is None:
            p.kill()
            p.wait()


@pytest.fixture
def bystander():
    """別の実行（グループの外の無関係なプロセス）。巻き込まれて止まらないことを見る。"""
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    yield p
    p.kill()
    p.wait()


def _raise_on(word, exc):
    def cb(masked):
        if word in masked:
            raise exc
    return cb


@pytest.mark.parametrize("exc", [RuntimeError("callback boom"), KeyboardInterrupt()], ids=["callback", "interrupt"])
@pytest.mark.parametrize("ignore_term", [False, True], ids=["normal", "ignores-sigterm"])
def test_outer_failure_stops_child_and_grandchild(tmp_path, procs, bystander, exc, ignore_term):
    argv, gc_pid = _child_argv(tmp_path, ignore_term=ignore_term)
    t0 = time.monotonic()
    with pytest.raises(type(exc)):
        train.run_streamed(argv, tmp_path / "outer.log", train.LogMasker(home="/fakehome/x"),
                           on_masked=_raise_on("gc-ready", exc))
    assert time.monotonic() - t0 < LIMIT_SEC
    child = [q for q in procs if q is not bystander]
    assert len(child) == 1 and child[0].poll() is not None, "子が残っている"
    pid = int(gc_pid.read_text())
    assert _wait_gone(pid), "孫が残っている"
    assert bystander.poll() is None, "無関係なプロセスまで止めた"


def test_normal_end_returns_code_and_leaves_nothing(tmp_path, procs, bystander):
    argv, gc_pid = _child_argv(tmp_path, child_exits=True)
    seen = []
    rc = train.run_streamed(argv, tmp_path / "outer.log", train.LogMasker(home="/fakehome/x"), on_masked=seen.append)
    assert rc == 0
    text = "".join(seen)
    assert "gc-ready" in text and "child-done 0" in text
    assert _wait_gone(int(gc_pid.read_text()))
    assert bystander.poll() is None


def test_inner_level_does_not_make_a_new_group(tmp_path, procs, monkeypatch):
    """目印が立っているプロセスの中では新しいグループを作らない（孫を外側のグループから逃がさない）。"""
    monkeypatch.setenv(train.PGROUP_ENV, "1")
    rc = train.run_streamed([sys.executable, "-c", "import os; print(os.getpgrp() == os.getpgid(os.getppid()))"],
                            tmp_path / "c.log", train.LogMasker(home="/fakehome/x"))
    assert rc == 0
    assert (tmp_path / "c.log").read_text(encoding="utf-8").strip() == "True"


def test_outer_level_makes_a_new_group(tmp_path, procs):
    rc = train.run_streamed([sys.executable, "-c", "import os; print(os.getpgrp() == os.getpid())"],
                            tmp_path / "c.log", train.LogMasker(home="/fakehome/x"))
    assert rc == 0
    assert (tmp_path / "c.log").read_text(encoding="utf-8").strip() == "True"
