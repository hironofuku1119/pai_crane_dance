"""子プロセスの所有と回収: 起動後の例外・割込みでも、子の終了を確認してから例外を投げ直す。"""
import subprocess
import sys
import time

import pytest

from crane_dance_act import batch, train

HOME = "/fakehome/bobsmith"
SLEEPER = [sys.executable, "-c", "import time,sys; print('x', flush=True); time.sleep(60)"]
LIMIT_SEC = 20  # 60 秒眠りを待たずに戻る目安


@pytest.fixture
def procs(monkeypatch):
    """subprocess.Popen を記録する包み。起動した proc を集める（掃除も保証）。"""
    got = []
    real = subprocess.Popen

    def rec(*a, **k):
        p = real(*a, **k)
        got.append(p)
        return p

    monkeypatch.setattr(subprocess, "Popen", rec)
    yield got
    for p in got:
        if p.poll() is None:
            p.kill()
            p.wait()


class BadFeedMasker:
    """feed が OSError を投げる偽物（ログ処理の失敗）。"""

    def feed(self, line):
        raise OSError("disk full")

    def close(self):
        return ""


def _plain_masker():
    return train.LogMasker(home=HOME)


def _on_line_runtime(_ln):
    raise RuntimeError("callback boom")


def _on_line_interrupt(_ln):
    raise KeyboardInterrupt


CASES = [
    ("callback", _plain_masker, _on_line_runtime, RuntimeError),
    ("interrupt", _plain_masker, _on_line_interrupt, KeyboardInterrupt),
    ("log_write", BadFeedMasker, None, OSError),
]


def _assert_child_gone(procs, t0):
    assert len(procs) == 1
    assert procs[0].poll() is not None, "子が残っている"
    assert time.monotonic() - t0 < LIMIT_SEC


@pytest.mark.parametrize("name,mk,cb,exc", CASES, ids=[c[0] for c in CASES])
def test_batch_run_child_stops_child_on_error(tmp_path, procs, name, mk, cb, exc):
    t0 = time.monotonic()
    with pytest.raises(exc):
        batch.run_child(SLEEPER, tmp_path / "c.log", mk(), on_line=cb)
    _assert_child_gone(procs, t0)


@pytest.mark.parametrize("name,mk,cb,exc", CASES, ids=[c[0] for c in CASES])
def test_train_run_streamed_stops_child_on_error(tmp_path, procs, name, mk, cb, exc):
    t0 = time.monotonic()
    with pytest.raises(exc):
        train.run_streamed(SLEEPER, tmp_path / "c.log", mk(), on_masked=cb)
    _assert_child_gone(procs, t0)


def test_original_exception_not_hidden_by_stop_failure(tmp_path, procs, monkeypatch):
    # _stop_child 自体が内部で失敗しても元の例外が出る（terminate が壊れても握りつぶす）
    real = subprocess.Popen

    class Broken:
        def __init__(self, *a, **k):
            self._p = real(*a, **k)
            procs.append(self._p)
            self.stdout = self._p.stdout

        def poll(self):
            return self._p.poll()

        def terminate(self):
            raise PermissionError("no terminate")

        def kill(self):
            self._p.kill()

        def wait(self, timeout=None):
            return self._p.wait(timeout=timeout)

    monkeypatch.setattr(subprocess, "Popen", Broken)
    with pytest.raises(RuntimeError, match="callback boom"):
        train.run_streamed(SLEEPER, tmp_path / "c.log", _plain_masker(), on_masked=_on_line_runtime)
    assert procs[0].poll() is not None


def _secret_child(extra=""):
    return [
        sys.executable, "-c",
        f"import sys;print('path {HOME}/work/data.npz');print('API_KEY=top-secret-value');print('last line'){extra}",
    ]  # fmt: skip


def _assert_masked(saved):
    for secret in (HOME, "bobsmith", "top-secret-value"):
        assert secret not in saved
    assert "last line" in saved


def test_batch_normal_run_keeps_masking_and_return(tmp_path):
    seen = []
    rc, last = batch.run_child(_secret_child(";sys.exit(3)"), tmp_path / "c.log", _plain_masker(), on_line=seen.append)
    assert rc == 3 and last == "last line"
    _assert_masked((tmp_path / "c.log").read_text(encoding="utf-8"))
    assert seen[-1] == "last line" and all("top-secret-value" not in s for s in seen)


def test_train_normal_run_keeps_masking_and_return(tmp_path):
    seen = []
    rc = train.run_streamed(_secret_child(";sys.exit(3)"), tmp_path / "c.log", _plain_masker(), on_masked=seen.append)
    assert rc == 3
    _assert_masked((tmp_path / "c.log").read_text(encoding="utf-8"))
    assert all("top-secret-value" not in s for s in seen)


def test_run_training_goes_through_run_streamed(tmp_path, monkeypatch):
    """run_training が中核 run_streamed を通ること（学習側の経路）。重い準備なしに、検査を通して中核を偽物に差し替える。"""
    monkeypatch.setattr(train, "preflight", lambda *a, **k: [])
    monkeypatch.setattr(train, "check_device_env", lambda *a, **k: [])
    monkeypatch.setattr(train, "probe_accelerate_device", lambda *a, **k: "cpu")
    monkeypatch.setattr(train.state_mode_mod, "read_manifest_state_mode", lambda root: "x")
    monkeypatch.setattr(train, "load_contract", lambda p: {})
    called = {}

    def fake(argv, log_path, masker, on_masked=None, env=None):
        called["argv"] = argv
        called["on_masked"] = on_masked
        raise RuntimeError("via run_streamed")

    monkeypatch.setattr(train, "run_streamed", fake)
    spec = train.TrainSpec(dataset_root=tmp_path / "ds", repo_id="local/x", output_dir=tmp_path / "out", steps=2, save_freq=2, device="cpu")
    with pytest.raises(RuntimeError, match="via run_streamed"):
        train.run_training(spec)
    assert called["argv"][0].endswith("lerobot-train") and callable(called["on_masked"])
