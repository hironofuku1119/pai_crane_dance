import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from crane_dance_act import synth, to_lerobot, train

ACT_DIR = pathlib.Path(__file__).resolve().parents[1]
CONTRACT_PATH = ACT_DIR.parent / "contract" / "contract.json"
CHOREO_PATH = ACT_DIR.parent / "contract" / "choreography.json"
REPO_ID = "local/crane_dance_test"
TRUE = lambda: True  # noqa: E731
FALSE = lambda: False  # noqa: E731


@pytest.fixture(scope="module")
def contract():
    return synth.load_contract(CONTRACT_PATH)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory, contract):
    d = tmp_path_factory.mktemp("synth") / "synth"
    choreo = synth.load_choreography(CHOREO_PATH)
    synth.write_varied_set(contract, choreo, synth.file_sha256(CHOREO_PATH), 2, 0, d)
    root = tmp_path_factory.mktemp("lr") / "ds"
    to_lerobot.convert(sorted(d.glob("ep*.npz")), contract, CHOREO_PATH, root, REPO_ID, d / "manifest.json")
    return root


def _cmd(device="mps", num_workers=0):
    return train.build_command("/x/ds", REPO_ID, "/x/out", 10, 8, 1000, 10, num_workers, device)


def test_build_command_fixed_flags():
    c = _cmd()
    for f in (
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
        "--policy.chunk_size=50",
        "--policy.n_action_steps=50",
        "--policy.type=act",
        "--num_workers=0",
        "--steps=10",
        "--seed=1000",
        f"--dataset.repo_id={REPO_ID}",
    ):
        assert f in c
    assert "--num_workers=2" in _cmd(num_workers=2)
    assert not any(a.startswith("--policy.device") for a in c)
    assert "--policy.device=cpu" in _cmd("cpu")


def test_build_env_offline():
    assert train.build_env({"HF_HUB_OFFLINE": "0", "A": "b"})["HF_HUB_OFFLINE"] == "1"
    assert train.build_env({})["HF_HUB_OFFLINE"] == "1"


def test_preflight_ok_and_cpu_skips_mps(dataset, contract):
    assert train.preflight(dataset, contract, CHOREO_PATH, "mps", mps_available=TRUE) == []
    assert train.preflight(dataset, contract, CHOREO_PATH, "cpu", mps_available=FALSE) == []


def test_preflight_no_mps(dataset, contract):
    errs = train.preflight(dataset, contract, CHOREO_PATH, "mps", mps_available=FALSE)
    assert any("MPS" in e for e in errs)


def test_preflight_low_std(dataset, contract, tmp_path):
    from lerobot.datasets.io_utils import load_stats, write_stats

    root = tmp_path / "ds"
    shutil.copytree(dataset, root)
    stats = load_stats(root)
    stats["action"]["std"] = stats["action"]["std"] * 0 + 0.001
    write_stats(stats, root)
    errs = train.preflight(root, contract, CHOREO_PATH, "cpu")
    assert any("action" in e and "下限" in e for e in errs)


@pytest.mark.parametrize("key", ["observation.state", "action"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "+inf", "-inf"])
def test_preflight_rejects_nonfinite_std(dataset, contract, tmp_path, key, bad):
    # NaN は大小比較をすり抜け、+inf は下限を満たしてしまう → 有限性を先に見る
    from lerobot.datasets.io_utils import load_stats, write_stats

    root = tmp_path / "ds"
    shutil.copytree(dataset, root)
    stats = load_stats(root)
    for j in range(len(stats[key]["std"])):
        s = load_stats(dataset)
        s[key]["std"][j] = bad
        write_stats(s, root)
        errs = train.preflight(root, contract, CHOREO_PATH, "cpu")
        assert any(key in e and "有限" in e for e in errs), (key, j, bad)


def test_check_device_env():
    # Accelerate の装置を上書きする環境変数
    assert train.check_device_env({}, "mps") == []
    assert train.check_device_env({"ACCELERATE_USE_CPU": "1"}, "mps")
    assert train.check_device_env({"ACCELERATE_USE_CPU": "true"}, "mps")
    assert train.check_device_env({"ACCELERATE_USE_CPU": "0"}, "mps") == []
    assert train.check_device_env({"ACCELERATE_TORCH_DEVICE": "cpu"}, "mps")
    assert train.check_device_env({"ACCELERATE_TORCH_DEVICE": "mps"}, "mps") == []
    assert train.check_device_env({"ACCELERATE_USE_CPU": "1"}, "cpu") == []
    # cpu 指定でも ACCELERATE_TORCH_DEVICE=mps は Accelerator(cpu=True) より優先される
    assert train.check_device_env({"ACCELERATE_TORCH_DEVICE": "mps"}, "cpu")
    assert train.check_device_env({"ACCELERATE_TORCH_DEVICE": "cpu"}, "cpu") == []


def test_resolve_actual_device():
    assert train.resolve_actual_device("mps", "mps", "mps") == ("mps", [])
    assert train.resolve_actual_device("mps", "mps", "unknown") == ("mps", [])
    a, e = train.resolve_actual_device("mps", "cpu", "mps")
    assert a == "cpu" and e
    a, e = train.resolve_actual_device("mps", "unknown", "cpu")
    assert a == "cpu" and e
    a, e = train.resolve_actual_device("mps", "unknown", "unknown")
    assert a == "unknown" and e
    assert train.resolve_actual_device("cpu", "cpu", "cpu") == ("cpu", [])


def test_probe_follows_env():
    # 本物の Accelerate に聞く：CPU 強制の変数があれば cpu を返す
    env = train.build_env(os.environ)
    env["ACCELERATE_USE_CPU"] = "1"
    assert train.probe_accelerate_device(env) == "cpu"
    # cpu=True（lerobot の --policy.device=cpu と同じ）でも、ACCELERATE_TORCH_DEVICE=mps なら mps が選ばれる＝決め打ちできない
    env2 = train.build_env(os.environ)
    env2.pop("ACCELERATE_USE_CPU", None)
    env2["ACCELERATE_TORCH_DEVICE"] = "mps"
    if train._mps_available():
        assert train.probe_accelerate_device(env2, cpu=True) == "mps"
    env3 = train.build_env(os.environ)
    env3.pop("ACCELERATE_TORCH_DEVICE", None)
    assert train.probe_accelerate_device(env3, cpu=True) == "cpu"


def test_train_act_rejects_mps_forcing_env_for_cpu(dataset, tmp_path):
    # CPU 指定＋MPS 強制の env は、学習を起動する前に失敗する（MPS の学習を CPU として記録しない）
    out = tmp_path / "run"
    r = subprocess.run(
        [sys.executable, str(ACT_DIR / "scripts" / "train_act.py"), "--dataset-root", str(dataset),
         "--repo-id", REPO_ID, "--output-dir", str(out), "--steps", "2", "--save-freq", "2", "--device", "cpu"],
        capture_output=True, text=True, env={**os.environ, "HF_HUB_OFFLINE": "1", "ACCELERATE_TORCH_DEVICE": "mps"}, timeout=300,
    )  # fmt: skip
    assert r.returncode != 0
    assert "PREFLIGHT NG" in r.stderr
    assert not out.exists() and not (tmp_path / "run.log").exists()


def test_resolve_rejects_cpu_request_with_mps_actual():
    a, e = train.resolve_actual_device("cpu", "mps", "cpu")
    assert a == "mps" and e


def test_train_act_rejects_cpu_forcing_env_for_mps(dataset, tmp_path):
    # MPS 要求＋CPU 強制の env は、学習を起動する前に失敗する（出力先もログも作らない）
    out = tmp_path / "run"
    for extra in ({"ACCELERATE_USE_CPU": "1"}, {"ACCELERATE_TORCH_DEVICE": "cpu"}):
        r = subprocess.run(
            [sys.executable, str(ACT_DIR / "scripts" / "train_act.py"), "--dataset-root", str(dataset),
             "--repo-id", REPO_ID, "--output-dir", str(out), "--steps", "2", "--save-freq", "2", "--device", "mps"],
            capture_output=True, text=True, env={**os.environ, "HF_HUB_OFFLINE": "1", **extra}, timeout=300,
        )  # fmt: skip
        assert r.returncode != 0
        assert "PREFLIGHT NG" in r.stderr
        assert not out.exists() and not (tmp_path / "run.log").exists()


def test_mask_log():
    # 模擬のホーム・利用者名・データセットと出力先の実パス・秘密値が残らず、loss と装置の解析は保たれる
    home = "/Users/fakeuser123"
    ds = f"{home}/work/act/out/lerobot/ds_x"
    out = f"{home}/work/act/out/train/run_x"
    raw = (
        f"INFO 'dataset': {{'root': '{ds}'}} 'output_dir': '{out}'\n"
        f"objc: Class X is implemented in both {home}/work/.venv/lib/a.dylib and -Users-fakeuser123-cache\n"
        "user fakeuser123 started; HF_TOKEN=hf_abcdef MY_API_KEY='q w e' bearer sk-123 --token tok456\n"
        "WARNING:lerobot.configs.policies:Device 'None' is not available. Switching to 'mps'.\n"
        "step:50 smpl:400 loss:13.530 grdn:2 lr:1e-5 updt_s:0.050 data_s:0.008\n"
        "step:100 smpl:800 loss:4.329 updt_s:0.046 data_s:0.008\n"
    )
    named = ((ds, "<dataset_root:ds_x>"), (out, "<output_dir:run_x>"))
    m = train.mask_log(raw, home=home, named_paths=named)
    for leak in (home, "fakeuser123", ds, out, "hf_abcdef", "q w e", "sk-123", "tok456", "-Users-fakeuser123"):
        assert leak not in m, leak
    assert "<dataset_root:ds_x>" in m and "<output_dir:run_x>" in m
    assert train.detect_device(m) == "mps"
    assert train.parse_log(m) == train.parse_log(raw)
    assert train.mask_log(m, home=home, named_paths=named) == m  # 何度通しても同じ


def _feed_all(masker, text):
    out = "".join(masker.feed(line) for line in text.splitlines(keepends=True))
    return out + masker.close()


def test_log_masker_multiline_quoted_secrets():
    # 引用符つきの秘密値が改行・空白・エスケープ引用符を含んでも、一部も残らない
    home = "/Users/fakeuser123"
    raw = (
        "WARNING:lerobot.configs.policies:Device 'None' is not available. Switching to 'mps'.\n"
        'MY_API_KEY="part1 alpha\npart2 \\"beta\\" gamma\npart3-end" next=ok\n'
        "HF_TOKEN='tok-one\ntok-two'\n"
        'bearer "bear-a\nbear-b"\n'
        '--password "pw-a\n pw-b"\n'
        "step:50 smpl:400 loss:13.530 grdn:2 lr:1e-5 updt_s:0.050 data_s:0.008\n"
        "step:100 smpl:800 loss:4.329 updt_s:0.046 data_s:0.008\n"
    )
    m = _feed_all(train.LogMasker(home=home), raw)
    for leak in ("part1", "alpha", "part2", "beta", "gamma", "part3-end", "tok-one", "tok-two", "bear-a", "bear-b", "pw-a", "pw-b"):
        assert leak not in m, leak
    assert "next=ok" in m
    assert train.detect_device(m) == "mps"
    assert train.parse_log(m) == train.parse_log(raw)


def test_log_masker_unclosed_quote_redacts_rest():
    # 最後まで閉じない引用は、開始の引用符より後ろを全部伏せる（ためすぎの上限でも同じ）
    raw = 'step:10 loss:1.000 updt_s:0.1 data_s:0.0\nSECRET_X="never closed\nleak-1\nleak-2\n'
    m = _feed_all(train.LogMasker(home="/Users/fakeuser123"), raw)
    assert "never" not in m and "leak-1" not in m and "leak-2" not in m
    assert "loss:1.000" in m
    # ためすぎ（MAX_HOLD）で先に伏せた後も、引用が閉じるまでの続きの行は残らない・閉じた後ろは通常どおり
    mk = train.LogMasker(home="/Users/fakeuser123")
    mk.MAX_HOLD = 100
    out = mk.feed('A_TOKEN="xsecret\n') + "".join(mk.feed("leak-zz " * 5 + "\n") for _ in range(5))
    out += mk.feed('leak-end" after-close loss:0.500\n') + mk.close()
    assert "xsecret" not in out and "leak-zz" not in out and "leak-end" not in out
    assert "<redacted>" in out and "after-close loss:0.500" in out


def test_preflight_sha_mismatch(dataset, contract, tmp_path):
    root = tmp_path / "ds"
    shutil.copytree(dataset, root)
    m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    m["choreography_sha256"] = "0" * 64
    (root / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    errs = train.preflight(root, contract, CHOREO_PATH, "cpu")
    assert any("sha256" in e for e in errs)


def test_run_record_env_whitelist():
    env = {"HF_HUB_OFFLINE": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "1", "HF_TOKEN": "hf_secret", "MY_API_SECRET": "zzz"}
    cmd = train.build_command("/srv/zz/x/ds", REPO_ID, "/srv/zz/x/out", 10, 8, 1000, 10, 0, "mps")
    r = train.run_record(cmd, 1000, "abc", "/srv/zz/x/ds", REPO_ID, "mps", "unknown", env, 10, 8)
    assert set(r["env"]) <= set(train.ALLOWED_ENV_KEYS)
    assert "hf_secret" not in json.dumps(r) and "zzz" not in json.dumps(r)
    assert r["dataset_root_name"] == "ds"
    assert "/srv/zz" not in json.dumps(r)
    assert r["device_actual"] == "unknown"


def test_parse_log_and_device():
    log = "WARNING:lerobot.configs.policies:Device 'None' is not available. Switching to 'mps'.\n"
    assert train.detect_device(log) == "mps"
    assert train.detect_device("nothing here") == "unknown"
    assert train.detect_device("  'device': 'cpu',") == "cpu"
    assert "--log_freq=50" in _cmd()
    l2 = "step:10 smpl:80 loss:1.500 grdn:3 lr:1e-5 updt_s:0.100 data_s:0.010\nstep:20 loss:0.500 updt_s:0.200 data_s:0.020\n"
    first, last, sec = train.parse_log(l2)
    assert (first, last) == (1.5, 0.5)
    assert sec == pytest.approx(0.15 + 0.015)
    assert train.parse_log("") == (None, None, None)


@pytest.mark.skipif(os.environ.get("ACT_SKIP_SLOW") == "1", reason="ACT_SKIP_SLOW=1")
def test_train_act_cpu_two_steps(dataset, tmp_path):
    out = tmp_path / "run"
    r = subprocess.run(
        [
            sys.executable, str(ACT_DIR / "scripts" / "train_act.py"),
            "--dataset-root", str(dataset), "--repo-id", REPO_ID, "--output-dir", str(out),
            "--steps", "2", "--save-freq", "2", "--num-workers", "0", "--device", "cpu", "--batch-size", "2",
        ],
        capture_output=True, text=True, env={**os.environ, "HF_HUB_OFFLINE": "1"}, timeout=600,
    )  # fmt: skip
    assert r.returncode == 0, r.stdout + r.stderr
    assert train.post_check(out) == []
    assert (out / "run.json").exists()
    assert json.loads((out / "run.json").read_text(encoding="utf-8"))["device_actual"] == "cpu"
    assert (tmp_path / "run.log").exists()
    # 保存したログに実パス・利用者名が残らない（伏せてから保存）
    saved = (tmp_path / "run.log").read_text(encoding="utf-8")
    home = os.path.expanduser("~")
    assert home not in saved and str(out) not in saved and str(dataset) not in saved
    assert os.path.basename(home) not in saved
    assert train.find_pretrained_dirs(out)
    again = subprocess.run(
        [sys.executable, str(ACT_DIR / "scripts" / "train_act.py"), "--dataset-root", str(dataset),
         "--output-dir", str(out), "--steps", "2", "--save-freq", "2", "--device", "cpu"],
        capture_output=True, text=True,
    )  # fmt: skip
    assert again.returncode != 0
