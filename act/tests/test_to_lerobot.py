import json
import pathlib
import shutil

import numpy as np
import pytest

from crane_dance_act import synth, to_lerobot

ACT_DIR = pathlib.Path(__file__).resolve().parents[1]
CONTRACT_PATH = ACT_DIR.parent / "contract" / "contract.json"
CHOREO_PATH = ACT_DIR.parent / "contract" / "choreography.json"
REPO_ID = "local/crane_dance_test"
K = 3


@pytest.fixture(scope="module")
def contract():
    return synth.load_contract(CONTRACT_PATH)


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory, contract):
    d = tmp_path_factory.mktemp("synth") / "synth"
    choreo = synth.load_choreography(CHOREO_PATH)
    synth.write_varied_set(contract, choreo, synth.file_sha256(CHOREO_PATH), K, 0, d)
    return d


@pytest.fixture(scope="module")
def converted(tmp_path_factory, contract, synth_dir):
    root = tmp_path_factory.mktemp("lr") / "ds"
    paths = sorted(synth_dir.glob("ep*.npz"))
    out = to_lerobot.convert(paths, contract, CHOREO_PATH, root, REPO_ID, synth_dir / "manifest.json")
    return out, paths


def _npz(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def test_offline_is_fixed():
    import huggingface_hub.constants as c

    assert c.HF_HUB_OFFLINE is True


def test_roundtrip_from_disk(converted, contract):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out, paths = converted
    ds = LeRobotDataset(REPO_ID, root=out)
    eps = [_npz(p) for p in paths]
    assert ds.num_episodes == K
    assert ds.num_frames == sum(len(e["timestamp"]) for e in eps)
    assert ds.fps == contract["recording"]["fps"]
    feats = ds.meta.features
    names = list(contract["joint_names"])
    assert tuple(feats["observation.state"]["shape"]) == (5,)
    assert tuple(feats["observation.environment_state"]["shape"]) == (3,)
    assert tuple(feats["action"]["shape"]) == (5,)
    assert list(feats["observation.state"]["names"]) == names
    assert list(feats["action"]["names"]) == names
    assert list(feats["observation.environment_state"]["names"]) == to_lerobot.ENV_NAMES

    fps = ds.fps
    offset = 0
    for i, e in enumerate(eps):
        n = len(e["timestamp"])
        for k in (0, 1, n // 2, n - 1):
            item = ds[offset + k]
            assert int(item["episode_index"]) == i
            assert int(item["frame_index"]) == k
            assert abs(float(item["timestamp"]) - k / fps) <= 1e-5
            assert np.array_equal(item["observation.state"].numpy(), e["observation_state"][k])
            assert np.array_equal(item["observation.environment_state"].numpy(), e["observation_environment_state"][k])
            assert np.array_equal(item["action"].numpy(), e["action"][k])
        offset += n


def test_admission_and_overwrite_and_cleanup(tmp_path, contract, synth_dir):
    paths = sorted(synth_dir.glob("ep*.npz"))
    # sha 混在
    bad = tmp_path / "bad.npz"
    e = _npz(paths[0])
    e["choreography_sha256"] = np.str_("0" * 64)
    np.savez(bad, **e)
    root = tmp_path / "o1" / "ds"
    with pytest.raises(ValueError):
        to_lerobot.convert([paths[0], bad], contract, CHOREO_PATH, root, REPO_ID)
    # 全部が今の振付と違う sha
    with pytest.raises(ValueError):
        to_lerobot.convert([bad], contract, CHOREO_PATH, root, REPO_ID)
    # manifest の sha 不一致
    m = json.loads((synth_dir / "manifest.json").read_text(encoding="utf-8"))
    m["choreography_sha256"] = "1" * 64
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError):
        to_lerobot.convert(paths, contract, CHOREO_PATH, root, REPO_ID, mp)
    assert not root.exists()
    # 正式な出力先が既にある
    root.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        to_lerobot.convert(paths, contract, CHOREO_PATH, root, REPO_ID)
    # 形の壊れた npz は、出力を作る前に止まる
    broken = tmp_path / "broken.npz"
    e = _npz(paths[1])
    e["action"] = e["action"][:, :4]
    np.savez(broken, **e)
    root2 = tmp_path / "o2" / "ds"
    with pytest.raises(ValueError):
        to_lerobot.convert([paths[0], broken], contract, CHOREO_PATH, root2, REPO_ID)
    assert not root2.parent.exists()


def test_failure_midway_leaves_nothing(tmp_path, contract, synth_dir, monkeypatch):
    # 変換の途中（finalize の後・rename の前）で例外 → 一時フォルダも正式なフォルダも残らない
    paths = sorted(synth_dir.glob("ep*.npz"))

    def boom(*a, **k):
        raise RuntimeError("途中で失敗")

    monkeypatch.setattr(to_lerobot, "apply_std_floor", boom)
    root = tmp_path / "o3" / "ds"
    with pytest.raises(RuntimeError):
        to_lerobot.convert(paths[:1], contract, CHOREO_PATH, root, REPO_ID)
    assert not root.exists()
    assert list(root.parent.iterdir()) == []


def _mut_fps25(contract, choreo, sha):
    # 同じ振付 sha のまま 25 Hz・N=780 で作った（自分の中では辻褄の合う）npz
    c25 = json.loads(json.dumps(contract))
    c25["recording"]["fps"] = 25
    return synth.build_episode(c25, choreo, sha)


def _mut(key):
    def f(contract, choreo, sha):
        e = synth.build_episode(contract, choreo, sha)
        if key == "version":
            e["contract_version"] = np.str_("0.0.9")
        elif key == "timestamp":
            e["timestamp"] = e["timestamp"].copy()
            e["timestamp"][100] += 1e-3
        elif key == "n":
            for k in synth.DATA_KEYS:
                e[k] = e[k][:-1]
        elif key == "duration":
            e["duration_s"] = np.float64(float(e["duration_s"]) + 0.5)
        elif key == "beats":
            e["beat_times_s"] = e["beat_times_s"] + 0.01
        # NaN
        elif key == "nan_duration":
            e["duration_s"] = np.float64("nan")
        elif key == "nan_beats":
            e["beat_times_s"] = e["beat_times_s"].copy()
            e["beat_times_s"][3] = np.nan
        elif key == "nan_timestamp":
            e["timestamp"] = e["timestamp"].copy()
            e["timestamp"][200] = np.nan
        elif key == "nan_timestamp_and_shift":
            e["timestamp"] = e["timestamp"].copy()
            e["timestamp"][200] = np.nan
            e["timestamp"][400] += 1.0
        elif key == "nan_beats_and_shift":
            e["beat_times_s"] = e["beat_times_s"].copy()
            e["beat_times_s"][3] = np.nan
            e["beat_times_s"][10] += 1.0
        elif key == "inf_timestamp":
            e["timestamp"] = e["timestamp"].copy()
            e["timestamp"][5] = np.inf
        elif key == "nan_action":
            e["action"] = e["action"].copy()
            e["action"][7, 2] = np.nan
        return e

    return f


MUTATIONS = [
    "version", "timestamp", "n", "duration", "beats",
    "nan_duration", "nan_beats", "nan_timestamp", "nan_timestamp_and_shift", "nan_beats_and_shift",
    "inf_timestamp", "nan_action",
]


@pytest.mark.parametrize(
    "make",
    [_mut_fps25, *[_mut(k) for k in MUTATIONS]],
    ids=["fps25", *MUTATIONS],
)
def test_rejects_contract_mismatch_with_same_sha(tmp_path, contract, make):
    # sha が同じでも、契約版・fps・N・timestamp・曲長・拍の表が合わなければ出力を作る前に止まる
    choreo = synth.load_choreography(CHOREO_PATH)
    sha = synth.file_sha256(CHOREO_PATH)
    e = make(contract, choreo, sha)
    assert str(e["choreography_sha256"]) == sha
    p = tmp_path / "ep.npz"
    np.savez(p, **e)
    root = tmp_path / "o" / "ds"
    with pytest.raises(ValueError):
        to_lerobot.convert([p], contract, CHOREO_PATH, root, REPO_ID)
    assert not root.parent.exists()


def test_valid_episode_passes_contract_check(contract):
    choreo = synth.load_choreography(CHOREO_PATH)
    e = synth.build_episode(contract, choreo, synth.file_sha256(CHOREO_PATH))
    assert to_lerobot.check_episode_contract(e, contract, choreo) == []


def test_std_floor_and_idempotent(converted, tmp_path):
    from lerobot.datasets.io_utils import load_stats

    out, _ = converted
    stats = load_stats(out)
    for key in to_lerobot.FLOOR_KEYS:
        assert np.all(stats[key]["std"] >= to_lerobot.STD_FLOOR_RAD - 1e-12), key
    copy = tmp_path / "copy"
    shutil.copytree(out, copy)
    before = (copy / "meta" / "stats.json").read_bytes()
    to_lerobot.apply_std_floor(copy)
    once = (copy / "meta" / "stats.json").read_bytes()
    to_lerobot.apply_std_floor(copy)
    twice = (copy / "meta" / "stats.json").read_bytes()
    assert once == twice
    assert json.loads(before) == json.loads(once)  # 変換時に既にかかっている


def test_manifest_copied_and_openable(converted, synth_dir):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out, _ = converted
    assert (out / "manifest.json").read_bytes() == (synth_dir / "manifest.json").read_bytes()
    assert LeRobotDataset(REPO_ID, root=out).num_episodes == K


def test_act_cpu_forward_and_floor_effect(converted):
    import torch
    from lerobot.configs import FeatureType, NormalizationMode
    from lerobot.datasets.io_utils import load_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.feature_utils import dataset_to_policy_features

    out, _ = converted
    stats = load_stats(out)
    probe = LeRobotDataset(REPO_ID, root=out)
    feats = dataset_to_policy_features(probe.meta.features)
    output_features = {k: f for k, f in feats.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in feats.items() if k not in output_features}
    chunk = 10
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        chunk_size=chunk,
        n_action_steps=chunk,
        pretrained_backbone_weights=None,
        normalization_mapping={"STATE": NormalizationMode.MEAN_STD, "ACTION": NormalizationMode.MEAN_STD},
    )
    ds = LeRobotDataset(REPO_ID, root=out, delta_timestamps={"action": [i / probe.fps for i in range(chunk)]})
    pre, _post = make_pre_post_processors(cfg, dataset_stats=stats)
    policy = ACTPolicy(cfg)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)))
    loss, _ = policy.forward(pre(batch))
    assert torch.isfinite(loss).item()

    s = torch.zeros(1, 5)
    s[0, 1] = 0.3
    env = torch.zeros(1, 3)
    a = pre({"observation.state": s.clone(), "observation.environment_state": env.clone()})
    s2 = s.clone()
    s2[0, 1] += 0.01
    b = pre({"observation.state": s2, "observation.environment_state": env.clone()})
    diff = (b["observation.state"] - a["observation.state"]).abs()[0]
    bound = 0.01 / to_lerobot.STD_FLOOR_RAD
    assert diff[1].item() <= bound * (1 + 1e-6)
    # joint1 の元の std は 0（常に 0 の列）。下限が効いていれば 0.01 / 下限 になる。
    s3 = s.clone()
    s3[0, 0] += 0.01
    c = pre({"observation.state": s3, "observation.environment_state": env.clone()})
    d0 = (c["observation.state"] - a["observation.state"]).abs()[0, 0].item()
    assert abs(d0 - bound) <= bound * 1e-5
