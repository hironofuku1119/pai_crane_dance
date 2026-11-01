"""テスト用に、小さい本物の ACT チェックポイント（pretrained_model）を CPU で作る。学習はしない（初期値のまま保存）。"""
import pathlib

from crane_dance_act import state_mode as state_mode_mod
from crane_dance_act import synth, to_lerobot

ACT_DIR = pathlib.Path(__file__).resolve().parents[1]
CONTRACT_PATH = ACT_DIR.parent / "contract" / "contract.json"
CHOREO_PATH = ACT_DIR.parent / "contract" / "choreography.json"
REPO_ID = "local/crane_dance_helper_test"


def make_dataset(tmp, sm=None, episodes=2, name="ds"):
    """合成 npz → LeRobotDataset。manifest に state_mode が入る。(root, npz_dir) を返す。"""
    contract = synth.load_contract(CONTRACT_PATH)
    choreo = synth.load_choreography(CHOREO_PATH)
    d = pathlib.Path(tmp) / f"{name}-npz"
    synth.write_varied_set(contract, choreo, synth.file_sha256(CHOREO_PATH), episodes, 0, d, state_mode=sm)
    root = pathlib.Path(tmp) / f"{name}-lr" / "ds"
    to_lerobot.convert(sorted(d.glob("ep*.npz")), contract, CHOREO_PATH, root, REPO_ID, d / "manifest.json")
    return root, d


def make_checkpoint(tmp, dataset_root, sm=None, chunk=10, write_state_mode=True, name="run"):
    """dataset_root の stats で初期化した小さい方策を tmp/<name>/checkpoints/000010/pretrained_model に保存する。
    sm を渡すと（かつ write_state_mode なら）横に state_mode.json を書く。pretrained_model のパスを返す。"""
    import torch
    from lerobot.configs import FeatureType, NormalizationMode
    from lerobot.datasets.io_utils import load_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.feature_utils import dataset_to_policy_features

    stats = load_stats(dataset_root)
    probe = LeRobotDataset(REPO_ID, root=dataset_root)
    feats = dataset_to_policy_features(probe.meta.features)
    output_features = {k: f for k, f in feats.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in feats.items() if k not in output_features}
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        chunk_size=chunk,
        n_action_steps=chunk,
        pretrained_backbone_weights=None,
        normalization_mapping={"STATE": NormalizationMode.MEAN_STD, "ACTION": NormalizationMode.MEAN_STD},
    )
    torch.manual_seed(0)
    policy = ACTPolicy(cfg)
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    ck = pathlib.Path(tmp) / name / "checkpoints" / "000010" / "pretrained_model"
    policy.save_pretrained(ck)
    pre.save_pretrained(ck)
    post.save_pretrained(ck)
    if sm is not None and write_state_mode:
        state_mode_mod.write_atomic(state_mode_mod.checkpoint_file(ck), sm)
    return ck
