"""バッチ道具のテスト用の設定の文字列。EXAMPLE_TOML は比較 B1〜B5 と同じ中身の 5 run（batches/example.toml の予定の中身）。"""

_COMMON_TRAIN = """steps = 20000
batch_size = 32
save_freq = 1000
lr = 1e-4
lr_backbone = 1e-4
"""

_COMMON_TAIL = """
[run.eval]
ckpts = [5000, 10000, 20000]

[run.export]
ckpts = [10000, 20000]
modes = ["open", "closed"]
formats = ["npz", "yaml"]
"""


def _run(name, desc, data="", extra_train=""):
    return (
        f'\n[[run]]\nname = "{name}"\ndescription = "{desc}"\n\n[run.data]\n{data}\n[run.train]\n{_COMMON_TRAIN}{extra_train}{_COMMON_TAIL}'
    )


EXAMPLE_TOML = (
    '[batch]\ndescription = "比較 B1〜B5 の再現（lr 1e-4・batch 32・20k step）"\n'
    + _run("b1", "今の観測（σ0.0175・τ0.02〜0.10）・lr1e-4・b32・20k",
           'seed = 0\nepisodes = 20\nsigma_rad = 0.0175\nclip_rad = 0.035\ntau_range_s = [0.02, 0.10]\nstate_mode = "lagged"\nenv_variant = "full"\n',
           "chunk_size = 50\n")
    + _run("b2", "強いばらつき（σ0.10・±0.30・τ0.02〜0.25）・lr1e-4・b32・20k",
           "sigma_rad = 0.10\nclip_rad = 0.30\ntau_range_s = [0.02, 0.25]\n")
    + _run("b3", "state 定数（初期姿勢）・lr1e-4・b32・20k", 'state_mode = "constant"\n')
    + _run("b4", "B2 のデータ・chunk100・lr1e-4・b32・20k",
           "sigma_rad = 0.10\nclip_rad = 0.30\ntau_range_s = [0.02, 0.25]\n", "chunk_size = 100\n")
    + _run("b5", "state 定数＋拍位相定数（進行度だけ）・lr1e-4・b32・20k",
           'state_mode = "constant"\nenv_variant = "no_phase"\n')
)  # fmt: skip


def tiny_toml(steps=2, save_freq=2, names=("t1",), chunk=10, extra_run=""):
    """CPU で数秒の極小の設定（合成 2 本・steps 2・評価 1 本・書き出し 1 つ）。"""
    out = ""
    for n in names:
        out += f"""
[[run]]
name = "{n}"
description = "極小"
{extra_run}
[run.data]
episodes = 2

[run.train]
steps = {steps}
batch_size = 2
save_freq = {save_freq}
chunk_size = {chunk}

[run.eval]
ckpts = [{steps}]
count = 1
[[run.eval.columns]]
kind = "open"
m = 10
[[run.eval.columns]]
kind = "closed"
m = 10
tau = 0.06

[run.export]
ckpts = [{steps}]
modes = ["open", "closed"]
formats = ["npz", "yaml"]
"""
    return out
