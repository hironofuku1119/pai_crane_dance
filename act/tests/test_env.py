"""環境確認: 主要 import が通ることと torch が 2.11 系であること。"""
import importlib
import importlib.metadata as md


def test_core_imports_and_torch_version():
    for mod in ("torch", "lerobot", "librosa"):
        importlib.import_module(mod)
    assert md.version("torch").startswith("2.11.")
