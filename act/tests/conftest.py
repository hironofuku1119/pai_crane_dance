import os

# lerobot・huggingface_hub の import より前に固定する（値は import 時に 1 回だけ読まれる）。
os.environ["HF_HUB_OFFLINE"] = "1"
