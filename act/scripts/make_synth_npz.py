"""振付から合成の npz（1 エピソード）を作る。cwd に依存しない。"""
import argparse
import pathlib
import sys

ACT_DIR = pathlib.Path(__file__).resolve().parents[1]
# スクリプト実行では sys.path[0] がスクリプトのフォルダになる（Python 3.12 docs, sys_path_init）ので act/ を足す。
sys.path.insert(0, str(ACT_DIR))

from crane_dance_act import synth  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", type=pathlib.Path, default=ACT_DIR.parent / "contract" / "contract.json")
    ap.add_argument("--choreography", type=pathlib.Path, default=ACT_DIR.parent / "contract" / "choreography.json")
    ap.add_argument("--out", type=pathlib.Path, default=ACT_DIR / "out" / "ep000.npz")
    ap.add_argument("--episodes", type=int, default=None, help="指定すると複数本モード（観測のばらつきあり）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=pathlib.Path, default=ACT_DIR / "out" / "synth")
    a = ap.parse_args()

    contract = synth.load_contract(a.contract)
    choreo = synth.load_choreography(a.choreography)
    sha = synth.file_sha256(a.choreography)
    if a.episodes is not None:
        if a.episodes < 1:
            ap.error("--episodes は 1 以上")
        m = synth.write_varied_set(contract, choreo, sha, a.episodes, a.seed, a.out_dir)
        print(f"episodes: {m['episodes']}")
        print(f"seed: {m['seed']}")
        print(f"contract_version: {m['contract_version']}")
        print(f"choreography_sha256: {sha[:12]}")
        print(f"out_dir: {a.out_dir.name}")
        return 0
    ep = synth.build_episode(contract, choreo, sha)
    synth.save_npz(a.out, ep, contract)

    print(f"N: {len(ep['timestamp'])}")
    print(f"duration_s: {float(ep['duration_s'])}")
    print(f"fps: {int(ep['fps'])}")
    print(f"contract_version: {ep['contract_version']}")
    print(f"choreography_sha256: {sha[:12]}")
    print(f"out: {a.out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
