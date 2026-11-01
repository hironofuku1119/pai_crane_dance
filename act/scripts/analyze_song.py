"""曲の拍・小節頭を解析して measurements/beats.json を書き出す（act/ で実行）。"""
import argparse
import hashlib
import json
import pathlib
import sys

# このプロジェクトは build-system を持たず install されない。スクリプト実行では
# スクリプトのあるフォルダが sys.path[0] になる（Python 3.12 docs, sys_path_init）ので、act/ を足す。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
from crane_dance_act import beat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="../data/PAImusic.wav")
    ap.add_argument("--first-downbeat", default="auto", help="auto か、最初の小節頭の秒")
    ap.add_argument("--bpm", type=float, default=None, help="beat_track に渡すヒント")
    ap.add_argument("--beats-per-bar", type=int, default=4)
    ap.add_argument("--keyframe-bars", default="1,2,4,6,8,10,14,17", help="表示だけ（beats.json には書かない）")
    ap.add_argument("--out", default="measurements/beats.json")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    wav = pathlib.Path(a.wav)
    if not wav.is_file():
        print(f"曲が無い: {wav.name}（git 外。--wav か data/ に置く）")
        return 2

    y, sr = beat.load_mono(wav)
    duration = len(y) / sr
    env = beat.onset_envelope(y, sr)
    bt_kwargs = beat.beat_track_kwargs(a.bpm)  # 呼び出しにも記録にも同じ辞書を使う
    tempo, beats = beat.estimate_beats(env, sr, kwargs=bt_kwargs)
    fit = beat.fit_bpm(beats)

    auto = beat.estimate_downbeats(env, sr, beats, a.beats_per_bar)
    warns = []
    if a.first_downbeat == "auto":
        downs = auto
        method = "onset-phase(auto)"
    else:
        fd = float(a.first_downbeat)
        downs = beat.downbeats_from_first(beats, fd, a.beats_per_bar)
        method = f"first_downbeat={a.first_downbeat}"
        if len(downs) != len(auto) or np.any(np.abs(downs - auto) > 0.05):
            warns.append("手動指定の小節頭が onset-phase(auto) の推定と食い違っている")
    if abs(tempo - fit) >= 1:
        msg = f"beat_track のテンポ {tempo:.2f} と当てはめ BPM {fit:.2f} が 1 以上離れている"
        warns.append(msg)
        print("警告:", msg)

    print(f"曲: {wav.name}  長さ: {duration:.2f}s  sr: {sr}  hop: {beat.HOP}")
    print(f"beat_track テンポ: {tempo:.2f}  当てはめ BPM: {fit:.2f}  拍数: {len(beats)}  小節頭: {len(downs)}")
    if a.bpm is not None:
        print(f"指定 BPM(ヒント): {a.bpm}  当てはめとの差: {fit - a.bpm:+.2f}")
    print("小節 拍 時刻[s]")
    for bar, b, t in beat.bar_table(beats, downs, a.beats_per_bar):
        print(f"{bar:4d} {b:2d} {t:8.3f}")
    for k in [int(x) for x in a.keyframe_bars.split(",") if x.strip()]:
        if 1 <= k <= len(downs):
            print(f"keyframe 小節 {k}: {downs[k - 1]:.3f}s")
        else:
            print(f"keyframe 小節 {k}: 範囲外（小節頭 {len(downs)} 個）")

    load_kwargs, soxr_version = beat.load_record(wav, sr)
    source = {
        "audio_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
        "librosa_version": librosa.__version__,
        "sample_rate_hz": int(sr),
        "hop_length": beat.HOP,
        "downbeat_method": method,
        "load_kwargs": load_kwargs,
        "beat_track_kwargs": bt_kwargs,
        "soxr_version": soxr_version,
    }
    data = beat.make_beats_json(fit, beats, downs, duration, warns, source, meter=f"{a.beats_per_bar}/4")
    if a.dry_run:
        print("(--dry-run: 書き込まない)")
        return 0
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(f"書き出し: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
