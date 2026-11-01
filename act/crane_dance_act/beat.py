"""拍・小節頭の推定と beats.json の組み立て。"""

import librosa
import numpy as np

# 手で作った小節頭の表（小節番号 -> 時刻）は 44.1 kHz・hop 512 で作られている。同じ条件に揃える。
# 実測： 22050 Hz に落とすと小節頭の位相が 2 拍ずれる。そのため元のサンプリング周波数で読む（None）。
# beat_track の既定 trim=True は冒頭の弱い拍を削るため、estimate_beats では trim=False を使う。
HOP = 512
SR = None

# 読み込みと beat_track に実際に渡す引数を、ここ一か所に置く（記録と実際の呼び出しをずらさないため）。
# 既定値は librosa 1.0.0 のもの（start_bpm=120.0・tightness=100・units='frames'）。trim だけ既定と違う。
LOAD_KWARGS = {"sr": SR, "mono": True}
# サンプリング周波数を変えて読むときだけ使うリサンプラ（librosa 既定）。変えないときは使われない。
RES_TYPE = "soxr_hq"
BEAT_TRACK_KWARGS = {"start_bpm": 120.0, "tightness": 100, "trim": False, "units": "frames"}


def load_mono(path):
    """曲を LOAD_KWARGS の条件（既定は元のサンプリング周波数のモノラル）で読む。(y, sr) を返す。"""
    return librosa.load(str(path), res_type=RES_TYPE, **LOAD_KWARGS)


def load_record(path, sr):
    """beats.json の source に書く load_kwargs と soxr_version を返す。sr は実際に読んだ値。"""
    import soundfile

    native = soundfile.info(str(path)).samplerate
    resampled = sr != native
    soxr_version = None
    if resampled:
        import soxr  # リサンプルしたときだけ import する

        soxr_version = soxr.__version__
    load_kwargs = {
        "sr": int(sr),
        "mono": LOAD_KWARGS["mono"],
        "res_type": RES_TYPE if resampled else None,
        "offset": 0.0,
        "duration": None,
    }
    return load_kwargs, soxr_version


def onset_envelope(y, sr):
    """onset 強度の包絡（1 度だけ計算して拍・小節頭の両方に渡す）。"""
    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP, aggregate=np.median)


def beat_track_kwargs(bpm_hint=None):
    """beat_track に渡す引数の辞書（BEAT_TRACK_KWARGS のコピー。bpm_hint があれば "bpm" を足す）。
    この辞書を呼び出しにも beats.json の記録にも使う。onset_envelope・sr・hop_length は含めない。"""
    kwargs = dict(BEAT_TRACK_KWARGS)
    if bpm_hint is not None:
        kwargs["bpm"] = bpm_hint
    return kwargs


def estimate_beats(env, sr, bpm_hint=None, kwargs=None):
    """beat_track で (tempo[BPM], 拍の時刻[秒]) を返す。kwargs が無ければ beat_track_kwargs(bpm_hint) を使う。"""
    if kwargs is None:
        kwargs = beat_track_kwargs(bpm_hint)
    tempo, frames = librosa.beat.beat_track(
        onset_envelope=env, sr=sr, hop_length=HOP, **kwargs
    )
    times = librosa.frames_to_time(frames, sr=sr, hop_length=HOP)
    return float(np.atleast_1d(tempo)[0]), np.asarray(times, dtype=float)


def fit_bpm(beat_times_s):
    """拍の時刻に直線を当てはめ、傾き（拍間隔）から BPM を返す。拍が 2 つ未満なら ValueError。"""
    t = np.asarray(beat_times_s, dtype=float)
    if len(t) < 2:
        raise ValueError("拍が 2 つ未満で BPM を当てはめられない")
    slope = np.polyfit(np.arange(len(t)), t, 1)[0]
    return float(60.0 / slope)


def estimate_downbeats(env, sr, beat_times_s, beats_per_bar=4):
    """小節頭の時刻を返す。位相 p ごとに env[拍 p, p+N, ...] の和を求め、最大の p を小節頭とする。"""
    beat_times_s = np.asarray(beat_times_s, dtype=float)
    if len(beat_times_s) == 0:
        return np.array([], dtype=float)
    frames = np.clip(librosa.time_to_frames(beat_times_s, sr=sr, hop_length=HOP), 0, len(env) - 1)
    strength = env[frames]
    scores = [strength[p::beats_per_bar].sum() for p in range(beats_per_bar)]
    phase = int(np.argmax(scores))
    return beat_times_s[phase::beats_per_bar]


def downbeats_from_first(beat_times_s, first_downbeat_s, beats_per_bar=4):
    """first_downbeat_s に最も近い拍を最初の小節頭とし、そこから beats_per_bar 拍おきに小節頭を取る（それより前の拍は小節頭にしない）。0.05s 以上離れたら ValueError。"""
    beat_times_s = np.asarray(beat_times_s, dtype=float)
    if len(beat_times_s) == 0:
        raise ValueError("拍が無い")
    i0 = int(np.argmin(np.abs(beat_times_s - first_downbeat_s)))
    gap = abs(beat_times_s[i0] - first_downbeat_s)
    if gap >= 0.05:
        raise ValueError(f"最も近い拍が {gap:.3f}s 離れている（first_downbeat={first_downbeat_s}）")
    return beat_times_s[i0::beats_per_bar]


def bar_table(beat_times_s, downbeat_times_s, beats_per_bar=4):
    """(小節番号, 拍番号, 時刻) の行のリスト。小節は最初の小節頭から 1 始まり、それより前の拍（アウフタクト）は全て小節 0。"""
    beat_times_s = np.asarray(beat_times_s, dtype=float)
    downs = np.asarray(downbeat_times_s, dtype=float)
    rows = []
    if len(downs) == 0:
        return rows
    start = int(np.argmin(np.abs(beat_times_s - downs[0])))
    for i, t in enumerate(beat_times_s):
        k = i - start
        if k < 0:
            rows.append((0, k % beats_per_bar + 1, float(t)))
        else:
            rows.append((k // beats_per_bar + 1, k % beats_per_bar + 1, float(t)))
    return rows


def make_beats_json(bpm, beat_times_s, downbeat_times_s, duration_s, warnings, source, meter="4/4"):
    """契約 beats_format のキーだけを持つ dict を返す。source は dict(audio_sha256 など)。"""
    r = lambda a: [round(float(x), 4) for x in a]
    duration = round(float(duration_s), 4)
    bpm = round(float(bpm), 4)
    return {
        "bpm": bpm,
        "beat_times_s": r(beat_times_s),
        "downbeat_times_s": r(downbeat_times_s),
        "duration_s": duration,
        "sections": [{"start_s": 0.0, "end_s": duration, "bpm": bpm, "meter": meter}],
        "warnings": list(warnings),
        "source": dict(source),
    }
