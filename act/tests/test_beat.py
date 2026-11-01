import os
import pathlib

import numpy as np
import pytest

from crane_dance_act import beat


def _click_track(bpm, sr=22050, dur=32.0):
    y = np.zeros(int(sr * dur), dtype=np.float32)
    period = 60.0 / bpm
    n = int(0.010 * sr)
    tt = np.arange(n) / sr
    burst = np.sin(2 * np.pi * 1000 * tt) * np.exp(-tt / 0.003)
    strong = []
    i = 0
    while i * period + 0.01 < dur:
        t = i * period
        amp = 2.0 if i % 4 == 0 else 1.0
        if i % 4 == 0:
            strong.append(t)
        s = int(t * sr)
        e = min(s + n, len(y))
        y[s:e] += (amp * burst[: e - s]).astype(np.float32)
        i += 1
    return y, sr, np.array(strong)


def _check_click(bpm):
    y, sr, strong = _click_track(bpm)
    env = beat.onset_envelope(y, sr)
    _, beats = beat.estimate_beats(env, sr)
    assert abs(beat.fit_bpm(beats) - bpm) <= 0.5
    downs = beat.estimate_downbeats(env, sr, beats, 4)
    assert len(downs) > 0
    for d in downs:
        assert np.min(np.abs(strong - d)) <= 0.05, (d, bpm)


def test_click_120():
    _check_click(120)


def test_click_100():
    _check_click(100)


def test_json_keys():
    src = {"audio_sha256": "x", "librosa_version": "v", "sample_rate_hz": 44100,
           "hop_length": 512, "downbeat_method": "m", "load_kwargs": {},
           "beat_track_kwargs": {}, "soxr_version": None}
    d = beat.make_beats_json(120.0, [0.5], [0.5], 3.0, [], src)
    assert set(d) == {"bpm", "beat_times_s", "downbeat_times_s", "duration_s",
                      "sections", "warnings", "source"}
    assert set(d["source"]) == {"audio_sha256", "librosa_version", "sample_rate_hz",
                                "hop_length", "downbeat_method", "load_kwargs",
                                "beat_track_kwargs", "soxr_version"}
    assert len(d["sections"]) == 1
    assert set(d["sections"][0]) == {"start_s", "end_s", "bpm", "meter"}


def test_downbeats_from_first():
    beats = 0.5 + 0.5 * np.arange(20)
    d = beat.downbeats_from_first(beats, 1.52, 4)
    assert np.allclose(d, beats[2::4]), d
    assert np.allclose(beat.downbeats_from_first(beats, 0.5, 4), beats[0::4])
    with pytest.raises(ValueError):
        beat.downbeats_from_first(beats, 1.25, 4)


def test_downbeats_from_first_later_start():
    # 指定に最も近い拍が最初の小節頭になる（それより前の拍を小節頭にしない）
    beats = 0.5 + 0.5 * np.arange(20)
    d = beat.downbeats_from_first(beats, 3.0, 4)
    assert np.allclose(d, beats[5::4]), d
    assert d[0] == pytest.approx(3.0)


def test_bar_table_pickup():
    beats = 0.5 * np.arange(1, 21)
    downs = beats[6::4]  # 6 拍が小節頭より前
    rows = beat.bar_table(beats, downs, 4)
    assert len(rows) == len(beats)
    assert all(r[0] == 0 for r in rows[:6])
    assert rows[6] == (1, 1, float(beats[6]))
    assert all(r[0] >= 0 for r in rows)
    assert beat.bar_table(beats, [], 4) == []


def test_empty_inputs():
    assert len(beat.estimate_downbeats(np.zeros(10), 22050, [], 4)) == 0
    for bad in ([], [1.0]):
        with pytest.raises(ValueError):
            beat.fit_bpm(bad)
    with pytest.raises(ValueError):
        beat.downbeats_from_first([], 1.0)


# 手で作った暫定表: 小節番号 -> 小節頭の時刻[s]
LAB_KEYFRAMES = {1: 0.093, 2: 1.927, 4: 5.608, 6: 9.288, 8: 12.945, 10: 16.625, 14: 23.963, 17: 29.408}


def test_real_song():
    p = os.environ.get("CRANE_SONG_WAV")
    if not p or not pathlib.Path(p).is_file():
        pytest.skip("CRANE_SONG_WAV 未設定")
    y, sr = beat.load_mono(p)
    assert abs(len(y) / sr - 31.20) <= 0.01
    env = beat.onset_envelope(y, sr)
    _, beats = beat.estimate_beats(env, sr)
    assert len(beats) == 66
    assert abs(beats[0] - 0.093) <= 0.02
    assert abs(beat.fit_bpm(beats) - 130.88) <= 0.5
    auto = beat.estimate_downbeats(env, sr, beats, 4)
    assert len(auto) == 17
    assert abs(auto[0] - 0.093) <= 0.02
    man = beat.downbeats_from_first(beats, 0.093, 4)
    assert np.allclose(man, auto)
    for bar, t in LAB_KEYFRAMES.items():
        assert abs(auto[bar - 1] - t) <= 0.05, (bar, auto[bar - 1], t)
