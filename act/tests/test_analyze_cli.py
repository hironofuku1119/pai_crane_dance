"""scripts/analyze_song.py を実際に子プロセスで走らせ、引数が JSON と表に届くことを確かめる。"""
import importlib.util
import json
import pathlib
import subprocess
import sys

import librosa
import pytest
import soundfile as sf

from test_beat import _click_track

ACT_DIR = pathlib.Path(__file__).resolve().parents[1]


def _run(tmp_path, *extra):
    y, sr, _ = _click_track(120, sr=22050, dur=12.0)
    wav = tmp_path / "click.wav"
    sf.write(str(wav), y, sr)
    out = tmp_path / "beats.json"
    r = subprocess.run(
        [sys.executable, "scripts/analyze_song.py", "--wav", str(wav), "--out", str(out), *extra],
        cwd=ACT_DIR, capture_output=True, text=True, check=True,
    )
    return json.loads(out.read_text()), r.stdout


def _table(stdout):
    """小節表の行 (小節, 拍, 時刻) を取り出す。keyframe 行などは除く。"""
    rows = []
    for line in stdout.splitlines():
        p = line.split()
        if len(p) == 3 and p[0].isdigit() and p[1].isdigit():
            rows.append((int(p[0]), int(p[1]), float(p[2])))
    return rows


def test_first_downbeat_reaches_json_and_table(tmp_path):
    data, out = _run(tmp_path, "--first-downbeat", "3.0")
    assert abs(data["downbeat_times_s"][0] - 3.0) <= 0.05
    rows = _table(out)
    assert rows
    before = [r for r in rows if r[2] < 3.0 - 0.05]
    assert before, "3.0 より前の拍が表に無い"
    assert all(r[0] == 0 for r in before), before


def test_meter_follows_beats_per_bar(tmp_path):
    data3, _ = _run(tmp_path, "--beats-per-bar", "3")
    assert data3["sections"][0]["meter"] == "3/4"
    data4, _ = _run(tmp_path)
    assert data4["sections"][0]["meter"] == "4/4"


def test_source_records_actual_call_args(tmp_path):
    data, _ = _run(tmp_path)
    src = data["source"]
    assert set(src["load_kwargs"]) == {"sr", "mono", "res_type", "offset", "duration"}
    assert set(src["beat_track_kwargs"]) == {"start_bpm", "tightness", "trim", "units"}
    # sample_rate_hz は load_kwargs.sr と同じ値。リサンプルしていないので res_type・soxr_version は null
    assert src["load_kwargs"]["sr"] == src["sample_rate_hz"] == 22050
    assert src["load_kwargs"]["res_type"] is None
    assert src["soxr_version"] is None
    assert src["beat_track_kwargs"]["trim"] is False


def _load_script():
    spec = importlib.util.spec_from_file_location("analyze_song_inproc", ACT_DIR / "scripts" / "analyze_song.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("extra, expected_bpm", [([], None), (["--bpm", "90"], 90.0)])
def test_recorded_beat_track_kwargs_equal_actual_call(tmp_path, monkeypatch, extra, expected_bpm):
    y, sr, _ = _click_track(120, sr=22050, dur=12.0)
    wav = tmp_path / "click.wav"
    sf.write(str(wav), y, sr)
    out = tmp_path / "beats.json"

    calls = []
    real = librosa.beat.beat_track

    def spy(*args, **kwargs):
        calls.append(dict(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(librosa.beat, "beat_track", spy)
    monkeypatch.setattr(sys, "argv", ["analyze_song.py", "--wav", str(wav), "--out", str(out), *extra])
    assert _load_script().main() == 0

    assert len(calls) == 1
    actual = {k: v for k, v in calls[0].items() if k not in ("onset_envelope", "sr", "hop_length")}
    recorded = json.loads(out.read_text())["source"]["beat_track_kwargs"]
    assert recorded == actual
    if expected_bpm is None:
        assert "bpm" not in recorded
    else:
        assert recorded["bpm"] == expected_bpm
