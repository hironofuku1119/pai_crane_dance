"""npz を標準ライブラリだけで読む（numpy を使わない・ROS に依存しない・Python 3.10 以上）。

規則は scripts/check_contract.py --npz と同じ。違反は ValueError（日本語の理由つき）。
鍵が契約の 10 個ちょうどかどうかの照合は plan.py の側でする（ここは読めるものを読む）。
"""
from __future__ import annotations

import ast
import re
import struct
import zipfile
from collections import Counter

HEADER_MAX = 10000
_FIXED = {"<f4": (4, "<f"), "<f8": (8, "<d"), "<i8": (8, "<q")}


def _itemsize(descr: str) -> int:
    if descr in _FIXED:
        return _FIXED[descr][0]
    # 文字数は 1〜6 桁（検査器の RE_U_DESCR と同じ。巨大な数字列の int() 変換で例外を出さないため）
    m = re.fullmatch(r"<U([1-9][0-9]{0,5})", descr)
    if m:
        return 4 * int(m.group(1))
    raise ValueError(f"descr {descr!r} は <f4・<f8・<i8・<U<n> のどれでもない")


def _read_one(zf: zipfile.ZipFile, name: str) -> dict:
    label = f"npz: {name}"
    try:
        raw = zf.read(name)
    except (zipfile.BadZipFile, KeyError, OSError, RuntimeError, EOFError) as e:
        raise ValueError(f"{label}: 読めない ({e})") from e
    if len(raw) < 8 or raw[:6] != b"\x93NUMPY":
        raise ValueError(f"{label}: magic が \\x93NUMPY でない")
    ver = (raw[6], raw[7])
    if ver == (1, 0):
        lf, fmt, enc = 2, "<H", "latin1"
    elif ver == (2, 0):
        lf, fmt, enc = 4, "<I", "latin1"
    elif ver == (3, 0):
        lf, fmt, enc = 4, "<I", "utf-8"
    else:
        raise ValueError(f"{label}: npy の版 {ver} は (1,0)・(2,0)・(3,0) のどれでもない")
    if len(raw) < 8 + lf:
        raise ValueError(f"{label}: ヘッダ長が読めない")
    hlen = struct.unpack(fmt, raw[8:8 + lf])[0]
    if hlen > HEADER_MAX:
        raise ValueError(f"{label}: ヘッダ長 {hlen} が {HEADER_MAX} バイトを超える")
    start = 8 + lf
    hbytes = raw[start:start + hlen]
    if len(hbytes) < hlen:
        raise ValueError(f"{label}: ヘッダが途中で切れている")
    try:
        hdr = ast.literal_eval(hbytes.decode(enc))
    except (ValueError, SyntaxError, MemoryError, RecursionError, UnicodeDecodeError) as e:
        raise ValueError(f"{label}: ヘッダを読めない（辞書でない）") from e
    if not (isinstance(hdr, dict) and set(hdr) == {"descr", "fortran_order", "shape"}):
        raise ValueError(f"{label}: ヘッダの鍵が descr・fortran_order・shape ちょうどでない")
    if not isinstance(hdr["fortran_order"], bool):
        raise ValueError(f"{label}: fortran_order が bool でない")
    if hdr["fortran_order"]:
        raise ValueError(f"{label}: fortran_order が True（2 次元の並びが入れ替わるので読まない）")
    descr = hdr["descr"]
    if not isinstance(descr, str):
        raise ValueError(f"{label}: descr が文字列でない")
    try:
        size = _itemsize(descr)
    except ValueError as e:
        raise ValueError(f"{label}: {e}") from e
    shape = hdr["shape"]
    if not (isinstance(shape, tuple)
            and all(isinstance(x, int) and not isinstance(x, bool) for x in shape)):
        raise ValueError(f"{label}: shape が int の tuple でない")
    if any(x < 0 for x in shape):
        raise ValueError(f"{label}: shape {shape} に負の次元がある")
    count = 1
    for x in shape:
        count *= x
    body = raw[start + hlen:]
    if len(body) != count * size:
        raise ValueError(f"{label}: 本体の大きさ {len(body)} バイトが、形と型から求めた "
                         f"{count * size} バイトと違う")
    if descr in _FIXED:
        code = _FIXED[descr][1][1]
        data = list(struct.unpack(f"<{count}{code}", body)) if count else []
    else:
        data = []
        for i in range(count):
            chunk = body[i * size:(i + 1) * size]
            try:
                text = chunk.decode("utf-32-le")
            except UnicodeDecodeError as e:
                raise ValueError(f"{label}: 文字列が UTF-32 として読めない") from e
            data.append(text.rstrip("\x00"))
    return {"shape": shape, "descr": descr, "data": data}


def read_npz(path) -> dict:
    """鍵 → {"shape": tuple, "descr": str, "data": 平らな list}。"""
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as e:
        raise ValueError(f"npz: zip として開けない ({e})") from e
    out: dict = {}
    with zf:
        names = zf.namelist()
        for name, cnt in Counter(names).items():
            if cnt > 1:
                raise ValueError(f"npz: 項目名 {name} が {cnt} 回ある（重複）")
            if "/" in name or "\\" in name:
                raise ValueError(f"npz: 項目名 {name} にディレクトリが含まれる")
            if not name.endswith(".npy") or len(name) <= 4:
                raise ValueError(f"npz: 項目名 {name} が <鍵>.npy の形でない")
        for name in names:
            out[name[:-4]] = _read_one(zf, name)
    return out


def rows(entry: dict) -> list:
    """2 次元の entry を行の list に直す。"""
    shape = entry["shape"]
    if len(shape) != 2:
        raise ValueError(f"rows: 2 次元でない（shape={shape}）")
    n, m = shape
    d = entry["data"]
    return [d[i * m:(i + 1) * m] for i in range(n)]
