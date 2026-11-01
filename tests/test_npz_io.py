"""crane_dance.npz_io の検査（標準ライブラリの unittest・numpy なし）。"""
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "ros" / "src" / "crane_dance"))
from crane_dance import npz_io  # noqa: E402


def npy(descr, shape, payload, *, version=(1, 0), fortran=False, header_pad=None) -> bytes:
    text = f"{{'descr': '{descr}', 'fortran_order': {fortran}, 'shape': {shape!r}, }}"
    lf = 2 if version == (1, 0) else 4
    base = 6 + 2 + lf + len(text) + 1
    pad = (-base) % 64 if header_pad is None else header_pad
    header = (text + " " * pad + "\n").encode("latin1")
    fmt = "<H" if lf == 2 else "<I"
    return b"\x93NUMPY" + bytes(version) + struct.pack(fmt, len(header)) + header + payload


def write_zip(path: Path, members, compression=zipfile.ZIP_STORED) -> None:
    with zipfile.ZipFile(path, "w", compression) as zf:
        items = members.items() if isinstance(members, dict) else members
        for name, data in items:
            zf.writestr(name, data)


class NpzIoTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "x.npz"

    def members(self):
        vals = [1.5, -2.0, 3.25, 4.0, 5.0, 6.0]
        return {
            "m.npy": npy("<f4", (2, 3), struct.pack("<6f", *vals)),
            "d.npy": npy("<f8", (2,), struct.pack("<2d", 0.1, 0.2)),
            "i.npy": npy("<i8", (), struct.pack("<q", 50)),
            "s.npy": npy("<U5", (), "0.2.0".encode("utf-32-le")),
            "t.npy": npy("<U5", (), "ab".ljust(5, "\x00").encode("utf-32-le")),
        }

    def test_roundtrip_stored_and_deflate(self):
        for comp in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            write_zip(self.path, self.members(), comp)
            out = npz_io.read_npz(self.path)
            self.assertEqual(set(out), {"m", "d", "i", "s", "t"})
            self.assertEqual(out["m"]["shape"], (2, 3))
            self.assertEqual(out["m"]["descr"], "<f4")
            self.assertEqual(out["m"]["data"], [1.5, -2.0, 3.25, 4.0, 5.0, 6.0])
            self.assertEqual(npz_io.rows(out["m"]), [[1.5, -2.0, 3.25], [4.0, 5.0, 6.0]])
            self.assertEqual(out["d"]["data"], [struct.unpack("<d", struct.pack("<d", 0.1))[0],
                                                struct.unpack("<d", struct.pack("<d", 0.2))[0]])
            self.assertEqual(out["i"]["data"], [50])
            self.assertEqual(out["i"]["shape"], ())
            self.assertEqual(out["s"]["data"], ["0.2.0"])
            self.assertEqual(out["t"]["data"], ["ab"])

    def reject(self, members, word=None):
        write_zip(self.path, members)
        with self.assertRaises(ValueError) as cm:
            npz_io.read_npz(self.path)
        if word:
            self.assertIn(word, str(cm.exception))

    def test_reject_fortran_order(self):
        self.reject({"a.npy": npy("<f4", (2, 2), b"\x00" * 16, fortran=True)}, "fortran_order")

    def test_reject_big_endian(self):
        self.reject({"a.npy": npy(">f4", (1,), b"\x00" * 4)}, "descr")

    def test_reject_bytes_and_object(self):
        self.reject({"a.npy": npy("|S64", (), b"\x00" * 64)}, "descr")
        self.reject({"a.npy": npy("|O", (), b"\x00" * 8)}, "descr")

    def test_reject_body_size(self):
        self.reject({"a.npy": npy("<f4", (3,), b"\x00" * 8)}, "本体")
        self.reject({"a.npy": npy("<f4", (3,), b"\x00" * 16)}, "本体")

    def test_reject_header_length(self):
        self.reject({"a.npy": npy("<f4", (1,), b"\x00" * 4, header_pad=20000)}, "ヘッダ長")

    def test_reject_version(self):
        self.reject({"a.npy": npy("<f4", (1,), b"\x00" * 4, version=(9, 0))}, "版")

    def test_reject_duplicate_name(self):
        good = npy("<f4", (1,), b"\x00" * 4)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_zip(self.path, [("a.npy", good), ("a.npy", good)])
        with self.assertRaises(ValueError) as cm:
            npz_io.read_npz(self.path)
        self.assertIn("重複", str(cm.exception))

    def test_reject_directory_name(self):
        self.reject({"a/b.npy": npy("<f4", (1,), b"\x00" * 4)}, "ディレクトリ")

    def test_reject_not_zip(self):
        self.path.write_bytes(b"not a zip")
        with self.assertRaises(ValueError):
            npz_io.read_npz(self.path)


if __name__ == "__main__":
    unittest.main()
