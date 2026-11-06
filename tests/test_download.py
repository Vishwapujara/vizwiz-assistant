import hashlib
import threading
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vizwiz.data import download as dl


class RangeHandler(SimpleHTTPRequestHandler):
    """Static file server that honours 'Range: bytes=N-' like the VizWiz server."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.translate_path(self.path)
        data = open(path, "rb").read()
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            body = data[start:]
            self.send_response(206)
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server(tmp_path):
    served = tmp_path / "served"
    served.mkdir()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(RangeHandler, directory=served))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield served, f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def make_source(served, base, name, payload, kind="annotations", expected=None):
    (served / name).write_bytes(payload)
    return dl.Source(name, f"{base}/{name}", len(payload), kind, expected)


def test_fresh_download_records_size_and_hash(server, tmp_path):
    served, base = server
    payload = bytes(range(256)) * 1000
    src = make_source(served, base, "a.bin", payload)
    raw = tmp_path / "raw"
    raw.mkdir()
    rec = dl.download_one(src, raw, {})
    assert (raw / "a.bin").read_bytes() == payload
    assert rec["size"] == len(payload)
    assert rec["sha256"] == hashlib.sha256(payload).hexdigest()
    assert not (raw / "a.bin.part").exists()


def test_resume_from_partial_file_gives_same_hash(server, tmp_path):
    served, base = server
    payload = b"vizwiz" * 50_000
    src = make_source(served, base, "b.bin", payload)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "b.bin.part").write_bytes(payload[:12_345])  # simulate a disconnect
    rec = dl.download_one(src, raw, {})
    assert (raw / "b.bin").read_bytes() == payload
    assert rec["sha256"] == hashlib.sha256(payload).hexdigest()


def test_finished_file_is_skipped(server, tmp_path, capsys):
    served, base = server
    src = make_source(served, base, "c.bin", b"x" * 100)
    raw = tmp_path / "raw"
    raw.mkdir()
    rec = dl.download_one(src, raw, {})
    again = dl.download_one(src, raw, {"c.bin": rec})
    assert again == rec
    assert "[skip]" in capsys.readouterr().out


def test_size_mismatch_fails(server, tmp_path):
    served, base = server
    (served / "d.bin").write_bytes(b"short")
    src = dl.Source("d.bin", f"{base}/d.bin", 999, "annotations")
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(RuntimeError, match="size mismatch"):
        dl.download_one(src, raw, {}, retries=1)


def test_check_zip_counts_images(tmp_path):
    path = tmp_path / "imgs.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(3):
            zf.writestr(f"train/VizWiz_train_{i:08d}.jpg", b"jpg")
        zf.writestr("train/readme.txt", b"hi")
    assert dl.check_zip(path, 3, count_jpg_only=True)["ok"]
    assert not dl.check_zip(path, 3, count_jpg_only=False)["ok"]


def test_sources_match_documented_sizes():
    sizes = {s.name: s.size for s in dl.SOURCES}
    assert sizes["train.zip"] == 11_298_421_598
    assert sum(sizes.values()) / 1e9 == pytest.approx(18.96, abs=0.05)
