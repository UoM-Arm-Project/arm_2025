"""Photo-on-demand camera tests with a simulated phone (no ROS, no network beyond localhost).

Run:  python -m pytest src/poker_vision/test/test_snapshot.py -q
"""
import os
import socket
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poker_vision.core.camera import FrameSource                       # noqa: E402
from poker_vision.core.snapshot import SnapshotCamera, decode_photo    # noqa: E402


def landscape_jpeg(orientation_tag: bool = True) -> bytes:
    """160x120 landscape test photo: red left half, blue right half. Optionally tagged EXIF orientation 6."""
    img = np.zeros((120, 160, 3), np.uint8)
    img[:, :80] = (0, 0, 255)
    img[:, 80:] = (255, 0, 0)
    data = cv2.imencode(".jpg", img)[1].tobytes()
    if not orientation_tag:
        return data
    # minimal APP1/Exif segment: TIFF little-endian, IFD0 with one entry: Orientation (0x0112) = 6
    tiff = (b"II*\x00\x08\x00\x00\x00" + b"\x01\x00" + b"\x12\x01\x03\x00\x01\x00\x00\x00\x06\x00\x00\x00"
            + b"\x00\x00\x00\x00")
    app1 = b"Exif\x00\x00" + tiff
    seg = b"\xff\xe1" + (len(app1) + 2).to_bytes(2, "big") + app1
    return data[:2] + seg + data[2:]


class FakePhone:
    """Loops like the Shortcut: poll /wait, on go-<id> 'take a photo' and upload it."""

    def __init__(self, port, delay=0.05, photo=None, send_id=True, polls=1000):
        self.url = f"http://127.0.0.1:{port}"
        self.delay, self.photo, self.send_id, self.polls = delay, photo or landscape_jpeg(), send_id, polls
        self.uploads, self.replies = 0, []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        for _ in range(self.polls):
            if self.stop.is_set():
                return
            try:
                answer = urllib.request.urlopen(self.url + "/wait", timeout=30).read().decode()
            except OSError:
                time.sleep(0.05)
                continue
            if answer.startswith("go"):
                time.sleep(self.delay)
                q = f"?req={answer}" if self.send_id else ""
                r = urllib.request.Request(self.url + "/upload" + q, data=self.photo, method="POST",
                                           headers={"Content-Type": "image/jpeg"})
                try:
                    self.replies.append(urllib.request.urlopen(r, timeout=10).read().decode())
                except urllib.error.HTTPError as e:
                    self.replies.append(e.read().decode())
                self.uploads += 1

    def close(self):
        self.stop.set()


def eventually(cond, timeout=3.0):
    """The phone counts an upload only once our reply reaches it, just after capture() returns."""
    end = time.monotonic() + timeout
    while time.monotonic() < end and not cond():
        time.sleep(0.02)
    return cond()


@pytest.fixture
def cam_factory():
    made = []

    def make(**cfg):
        cfg.setdefault("port", 0)
        cfg.setdefault("hold_s", 0.5)
        cfg.setdefault("warmup", False)
        cam = SnapshotCamera(cfg, log=lambda m: None)
        made.append(cam)
        return cam

    yield make
    for c in made:
        c.close()


def test_decode_ignores_exif_and_rotates():
    data = landscape_jpeg(orientation_tag=True)
    assert cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (160, 120)  # tag applied
    img = decode_photo(data, 0)
    assert img.shape[:2] == (120, 160)                       # tag ignored: stored landscape pixels
    assert img[60, 10, 2] > 200 and img[60, 150, 0] > 200     # red left, blue right
    r = decode_photo(data, 90)                               # 90 clockwise: left edge goes to the top
    assert r.shape[:2] == (160, 120) and r[10, 60, 2] > 200
    r = decode_photo(data, 270)
    assert r.shape[:2] == (160, 120) and r[150, 60, 2] > 200
    assert decode_photo(b"not an image") is None


def test_capture_round_trip(cam_factory):
    cam = cam_factory(rotate=90)
    phone = FakePhone(cam.port)
    try:
        for _ in range(3):
            frame, msg = cam.capture(timeout=5)
            assert frame is not None, msg
            assert frame.shape[:2] == (160, 120)
        assert eventually(lambda: phone.uploads == 3) and phone.replies == ["saved"] * 3
        assert cam.read()[0].shape[:2] == (160, 120)          # last photo kept for read()
    finally:
        phone.close()


def test_no_phone_says_so(cam_factory):
    cam = cam_factory()
    frame, msg = cam.capture(timeout=0.5)
    assert frame is None and "never polled" in msg


def test_late_photo_never_answers_the_next_read(cam_factory):
    cam = cam_factory()
    phone = FakePhone(cam.port, delay=1.0)                    # slower than the first read allows
    try:
        frame, msg = cam.capture(timeout=0.3)
        assert frame is None and "after the phone was asked" in msg
        time.sleep(1.2)                                       # the late photo arrives now
        assert "stale" in phone.replies
        phone.delay = 0.05
        frame, msg = cam.capture(timeout=5)                   # next read gets its own, new photo
        assert frame is not None, msg
        assert eventually(lambda: phone.uploads == 2)
    finally:
        phone.close()


def test_upload_without_id(cam_factory):
    cam = cam_factory()
    phone = FakePhone(cam.port, send_id=False)
    try:
        frame, msg = cam.capture(timeout=1.0)
        assert frame is None
        assert "missing req" in phone.replies
    finally:
        phone.close()
    cam2 = cam_factory(require_id=False)
    phone2 = FakePhone(cam2.port, send_id=False)
    try:
        frame, msg = cam2.capture(timeout=5)
        assert frame is not None, msg
    finally:
        phone2.close()


def test_heic_is_reported(cam_factory):
    cam = cam_factory()
    phone = FakePhone(cam.port, photo=b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64)
    try:
        frame, msg = cam.capture(timeout=5)
        assert frame is None and "HEIC" in msg and "JPEG" in msg
    finally:
        phone.close()


def test_phone_hanging_up_keeps_the_request(cam_factory):
    cam = cam_factory(hold_s=2.0)
    s = socket.create_connection(("127.0.0.1", cam.port))     # a poll that is abandoned (Shortcut stopped)
    s.sendall(b"GET /wait HTTP/1.1\r\nHost: x\r\n\r\n")
    time.sleep(0.2)
    s.close()
    result = {}
    t = threading.Thread(target=lambda: result.update(zip(("frame", "msg"), cam.capture(timeout=8))))
    t.start()
    time.sleep(0.3)
    phone = FakePhone(cam.port)                                # Shortcut restarted
    try:
        t.join(10)
        assert result.get("frame") is not None, result.get("msg")
    finally:
        phone.close()


def test_warmup_is_replaced_by_a_real_read(cam_factory):
    cam = cam_factory(warmup=True, warmup_timeout_s=30)
    time.sleep(0.1)                                            # warm-up request waiting, no phone yet
    phone = FakePhone(cam.port)
    try:
        frame, msg = cam.capture(timeout=5)
        assert frame is not None, msg
    finally:
        phone.close()


def test_frame_source_snapshot(cam_factory):
    src = FrameSource({"source": "snapshot", "snapshot": {"port": 0, "hold_s": 0.5, "warmup": False}},
                      log=lambda m: None)
    phone = FakePhone(src.snapshot.port)
    try:
        assert src.read()[0] is None                          # nothing until a photo is asked for
        frame, msg = src.capture(timeout=5)
        assert frame is not None, msg
        assert src.read()[0].shape == frame.shape
    finally:
        phone.close()
        src.close()


def test_frame_source_still_capture(tmp_path):
    p = tmp_path / "still.jpg"
    cv2.imwrite(str(p), np.zeros((10, 12, 3), np.uint8))
    src = FrameSource({"source": str(p)})
    frame, msg = src.capture()
    assert frame.shape == (10, 12, 3)
