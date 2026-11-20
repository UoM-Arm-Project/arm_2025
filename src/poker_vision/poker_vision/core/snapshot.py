"""Photo-on-demand camera: a phone takes a real photo each time a read asks for one.

Instead of streaming video, the phone runs a looping iOS Shortcut that long-polls
this node over the local network:

    phone  GET  /wait                 held up to hold_s; answers "go-<id>" or "idle"
    phone  (takes a photo, converts it to JPEG)
    phone  POST /upload?req=go-<id>   the JPEG as the request body (or a multipart form)
    any    GET  /ping                 "pong", to test the network from Safari

Each read gets a new id. A photo is used only if it carries the id of the read
that is waiting for it, so a photo that arrives after its read timed out can never
answer the next read. A new request replaces an older one still waiting.

Photos are decoded with the EXIF orientation tag ignored (Shortcuts tags photos as
portrait whatever the phone's real orientation), then turned by ``rotate`` degrees
clockwise to suit how the phone is mounted.

vision.yaml:
    side_camera:
      source: snapshot
      snapshot:
        port: 8765          # the Shortcut's URLs use http://<laptop-ip>:<port>/...
        host: 0.0.0.0
        hold_s: 20          # how long one /wait poll is held open
        timeout_s: 15       # a photo slower than this fails the read
        rotate: 0           # 0 | 90 | 180 | 270, clockwise, applied after decoding
        warmup: true        # take one throwaway photo at start-up (camera wake-up)
        require_id: true    # false accepts uploads without ?req= (old Shortcut)
"""
import itertools
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
_DISCONNECTS = (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)


def decode_photo(data: bytes, rotate: int = 0) -> Optional[np.ndarray]:
    """JPEG/PNG bytes -> BGR image, EXIF orientation ignored, then rotated clockwise."""
    if not data:
        return None
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        return None
    code = ROTATIONS[int(rotate) % 360]
    return img if code is None else cv2.rotate(img, code)


def strip_multipart(body: bytes, ctype: str) -> bytes:
    """The file part of a multipart/form-data body (Shortcuts "Form" request body)."""
    if "multipart/form-data" not in ctype or "boundary=" not in ctype:
        return body
    boundary = ctype.split("boundary=")[1].split(";")[0].strip('"').encode()
    for part in body.split(b"--" + boundary):
        head, sep, payload = part.partition(b"\r\n\r\n")
        if sep and (b"filename=" in head or b"Content-Type: image" in head):
            return payload.rstrip(b"\r\n")
    return body


@dataclass
class _Request:
    id: int
    tag: str
    created: float = field(default_factory=time.monotonic)
    dispatched: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    frame: Optional[np.ndarray] = None
    message: str = ""


class SnapshotCamera:
    def __init__(self, cfg: Optional[dict] = None, log: Optional[Callable[[str], None]] = None):
        cfg = dict(cfg or {})
        self.host = str(cfg.get("host", "0.0.0.0"))
        self.hold_s = float(cfg.get("hold_s", 20.0))
        self.timeout_s = float(cfg.get("timeout_s", 15.0))
        self.rotate = int(cfg.get("rotate", 0)) % 360
        if self.rotate not in ROTATIONS:
            raise ValueError(f"snapshot.rotate must be 0, 90, 180 or 270, not {cfg.get('rotate')!r}")
        self.require_id = bool(cfg.get("require_id", True))
        self.log = log or (lambda msg: print(f"[snapshot] {msg}"))
        self._ids = itertools.count(1)
        self._cond = threading.Condition()
        self._pending: Optional[_Request] = None
        self._capture_lock = threading.Lock()
        self.last_poll = 0.0
        self.last_frame: Optional[np.ndarray] = None
        self.last_stamp = 0.0
        self.photos = 0
        self._server = ThreadingHTTPServer((self.host, int(cfg.get("port", 8765))), self._handler())
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.log(f"waiting for the phone on port {self.port} (rotate {self.rotate}, timeout {self.timeout_s:.0f} s)")
        if cfg.get("warmup", True):
            threading.Thread(target=self._warmup, args=(float(cfg.get("warmup_timeout_s", 300.0)),),
                             daemon=True).start()

    # ------------------------------------------------------------------ public API
    @property
    def phone_seen_s(self) -> float:
        """Seconds since the phone last polled (inf if never)."""
        return float("inf") if not self.last_poll else time.monotonic() - self.last_poll

    def capture(self, timeout: Optional[float] = None, tag: str = "read") -> Tuple[Optional[np.ndarray], str]:
        """Ask the phone for one photo and wait for it. Returns (frame or None, message)."""
        timeout = self.timeout_s if timeout is None else float(timeout)
        with self._capture_lock:                       # reads never overlap; a newer one replaces warm-up
            req = self._new_request(tag)
        t0 = time.monotonic()
        if not req.done.wait(timeout):
            with self._cond:
                if self._pending is req:
                    self._pending = None
            seen = self.phone_seen_s
            if seen > self.hold_s + 10:
                why = ("the phone has never polled" if seen == float("inf")
                       else f"the phone last polled {seen:.0f} s ago")
                return None, f"no photo within {timeout:.0f} s: {why} (is the Shortcut running?)"
            if req.dispatched:
                return None, (f"no photo within {timeout:.0f} s after the phone was asked "
                              f"(check the phone for an 'Allow' prompt)")
            return None, f"no photo within {timeout:.0f} s (the phone did not pick up the request)"
        if req.frame is None:
            return None, req.message or "photo failed"
        return req.frame, f"photo {req.frame.shape[1]}x{req.frame.shape[0]} in {time.monotonic() - t0:.1f} s"

    def read(self):
        """Last photo (FrameSource-style); never triggers a new one."""
        with self._cond:
            return (None if self.last_frame is None else self.last_frame.copy()), self.last_stamp

    def close(self):
        with self._cond:
            if self._pending is not None:
                self._finish(self._pending, None, "camera closed")
            self._cond.notify_all()
        self._server.shutdown()
        self._server.server_close()

    # ------------------------------------------------------------------ internals
    def _warmup(self, timeout: float):
        frame, msg = self.capture(timeout, tag="warmup")
        self.log(f"warm-up photo: {msg}" if frame is not None else f"warm-up skipped: {msg}")

    def _new_request(self, tag: str) -> _Request:
        with self._cond:
            old = self._pending
            if old is not None:
                self._finish(old, None, f"replaced by a newer request ({tag})")
            req = _Request(next(self._ids), tag)
            self._pending = req
            self._cond.notify_all()
            return req

    @staticmethod
    def _finish(req: _Request, frame, message: str):
        req.frame, req.message = frame, message
        req.done.set()

    def _on_wait(self) -> str:
        self.last_poll = time.monotonic()
        with self._cond:
            ok = self._cond.wait_for(lambda: self._pending is not None and not self._pending.dispatched,
                                     timeout=self.hold_s)
            if not ok:
                return "idle"
            self._pending.dispatched = True
            return f"go-{self._pending.id}"

    def _undispatch(self, answer: str):
        """The phone hung up before it got 'go': let its next poll pick the request up."""
        with self._cond:
            if self._pending is not None and answer == f"go-{self._pending.id}":
                self._pending.dispatched = False
                self._cond.notify_all()

    def _on_upload(self, data: bytes, req_param: Optional[str]) -> Tuple[int, str]:
        self.last_poll = time.monotonic()
        rid = None
        if req_param:
            try:
                rid = int(req_param.strip().rsplit("-", 1)[-1])
            except ValueError:
                return 400, f"bad req {req_param!r}"
        with self._cond:
            req = self._pending
            if req is None:
                self.log(f"discarded a photo for request {rid}: no read is waiting (too late?)")
                return 200, "stale"
            if rid is None and self.require_id:
                self.log("discarded a photo without ?req=<id> (update the Shortcut, or set require_id: false)")
                return 400, "missing req"
            if rid is not None and rid != req.id:
                self.log(f"discarded a late photo for request {rid} (waiting for {req.id})")
                return 200, "stale"
            frame = decode_photo(data, self.rotate)
            if frame is None:
                kind = "HEIC" if data[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1") else "unreadable"
                self._finish(req, None, f"photo is {kind}: add 'Convert Image to JPEG' to the Shortcut")
                self._pending = None
                return 415, "not a JPEG"
            self.last_frame, self.last_stamp = frame, time.monotonic()
            self.photos += 1
            self._finish(req, frame, "ok")
            self._pending = None
            return 200, "saved"

    def _handler(self):
        cam = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code: int, text: str) -> bool:
                body = text.encode()
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return True
                except _DISCONNECTS:
                    return False

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/ping":
                    self._send(200, "pong")
                elif path == "/wait":
                    answer = cam._on_wait()
                    if not self._send(200, answer) and answer != "idle":
                        cam._undispatch(answer)
                else:
                    self._send(404, "not found")

            def do_POST(self):
                url = urlparse(self.path)
                if url.path != "/upload":
                    self._send(404, "not found")
                    return
                try:
                    data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                except _DISCONNECTS:
                    return
                data = strip_multipart(data, self.headers.get("Content-Type", ""))
                req = parse_qs(url.query).get("req", parse_qs(url.query).get("id", [None]))[0]
                code, text = cam._on_upload(data, req)
                self._send(code, text)

        return Handler
