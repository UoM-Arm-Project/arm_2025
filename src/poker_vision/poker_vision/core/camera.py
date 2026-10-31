"""Frame sources: USB camera, video file, still image or RTSP, read on a background thread.

Only the newest frame is kept, so inference never works on a stale backlog.
``source`` in vision.yaml:
    0 | "/dev/video2"          V4L2 camera (index or device path)
    "file.mp4" | "rtsp://..."  video file / network stream
    "image.jpg"                still image, returned every time (handy for tuning)
    "ros:/camera/image_raw"    handled by the node (subscribes to sensor_msgs/Image)
"""
import os
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def calibration(cam_cfg: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    K = cam_cfg.get("camera_matrix")
    D = cam_cfg.get("dist_coeffs")
    return (None if K is None else np.asarray(K, dtype=np.float64).reshape(3, 3),
            None if D is None else np.asarray(D, dtype=np.float64).reshape(-1))


class FrameSource:
    def __init__(self, cam_cfg: dict, base_dir: str = ""):
        src = cam_cfg.get("source", 0)
        self.still: Optional[np.ndarray] = None
        self.cap = None
        self._frame: Optional[np.ndarray] = None
        self._stamp = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        if isinstance(src, str) and src.lower().endswith(IMAGE_EXT):
            path = src if os.path.isabs(src) or not base_dir else os.path.join(base_dir, src)
            self.still = cv2.imread(os.path.expanduser(path))
            if self.still is None:
                raise FileNotFoundError(f"Cannot read image {path}")
            return
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        if isinstance(src, str) and src.startswith("/dev/video") and src[10:].isdigit():
            src = int(src[10:])
        backend = cv2.CAP_V4L2 if isinstance(src, int) or str(src).startswith("/dev/video") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(src, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera/video source {src!r}")
        if cam_cfg.get("fourcc"):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*str(cam_cfg["fourcc"])))
        if cam_cfg.get("width"):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam_cfg["width"]))
        if cam_cfg.get("height"):
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam_cfg["height"]))
        if cam_cfg.get("fps"):
            self.cap.set(cv2.CAP_PROP_FPS, float(cam_cfg["fps"]))
        if cam_cfg.get("autofocus") is False:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)          # as in the notebook: fixed focus
        if cam_cfg.get("focus") is not None:
            self.cap.set(cv2.CAP_PROP_FOCUS, float(cam_cfg["focus"]))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._loop_video = bool(cam_cfg.get("loop", True))
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                if self._loop_video and self.cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame, self._stamp = frame, time.monotonic()

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        if self.still is not None:
            return self.still.copy(), time.monotonic()
        with self._lock:
            return (None if self._frame is None else self._frame.copy()), self._stamp

    def close(self):
        self._stop.set()
        if self.cap is not None:
            self.cap.release()
