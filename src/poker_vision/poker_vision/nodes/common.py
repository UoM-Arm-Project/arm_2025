"""Shared plumbing: image <-> sensor_msgs/Image without cv_bridge, and frame grabbing.

numpy conversion (instead of cv_bridge) keeps the nodes working from the
workspace venv whatever numpy version ultralytics installs.
"""
import threading
import time
from typing import List, Optional

import cv2
import numpy as np
from sensor_msgs.msg import Image

from ..core.camera import FrameSource

ENCODINGS = {"bgr8": (3, None), "rgb8": (3, cv2.COLOR_RGB2BGR), "mono8": (1, cv2.COLOR_GRAY2BGR),
             "bgra8": (4, cv2.COLOR_BGRA2BGR), "rgba8": (4, cv2.COLOR_RGBA2BGR)}


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    if msg.encoding not in ENCODINGS:
        raise ValueError(f"Unsupported image encoding {msg.encoding!r} (use bgr8/rgb8/mono8/bgra8/rgba8)")
    ch, conv = ENCODINGS[msg.encoding]
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    img = buf.reshape(msg.height, msg.step)[:, : msg.width * ch].reshape(msg.height, msg.width, ch)
    if ch == 1:
        img = img[:, :, 0]
    return cv2.cvtColor(img, conv) if conv is not None else img.copy()


def bgr_to_image_msg(img: np.ndarray, stamp, frame_id: str) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = img.shape[:2]
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(img).tobytes()
    return msg


class Grabber:
    """Latest frame from a camera/file (FrameSource thread) or a ROS image topic.

    No inference happens here: grabbing keeps a fresh still ready so a read
    request never waits for camera start-up or auto-exposure.
    """

    def __init__(self, node, cam_cfg: dict, base_dir: str, callback_group=None):
        self.cfg = cam_cfg
        self.source: Optional[FrameSource] = None
        self._frame, self._stamp = None, 0.0
        self._lock = threading.Lock()
        src = cam_cfg.get("source", 0)
        if isinstance(src, str) and src.startswith("ros:"):
            node.create_subscription(Image, src[4:], self._on_image, 1, callback_group=callback_group)
            self.desc = f"topic {src[4:]}"
        else:
            self.source = FrameSource(cam_cfg, base_dir)
            self.desc = f"source {src!r}"

    def _on_image(self, msg: Image):
        img = image_msg_to_bgr(msg)
        with self._lock:
            self._frame, self._stamp = img, time.monotonic()

    def latest(self):
        if self.source is not None:
            return self.source.read()
        with self._lock:
            return (None if self._frame is None else self._frame.copy()), self._stamp

    def fresh_frames(self, count: int = 1, after: Optional[float] = None, timeout: float = 2.0) -> List[np.ndarray]:
        """``count`` distinct frames captured after ``after`` (default: now)."""
        after = time.monotonic() if after is None else after
        deadline = time.monotonic() + timeout
        frames, last = [], after
        while len(frames) < count and time.monotonic() < deadline:
            frame, stamp = self.latest()
            if frame is not None and (stamp > last or (self.source is not None and self.source.still is not None)):
                frames.append(frame)
                last = stamp
                if self.source is not None and self.source.still is not None:
                    frames.extend([frame] * (count - len(frames)))   # still image: same frame
            else:
                time.sleep(0.01)
        return frames

    def close(self):
        if self.source is not None:
            self.source.close()
