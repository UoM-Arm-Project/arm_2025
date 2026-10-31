"""Shared plumbing for the vision nodes: config, frame source, debug output.

Images are converted to/from sensor_msgs/Image with numpy directly instead of
cv_bridge, so the nodes work from the workspace venv whatever numpy version
ultralytics pulls in.
"""
import os
import time
from typing import Optional

import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import Image

from ..core.camera import FrameSource
from ..core.config import load_config

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


class VisionNode(Node):
    """Loads the YAML, opens the camera section, runs ``process(frame)`` at ``rate_hz``."""

    camera_key = ""
    debug_topic = ""

    def __init__(self, name: str):
        super().__init__(name)
        self.declare_parameter("config_file", "")
        path = self.get_parameter("config_file").value or os.path.join(
            get_package_share_directory("poker_vision"), "config", "vision.yaml")
        self.cfg = load_config(path)
        self.base_dir = self.cfg["_base_dir"]
        self.cam_cfg = self.cfg[self.camera_key]
        self.frame_id = self.cam_cfg.get("frame_id", self.camera_key)
        self.show = bool(self.cfg.get("debug", {}).get("show_window", False))
        self.publish_debug = bool(self.cfg.get("debug", {}).get("publish_image", True))
        self.debug_pub = self.create_publisher(Image, self.debug_topic, 1) if self.publish_debug else None
        self.get_logger().info(f"{name}: config {path}")

        self.source: Optional[FrameSource] = None
        self._ros_frame = None
        src = self.cam_cfg.get("source", 0)
        if isinstance(src, str) and src.startswith("ros:"):
            topic = src[4:]
            self.create_subscription(Image, topic, self._on_image, 1)
            self.get_logger().info(f"subscribing to {topic}")
        else:
            self.source = FrameSource(self.cam_cfg, self.base_dir)
            self.get_logger().info(f"camera source {src!r}")
        self._frames = 0
        self._t_stats = time.monotonic()
        self._busy = False
        self.create_timer(1.0 / float(self.cam_cfg.get("rate_hz", 10.0)), self._tick)

    def _on_image(self, msg: Image):
        try:
            self._ros_frame = image_msg_to_bgr(msg)
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)

    def _tick(self):
        if self._busy:
            return
        frame = self._ros_frame if self.source is None else self.source.read()[0]
        if frame is None:
            self.get_logger().warning("no camera frame yet", throttle_duration_sec=5.0)
            return
        self._busy = True
        try:
            debug = self.process(frame)
        except Exception as e:  # keep the node alive on a bad frame
            self.get_logger().error(f"processing failed: {e!r}", throttle_duration_sec=2.0)
            debug = None
        finally:
            self._busy = False
        if debug is not None:
            if self.debug_pub is not None:
                self.debug_pub.publish(bgr_to_image_msg(debug, self.get_clock().now().to_msg(), self.frame_id))
            if self.show:
                cv2.imshow(self.get_name(), debug)
                cv2.waitKey(1)
        self._frames += 1
        if time.monotonic() - self._t_stats >= 10.0:
            fps = self._frames / (time.monotonic() - self._t_stats)
            self.get_logger().info(f"{fps:.1f} frames/s; {self.stats()}")
            self._frames, self._t_stats = 0, time.monotonic()

    def process(self, frame: np.ndarray):
        raise NotImplementedError

    def stats(self) -> str:
        return ""

    def destroy_node(self):
        if self.source is not None:
            self.source.close()
        super().destroy_node()
