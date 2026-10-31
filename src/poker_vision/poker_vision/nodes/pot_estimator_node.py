"""poker_pot_estimator: side camera -> /vision/pot (poker_interfaces/PotEstimate).

Runs the chip segmentation model and converts masks to a chip value with the
rules in vision.yaml (``pot_model``). poker_logic takes the median of the
readings after each "turn done" press, so per-frame noise is fine.
"""
import cv2
import numpy as np
import rclpy

from poker_interfaces.msg import PotEstimate

from ..core.detector import make_backend
from ..core.pot import PotEstimator
from .common import VisionNode


class PotEstimatorNode(VisionNode):
    camera_key = "side_camera"
    debug_topic = "/vision/pot/debug"

    def __init__(self):
        super().__init__("poker_pot_estimator")
        mcfg = self.cfg["pot_model"]
        self.model = make_backend(mcfg, self.base_dir)
        self.estimator = PotEstimator(mcfg)
        if mcfg.get("warmup", True):
            self.model.warmup()
        self.pub = self.create_publisher(PotEstimate, "/vision/pot", 10)
        self.last = None
        self.get_logger().info(f"pot model: {mcfg.get('weights', mcfg.get('backend'))} "
                               f"on {mcfg.get('device', 'cpu')}, count_mode={self.estimator.mode}")

    def process(self, frame):
        dets = self.model(frame)
        reading = self.estimator.estimate(dets)
        self.last = reading
        msg = PotEstimate()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.value = float(reading.value)
        msg.confidence = float(reading.confidence)
        self.pub.publish(msg)
        if not (self.debug_pub or self.show):
            return None
        img = frame.copy()
        if self.estimator.roi is not None:
            cv2.polylines(img, [self.estimator.roi.astype(np.int32)], True, (255, 0, 0), 2)
        for d in dets:
            if d.polygon is not None and len(d.polygon) >= 3:
                cv2.polylines(img, [d.polygon.astype(np.int32)], True, (0, 255, 0), 2)
            else:
                x1, y1, x2, y2 = (int(v) for v in d.xyxy)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, d.name, (int(d.xyxy[0]), int(d.xyxy[1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, f"POT {reading.value:.0f}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        return img

    def stats(self):
        v = "-" if self.last is None else f"{self.last.value:.0f}"
        return f"model {self.model.last_latency_ms:.0f} ms; pot {v}"


def main(args=None):
    rclpy.init(args=args)
    node = PotEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
