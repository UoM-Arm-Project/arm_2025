"""poker_card_detector: overhead camera -> /vision/cards (poker_interfaces/CardDetectionArray).

Every frame: refresh the ArUco table frame (every ``refresh_every`` frames),
run the card-corner model, merge corners into cards, convert each card centre
to table metres and publish. An empty array is published when no cards are
seen, so poker_logic's debouncer knows the table is clear. Nothing is
published until a marker has been seen once (no table frame = no positions).

Zones are assigned by poker_game_manager from config/vision_zones.yaml (same
table frame), so CardDetection.zone is left empty here.
"""
import cv2
import rclpy

from poker_interfaces.msg import CardDetection, CardDetectionArray

from ..core.cards import pair_corners
from ..core.camera import calibration
from ..core.detector import make_backend
from ..core.table_frame import TableFrame
from .common import VisionNode


class CardDetectorNode(VisionNode):
    camera_key = "overhead_camera"
    debug_topic = "/vision/cards/debug"

    def __init__(self):
        super().__init__("poker_card_detector")
        K, D = calibration(self.cam_cfg)
        self.table = TableFrame(self.cfg.get("table_frame", {}), K, D)
        self.refresh_every = int(self.cfg.get("table_frame", {}).get("refresh_every", 5))
        mcfg = self.cfg["card_model"]
        self.model = make_backend(mcfg, self.base_dir)
        self.class_map = mcfg.get("class_map") or None
        self.pair_m = float(mcfg.get("pair_max_distance_m", 0.12))
        self.pair_px = float(mcfg.get("pair_max_distance_px", 500))
        if mcfg.get("warmup", True):
            self.model.warmup()
        self.pub = self.create_publisher(CardDetectionArray, "/vision/cards", 10)
        self._n = 0
        self.get_logger().info(f"card model: {mcfg.get('weights', mcfg.get('backend'))} "
                               f"on {mcfg.get('device', 'cpu')}")

    def process(self, frame):
        if self._n % max(1, self.refresh_every) == 0 or not self.table.ready:
            self.table.update(frame)
        self._n += 1
        if not self.table.ready:
            self.get_logger().warning("no ArUco table marker seen yet; not publishing card positions",
                                      throttle_duration_sec=5.0)
            return self._draw(frame, []) if self.debug_pub or self.show else None
        dets = self.model(frame)
        cards = pair_corners(dets, self.class_map, self.table.to_table, self.pair_m, self.pair_px)
        msg = CardDetectionArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "table"
        for c in cards:
            d = CardDetection()
            d.code, d.zone = c.code, ""
            d.x, d.y = float(c.table[0]), float(c.table[1])
            d.confidence = float(c.conf)
            msg.cards.append(d)
        self.pub.publish(msg)
        return self._draw(frame, cards) if self.debug_pub or self.show else None

    def _draw(self, frame, cards):
        img = frame.copy()
        self.table.draw(img)
        for c in cards:
            u, v = int(c.pixel[0]), int(c.pixel[1])
            cv2.circle(img, (u, v), 6, (0, 255, 255), -1)
            label = c.code if c.table is None else f"{c.code} ({c.table[0]:+.3f},{c.table[1]:+.3f})"
            cv2.putText(img, label, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if not self.table.ready:
            cv2.putText(img, "NO TABLE MARKER", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return img

    def stats(self):
        return (f"model {self.model.last_latency_ms:.0f} ms; markers {self.table.visible_ids}; "
                f"fit error {self.table.reproj_error_m * 1000:.1f} mm")


def main(args=None):
    rclpy.init(args=args)
    node = CardDetectorNode()
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
