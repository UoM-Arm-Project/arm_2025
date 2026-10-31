"""poker_vision_server: on-demand card and pot reads (no continuous inference).

Services
    /vision/read_cards  poker_interfaces/ReadCards   zone + expected count -> cards (table metres)
    /vision/read_pot    poker_interfaces/ReadPot     -> pot value + chips per colour
Publishes (one message per read, for logging / rqt)
    /vision/cards        poker_interfaces/CardDetectionArray
    /vision/pot          poker_interfaces/PotEstimate
    /vision/cards/debug  sensor_msgs/Image   annotated still of the last card read
    /vision/pot/debug    sensor_msgs/Image   annotated still of the last pot read

The cameras keep grabbing frames on background threads (cheap), so a fresh
still is ready the moment poker_game_manager asks. Models only run inside a
service call. Each service has its own callback group, so a slow pot read on
CPU does not block a card read.
"""
import os
import time

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image

from poker_interfaces.msg import CardDetection, CardDetectionArray, PotEstimate
from poker_interfaces.srv import ReadCards, ReadPot

from ..core.camera import calibration
from ..core.config import load_config
from ..core.detector import make_backend
from ..core.reader import CardReader, PotReader, draw_cards, draw_pot
from ..core.table_frame import TableFrame
from ..core.zones import load_zones
from .common import Grabber, bgr_to_image_msg


class VisionServer(Node):
    def __init__(self):
        super().__init__("poker_vision_server")
        self.declare_parameter("config_file", "")
        path = self.get_parameter("config_file").value or os.path.join(
            get_package_share_directory("poker_vision"), "config", "vision.yaml")
        self.cfg = load_config(path)
        base = self.cfg["_base_dir"]
        dbg = self.cfg.get("debug", {})
        self.publish_debug = bool(dbg.get("publish_image", True))
        self.save_dir = os.path.expanduser(str(dbg.get("save_dir") or ""))
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(f"config {path}")

        self.card_reader = self.pot_reader = None
        if self.cfg.get("card_model", {}).get("enabled", True):
            cam = self.cfg["overhead_camera"]
            group = MutuallyExclusiveCallbackGroup()
            self.overhead = Grabber(self, cam, base, group)
            K, D = calibration(cam)
            self.table = TableFrame(self.cfg.get("table_frame", {}), K, D)
            self.zones = load_zones(self.cfg)
            mcfg = self.cfg["card_model"]
            self.card_model = make_backend(mcfg, base)
            if mcfg.get("warmup", True):
                self.card_model.warmup()
            self.card_reader = CardReader(self.table, self.card_model, self.zones, mcfg)
            self.card_shots = int(mcfg.get("shots", 1))
            self.create_service(ReadCards, "/vision/read_cards", self._read_cards, callback_group=group)
            self.cards_pub = self.create_publisher(CardDetectionArray, "/vision/cards", 10)
            self.cards_dbg = self.create_publisher(Image, "/vision/cards/debug", 1)
            self.get_logger().info(f"cards: {mcfg.get('weights', mcfg.get('backend'))} on "
                                   f"{mcfg.get('device', 'cpu')}; overhead {self.overhead.desc}; "
                                   f"zones {sorted(self.zones)}")
        if self.cfg.get("pot_model", {}).get("enabled", True):
            cam = self.cfg["side_camera"]
            group = MutuallyExclusiveCallbackGroup()
            self.side = Grabber(self, cam, base, group)
            pcfg = self.cfg["pot_model"]
            self.pot_model = make_backend(pcfg, base)
            if pcfg.get("warmup", True):
                self.pot_model.warmup()
            self.pot_reader = PotReader(self.pot_model, pcfg)
            self.create_service(ReadPot, "/vision/read_pot", self._read_pot, callback_group=group)
            self.pot_pub = self.create_publisher(PotEstimate, "/vision/pot", 10)
            self.pot_dbg = self.create_publisher(Image, "/vision/pot/debug", 1)
            self.get_logger().info(f"pot: {pcfg.get('weights', pcfg.get('backend'))} on "
                                   f"{pcfg.get('device', 'cpu')}; side {self.side.desc}; "
                                   f"classes {pcfg.get('classes')}; roi {pcfg.get('roi') or 'full frame'}")
        self.get_logger().info("vision_server ready (inference runs only on request)")

    # ------------------------------------------------------------------ cards
    def _read_cards(self, req, res):
        frames = self.overhead.fresh_frames(self.card_shots)
        if not frames:
            res.success, res.message = False, "no frame from the overhead camera"
            return res
        try:
            read = self.card_reader.read(frames, req.zone, int(req.expected))
        except Exception as e:  # never kill the server on one bad read
            res.success, res.message = False, f"card read error: {e!r}"
            self.get_logger().error(res.message)
            return res
        res.success, res.message, res.inference_ms = read.success, read.message, float(read.inference_ms)
        arr = CardDetectionArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.header.frame_id = "table"
        for c in read.cards:
            d = CardDetection()
            d.code, d.zone = c.code, req.zone
            d.x, d.y = float(c.table[0]), float(c.table[1])
            d.confidence = float(c.conf)
            res.cards.append(d)
            arr.cards.append(d)
        self.cards_pub.publish(arr)
        self.get_logger().info(f"read_cards({req.zone}, {req.expected}): "
                               f"{[c.code for c in read.cards]} in {read.inference_ms:.0f} ms; {read.message}")
        self._debug(self.cards_dbg, draw_cards(frames[-1], read, self.table, self.zones), f"cards_{req.zone}")
        return res

    # ------------------------------------------------------------------ pot
    def _read_pot(self, req, res):
        frames = self.side.fresh_frames(1)
        if not frames:
            res.success, res.message = False, "no frame from the side camera"
            return res
        try:
            read = self.pot_reader.read(frames[0])
        except Exception as e:
            res.success, res.message = False, f"pot read error: {e!r}"
            self.get_logger().error(res.message)
            return res
        r = read.reading
        res.success, res.message, res.inference_ms = read.success, read.message, float(read.inference_ms)
        res.value, res.confidence = float(r.value), float(r.confidence)
        res.chip_classes = list(r.counts)
        res.chip_counts = [int(v) for v in r.counts.values()]
        msg = PotEstimate()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.cfg["side_camera"].get("frame_id", "side_camera")
        msg.value, msg.confidence = res.value, res.confidence
        self.pot_pub.publish(msg)
        self.get_logger().info(f"read_pot: {r.value:.0f} {r.counts} in {read.inference_ms:.0f} ms")
        self._debug(self.pot_dbg, draw_pot(frames[0], read), "pot")
        return res

    # ------------------------------------------------------------------ debug output
    def _debug(self, pub, img, tag):
        if self.publish_debug:
            pub.publish(bgr_to_image_msg(img, self.get_clock().now().to_msg(), tag))
        if self.save_dir:
            cv2.imwrite(os.path.join(self.save_dir, f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}.jpg"), img)

    def destroy_node(self):
        for g in ("overhead", "side"):
            if hasattr(self, g):
                getattr(self, g).close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisionServer()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
