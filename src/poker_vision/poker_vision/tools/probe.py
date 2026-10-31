"""vision_probe: run the same one-shot reads as the vision_server, without ROS.

    ros2 run poker_vision vision_probe --image table.jpg --zone board          # read the board zone
    ros2 run poker_vision vision_probe --image table.jpg --zone bot_hole --expected 2
    ros2 run poker_vision vision_probe --image side.jpg --pot                  # value the pot
    ros2 run poker_vision vision_probe --image table.jpg --click               # click -> table/arm coords
    ros2 run poker_vision vision_probe --camera --zone board                   # live: press r to read
    ros2 run poker_vision vision_probe --image table.jpg --zone board --full   # no crop, for comparison

Use --click (zones are drawn) to measure the zone rectangles for vision.yaml.
"""
import argparse
import os
import sys

import cv2
import numpy as np

from ..core.camera import FrameSource, calibration
from ..core.config import load_config
from ..core.detector import make_backend
from ..core.reader import CardRead, CardReader, PotReader, draw_cards, draw_pot
from ..core.table_frame import TableFrame, arm_from_table
from ..core.zones import load_zones


def default_config() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("poker_vision"), "config", "vision.yaml")
    except Exception:
        return os.path.join(os.path.dirname(__file__), "..", "..", "config", "vision.yaml")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=default_config())
    ap.add_argument("--image", help="still image instead of the configured camera")
    ap.add_argument("--camera", action="store_true", help="live configured camera; press r to read, q to quit")
    ap.add_argument("--zone", default="board", help="card zone to read (board, bot_hole, seat_3 ...)")
    ap.add_argument("--expected", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="run the card model on the whole image (no zone crop)")
    ap.add_argument("--pot", action="store_true", help="side camera pot read instead of cards")
    ap.add_argument("--click", action="store_true", help="click points to print table/arm coordinates")
    ap.add_argument("--no-model", action="store_true", help="only the ArUco table frame and zones")
    ap.add_argument("--save", help="write the annotated image here")
    a = ap.parse_args(argv)
    if not a.image and not a.camera:
        ap.error("give --image PATH or --camera")

    cfg = load_config(a.config)
    base = cfg["_base_dir"]
    cam_cfg = dict(cfg["side_camera" if a.pot else "overhead_camera"])
    if a.image:
        cam_cfg["source"] = a.image
    src = FrameSource(cam_cfg, base)
    zones = load_zones(cfg)
    K, D = calibration(cam_cfg)
    table = None if a.pot else TableFrame(cfg.get("table_frame", {}), K, D)
    arm_cfg = cfg.get("arm_from_table")

    reader = None
    if not a.no_model:
        if a.pot:
            reader = PotReader(make_backend(cfg["pot_model"], base), cfg["pot_model"])
        else:
            mcfg = dict(cfg["card_model"])
            if a.full:
                mcfg["crop_to_zone"] = False
            reader = CardReader(table, make_backend(mcfg, base), zones, mcfg)

    def read_once(frame):
        if a.pot:
            r = reader.read(frame)
            if r.reading is not None:
                print(f"pot {r.reading.value:.0f}  chips {r.reading.counts}  "
                      f"confidence {r.reading.confidence:.2f}  ({r.inference_ms:.0f} ms)")
            return draw_pot(frame, r)
        if reader is None:
            table.update(frame)
            r = CardRead(table.ready, "table frame only" if table.ready else "no ArUco marker found")
        else:
            r = reader.read([frame], a.zone, a.expected)
            print(f"zone {a.zone}: {'OK' if r.success else 'FAIL'} - {r.message} ({r.inference_ms:.0f} ms)")
            for c in r.cards:
                where = f"table ({c.table[0]:+.4f}, {c.table[1]:+.4f})"
                if arm_cfg:
                    arm = arm_from_table(c.table, arm_cfg)[0]
                    where += f" arm ({arm[0]:+.4f}, {arm[1]:+.4f})"
                print(f"  {c.code}: {c.corners} corner(s), conf {c.conf:.2f}, {where}"
                      + (f"  CONFLICT {c.conflict}" if c.conflict else ""))
            if r.crop:
                x0, y0, x1, y1 = r.crop
                print(f"  crop {x1 - x0}x{y1 - y0} px at ({x0}, {y0})")
        if table is not None and table.ready:
            err = table.reproj_error_m
            print(f"markers {table.visible_ids}; fit error {'n/a' if np.isnan(err) else f'{err * 1000:.2f} mm'}")
        return draw_cards(frame, r, table, zones)

    state = {"click": None}

    def on_mouse(event, u, v, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (u, v)
            if table is not None and table.ready:
                t = table.to_table([[u, v]])[0]
                msg = f"pixel ({u}, {v}) -> table x={t[0]:+.4f} y={t[1]:+.4f} m"
                if arm_cfg:
                    arm = arm_from_table(t, arm_cfg)[0]
                    msg += f" | arm x={arm[0]:+.4f} y={arm[1]:+.4f} m"
                print(msg)
            else:
                print(f"pixel ({u}, {v})")

    frame = None
    while frame is None:
        frame, _ = src.read()
        cv2.waitKey(30) if frame is None else None
    annotated = read_once(frame)
    if a.save:
        cv2.imwrite(a.save, annotated)
    if not (a.camera or a.click):
        src.close()
        return 0

    cv2.namedWindow("vision_probe", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("vision_probe", on_mouse)
    while True:
        if a.camera:
            frame, _ = src.read()
            shown = frame.copy() if frame is not None else annotated
            if table is not None and frame is not None:
                shown = draw_cards(frame, CardRead(True, "live - press r to read"), table, zones)
        else:
            shown = annotated.copy()
        if state["click"] is not None:
            cv2.drawMarker(shown, state["click"], (255, 0, 255), cv2.MARKER_CROSS, 25, 2)
        cv2.imshow("vision_probe", shown if not a.camera else (annotated if state.get("hold") else shown))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r") and frame is not None:
            annotated = read_once(frame)
            state["hold"] = True
            if a.save:
                cv2.imwrite(a.save, annotated)
        elif key == ord("l"):
            state["hold"] = False                      # back to the live view
    src.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
