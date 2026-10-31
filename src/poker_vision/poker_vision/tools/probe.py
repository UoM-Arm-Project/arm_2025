"""vision_probe: run the vision pipeline on one image / camera without ROS.

Replaces the notebook workflow:
    ros2 run poker_vision vision_probe --image table.jpg                # cards + table coords
    ros2 run poker_vision vision_probe --image side.jpg --pot           # pot value
    ros2 run poker_vision vision_probe --image table.jpg --click        # click -> table/arm coords
    ros2 run poker_vision vision_probe --camera                         # live overhead camera
Use --click to measure the zone rectangles for poker_logic's vision_zones.yaml.
"""
import argparse
import os
import sys

import cv2
import numpy as np

from ..core.camera import FrameSource, calibration
from ..core.cards import pair_corners
from ..core.config import load_config
from ..core.detector import make_backend
from ..core.pot import PotEstimator
from ..core.table_frame import TableFrame, arm_from_table


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
    ap.add_argument("--camera", action="store_true", help="use the configured camera (live)")
    ap.add_argument("--pot", action="store_true", help="run the side-camera pot model instead of cards")
    ap.add_argument("--click", action="store_true", help="click points to print table/arm coordinates")
    ap.add_argument("--no-model", action="store_true", help="only the ArUco table frame")
    ap.add_argument("--save", help="write the annotated image here")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    base = cfg["_base_dir"]
    cam_key = "side_camera" if a.pot else "overhead_camera"
    cam_cfg = dict(cfg[cam_key])
    if a.image:
        cam_cfg["source"] = a.image
    elif not a.camera:
        ap.error("give --image PATH or --camera")
    src = FrameSource(cam_cfg, base)
    K, D = calibration(cam_cfg)
    table = TableFrame(cfg.get("table_frame", {}), K, D) if not a.pot else None
    model = None if a.no_model else make_backend(cfg["pot_model" if a.pot else "card_model"], base)
    arm_cfg = cfg.get("arm_from_table")
    state = {"click": None}

    def on_mouse(event, u, v, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (u, v)
            if table is not None and table.ready:
                t = table.to_table([[u, v]])[0]
                arm = arm_from_table(t, arm_cfg)[0]
                print(f"pixel ({u}, {v}) -> table x={t[0]:+.4f} y={t[1]:+.4f} m"
                      + (f" | arm x={arm[0]:+.4f} y={arm[1]:+.4f} m" if arm_cfg else ""))
            else:
                print(f"pixel ({u}, {v})")

    live = a.camera or a.click
    if live:
        cv2.namedWindow("vision_probe")
        cv2.setMouseCallback("vision_probe", on_mouse)
    printed = False
    annotated = None
    while True:
        if annotated is not None and not a.camera:      # still image: compute once, then just display
            img = annotated.copy()
            if state["click"] is not None:
                cv2.drawMarker(img, state["click"], (255, 0, 255), cv2.MARKER_CROSS, 25, 2)
            cv2.imshow("vision_probe", img)
            if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                break
            continue
        frame, _ = src.read()
        if frame is None:
            cv2.waitKey(30)
            continue
        img = frame.copy()
        if table is not None:
            table.update(frame)
            table.draw(img)
            if not printed:
                if table.ready:
                    err = table.reproj_error_m
                    print(f"markers {table.visible_ids}; mode {table.mode}; "
                          f"fit error {'n/a' if np.isnan(err) else f'{err * 1000:.2f} mm'}")
                else:
                    print("no configured ArUco marker found (check dictionary / ids / lighting)")
        if model is not None:
            dets = model(frame)
            if a.pot:
                r = PotEstimator(cfg["pot_model"]).estimate(dets)
                for d in dets:
                    if d.polygon is not None and len(d.polygon) >= 3:
                        cv2.polylines(img, [d.polygon.astype(np.int32)], True, (0, 255, 0), 2)
                cv2.putText(img, f"POT {r.value:.0f}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
                if not printed:
                    print(f"pot value {r.value:.0f} (confidence {r.confidence:.2f}); stacks {r.stacks}")
            else:
                cards = pair_corners(dets, cfg["card_model"].get("class_map"),
                                     table.to_table if table.ready else None)
                for c in cards:
                    u, v = int(c.pixel[0]), int(c.pixel[1])
                    cv2.circle(img, (u, v), 6, (0, 255, 255), -1)
                    cv2.putText(img, c.code, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if not printed:
                    print(f"{len(dets)} corner detections -> {len(cards)} cards "
                          f"({model.last_latency_ms:.0f} ms)")
                    for c in cards:
                        where = "" if c.table is None else f" table ({c.table[0]:+.4f}, {c.table[1]:+.4f})"
                        if c.table is not None and arm_cfg:
                            arm = arm_from_table(c.table, arm_cfg)[0]
                            where += f" arm ({arm[0]:+.4f}, {arm[1]:+.4f})"
                        print(f"  {c.code}: {c.corners} corner(s), conf {c.conf:.2f}, "
                              f"pixel ({c.pixel[0]:.0f}, {c.pixel[1]:.0f}){where}")
        printed = True             # print the text summary once; the window keeps updating
        annotated = img.copy()
        if a.save:
            cv2.imwrite(a.save, img)
        if not live:
            break
        if state["click"] is not None:
            cv2.drawMarker(img, state["click"], (255, 0, 255), cv2.MARKER_CROSS, 25, 2)
        cv2.imshow("vision_probe", img)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    src.close()
    if live:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
