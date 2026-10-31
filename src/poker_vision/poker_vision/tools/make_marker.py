"""make_marker: write printable ArUco markers (from the notebook's first cell).

    ros2 run poker_vision make_marker --ids 0 1 2 3 --px 600 --out ~/markers

Print at 100% scale and measure the black square: that side length (metres)
is ``size`` in vision.yaml.
"""
import argparse
import os
import sys

import cv2

from ..core.table_frame import aruco_dictionary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", type=int, nargs="+", default=[0])
    ap.add_argument("--dictionary", default="DICT_4X4_50")
    ap.add_argument("--px", type=int, default=600, help="marker side in pixels")
    ap.add_argument("--border", type=int, default=60, help="white quiet zone in pixels")
    ap.add_argument("--out", default=".")
    a = ap.parse_args(argv)
    os.makedirs(os.path.expanduser(a.out), exist_ok=True)
    d = aruco_dictionary(a.dictionary)
    for i in a.ids:
        img = cv2.aruco.generateImageMarker(d, i, a.px)
        img = cv2.copyMakeBorder(img, a.border, a.border, a.border, a.border, cv2.BORDER_CONSTANT, value=255)
        path = os.path.join(os.path.expanduser(a.out), f"aruco_{a.dictionary}_{i}.png")
        cv2.imwrite(path, img)
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
