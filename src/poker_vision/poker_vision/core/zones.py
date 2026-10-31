"""Named table zones (board, bot_hole, seat_<n>) in the ArUco table frame.

Zones live in vision.yaml and do two jobs:
  * crop: the zone rectangle (+ margin) is projected into the overhead image
    and only that crop goes to the card model, so a 640-px model sees the
    cards large even when the camera covers the whole table;
  * filter: only cards whose centre falls inside the zone are returned.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Zone:
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, x: float, y: float, tol: float = 0.0) -> bool:
        return (self.x_min - tol <= x <= self.x_max + tol) and (self.y_min - tol <= y <= self.y_max + tol)

    def corners(self, margin: float = 0.0) -> np.ndarray:
        return np.array([[self.x_min - margin, self.y_min - margin], [self.x_max + margin, self.y_min - margin],
                         [self.x_max + margin, self.y_max + margin], [self.x_min - margin, self.y_max + margin]])


def load_zones(cfg: dict) -> Dict[str, Zone]:
    zones = {}
    for z in cfg.get("zones", []):
        zones[z["name"]] = Zone(z["name"], float(z["x_min"]), float(z["x_max"]),
                                float(z["y_min"]), float(z["y_max"]))
    return zones


def crop_box(zone: Zone, to_pixel, shape, margin_m: float = 0.03,
             min_px: int = 64) -> Optional[Tuple[int, int, int, int]]:
    """Pixel bounding box (x0, y0, x1, y1) of the zone + margin, clipped to the image."""
    h, w = shape[:2]
    px = np.asarray(to_pixel(zone.corners(margin_m)))
    x0, y0 = np.floor(px.min(axis=0)).astype(int)
    x1, y1 = np.ceil(px.max(axis=0)).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < min_px or y1 - y0 < min_px:
        return None
    return int(x0), int(y0), int(x1), int(y1)
