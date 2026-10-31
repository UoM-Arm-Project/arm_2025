"""Pixel -> table-frame (metres) conversion from ArUco markers on the table.

Two modes, chosen in ``vision.yaml``:

``homography`` (default)
    Every visible configured marker contributes its 4 corners, whose positions
    on the table are known (centre x/y, yaw, side length). A plane homography
    maps image pixels straight onto the table plane, so camera tilt, rotation
    and perspective are handled. More markers spread over the table = better
    accuracy far from any single marker.

``scale``
    The original notebook method (``marker.py``): metres-per-pixel from one
    marker's perimeter, offsets measured from its centre, image-down = -y.
    Only correct for a camera looking straight down with the marker aligned
    to the image axes; kept for comparison/backwards compatibility.

Table frame convention (same as the notebook): origin at the reference
marker's centre (or wherever the markers' x/y place it), +x = image right,
+y = image up when the marker is upright, metres. ``poker_logic``'s
``vision_zones.yaml`` must use this frame.

Note on the notebook's ``solvePnP`` cell: its ``marker_corners_3d`` lists the
corners as TL, TR, BL, BR, but OpenCV returns (and IPPE_SQUARE expects)
TL, TR, BR, BL, so the pose printed there is wrong. The homography here uses
the correct order.
"""
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class MarkerSpec:
    id: int
    size: float          # side length, metres
    x: float = 0.0       # centre in the table frame
    y: float = 0.0
    yaw: float = 0.0     # rotation of the marker's own x axis from table +x, radians

    def corners_table(self) -> np.ndarray:
        """Corners in OpenCV order TL, TR, BR, BL (marker upright: +y is 'up')."""
        h = self.size / 2.0
        local = np.array([[-h, h], [h, h], [h, -h], [-h, -h]], dtype=np.float64)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + np.array([self.x, self.y])


def aruco_dictionary(name: str):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary {name!r}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


class TableFrame:
    def __init__(self, cfg: dict, camera_matrix=None, dist_coeffs=None):
        self.mode = cfg.get("mode", "homography")
        if self.mode not in ("homography", "scale"):
            raise ValueError("table_frame.mode must be 'homography' or 'scale'")
        self.markers: Dict[int, MarkerSpec] = {}
        for m in cfg.get("markers", [{"id": 0, "size": 0.05}]):
            spec = MarkerSpec(int(m["id"]), float(m["size"]), float(m.get("x", 0.0)),
                              float(m.get("y", 0.0)), math.radians(float(m.get("yaw_deg", 0.0))))
            self.markers[spec.id] = spec
        self.reference_id = int(cfg.get("reference_id", next(iter(self.markers))))
        self.min_markers = int(cfg.get("min_markers", 1))
        self.max_age_s = float(cfg.get("max_age_s", 0.0))      # 0 = keep last good fit forever
        self.lock_after = int(cfg.get("lock_after_frames", 0))  # 0 = keep refining every update
        self.smoothing = float(cfg.get("smoothing", 0.7))      # 0 = use each new fit as-is
        self.K = None if camera_matrix is None else np.asarray(camera_matrix, dtype=np.float64)
        self.dist = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        if self.dist is not None and not np.any(self.dist):
            self.dist = None
        params = cv2.aruco.DetectorParameters()
        if cfg.get("corner_refinement", True):
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(aruco_dictionary(cfg.get("dictionary", "DICT_4X4_50")), params)

        self.H: Optional[np.ndarray] = None       # pixel (undistorted) -> table
        self.H_inv: Optional[np.ndarray] = None
        self.updated_at = 0.0
        self.visible_ids: List[int] = []
        self.good_fits = 0
        self.locked = False
        self.last_corners = None
        self.last_ids = None
        self.reproj_error_m = float("nan")

    # ------------------------------------------------------------------ status
    @property
    def ready(self) -> bool:
        if self.H is None:
            return False
        return self.max_age_s <= 0 or self.locked or time.monotonic() - self.updated_at <= self.max_age_s

    # ------------------------------------------------------------------ fitting
    def _undistort(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if self.dist is None or self.K is None:
            return pts
        return cv2.undistortPoints(pts.reshape(-1, 1, 2), self.K, self.dist, P=self.K).reshape(-1, 2)

    def update(self, image: np.ndarray, now: Optional[float] = None) -> bool:
        """Detect markers and refit. Returns True if a new fit was made."""
        if self.locked:
            return False
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        self.last_corners, self.last_ids = corners, ids
        if ids is None:
            self.visible_ids = []
            return False
        found = {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners) if int(i) in self.markers}
        self.visible_ids = sorted(found)
        if len(found) < self.min_markers:
            return False
        if self.mode == "homography":
            H = self._fit_homography(found)
        else:
            H = self._fit_scale(found)
        if H is None:
            return False
        H = H / H[2, 2]
        if self.H is not None and self.smoothing > 0:
            H = self.smoothing * self.H + (1.0 - self.smoothing) * H   # damp corner jitter
        self.H = H
        self.H_inv = np.linalg.inv(self.H)
        self.updated_at = time.monotonic() if now is None else now
        self.good_fits += 1
        if self.lock_after and self.good_fits >= self.lock_after:
            self.locked = True
        return True

    def _fit_homography(self, found) -> Optional[np.ndarray]:
        src, dst = [], []
        for mid, px in found.items():
            src.append(self._undistort(px))
            dst.append(self.markers[mid].corners_table())
        src, dst = np.vstack(src), np.vstack(dst)
        H, _ = cv2.findHomography(src, dst, 0)
        if H is None:
            return None
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        self.reproj_error_m = float(np.mean(np.linalg.norm(proj - dst, axis=1)))
        return H

    def _fit_scale(self, found) -> Optional[np.ndarray]:
        mid = self.reference_id if self.reference_id in found else next(iter(found))
        px = self._undistort(found[mid])
        spec = self.markers[mid]
        side_px = cv2.arcLength(px.astype(np.float32), True) / 4.0
        if side_px <= 0:
            return None
        m_per_px = spec.size / side_px
        cu, cv_ = px.mean(axis=0)
        # x = (u - cu) * s + mx ; y = -(v - cv) * s + my   (notebook transform_to_world_frame)
        self.reproj_error_m = float("nan")
        return np.array([[m_per_px, 0.0, -cu * m_per_px + spec.x],
                         [0.0, -m_per_px, cv_ * m_per_px + spec.y],
                         [0.0, 0.0, 1.0]])

    def unlock(self) -> None:
        self.locked = False
        self.good_fits = 0

    # ------------------------------------------------------------------ mapping
    def to_table(self, pixels: Sequence) -> np.ndarray:
        if self.H is None:
            raise RuntimeError("Table frame not calibrated yet (no marker seen)")
        pts = self._undistort(pixels)
        return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), self.H).reshape(-1, 2)

    def to_pixel(self, table_pts: Sequence) -> np.ndarray:
        """Table metres -> undistorted pixel (good enough for drawing overlays)."""
        if self.H_inv is None:
            raise RuntimeError("Table frame not calibrated yet")
        pts = np.asarray(table_pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)

    def draw(self, image: np.ndarray) -> None:
        if self.last_ids is not None:
            cv2.aruco.drawDetectedMarkers(image, self.last_corners, self.last_ids)
        if self.H_inv is not None:
            o, x, y = self.to_pixel([[0, 0], [0.05, 0], [0, 0.05]])
            cv2.arrowedLine(image, tuple(int(v) for v in o), tuple(int(v) for v in x), (0, 0, 255), 2)
            cv2.arrowedLine(image, tuple(int(v) for v in o), tuple(int(v) for v in y), (0, 255, 0), 2)


def arm_from_table(xy: Sequence[float], cfg: Optional[dict]) -> np.ndarray:
    """Optional fixed transform table frame -> arm base frame (x, y offset + yaw)."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if not cfg:
        return xy
    yaw = math.radians(float(cfg.get("yaw_deg", 0.0)))
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return xy @ rot.T + np.array([float(cfg.get("x", 0.0)), float(cfg.get("y", 0.0))])
