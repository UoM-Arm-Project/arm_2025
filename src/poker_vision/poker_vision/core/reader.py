"""One-shot reads: a still frame in, a validated answer out (no ROS).

``CardReader.read(frames, zone, expected)``
    refresh the ArUco table frame -> crop the zone (+ margin) -> card model on
    the crop -> shift boxes back to full-image pixels -> pair corners -> table
    metres -> keep cards inside the zone -> check the count. With several
    frames (``shots``) each card must appear in the majority of them.

``PotReader.read(frame)``
    crop the pot ROI -> chip segmentation (only the classes asked for) ->
    count chips -> value.
"""
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .cards import Card, pair_corners
from .detector import Detection
from .pot import PotEstimator, PotReading
from .table_frame import TableFrame
from .zones import Zone, crop_box


@dataclass
class CardRead:
    success: bool
    message: str
    cards: List[Card] = field(default_factory=list)
    crop: Optional[Tuple[int, int, int, int]] = None
    detections: List[Detection] = field(default_factory=list)   # full-image pixels (last shot)
    inference_ms: float = 0.0


@dataclass
class PotRead:
    success: bool
    message: str
    reading: Optional[PotReading] = None
    crop: Optional[Tuple[int, int, int, int]] = None
    detections: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0


def _shift(dets: List[Detection], ox: int, oy: int) -> List[Detection]:
    off = np.array([ox, oy, ox, oy], dtype=np.float64)
    return [Detection(d.name, d.conf, d.xyxy + off, None if d.polygon is None else d.polygon + np.array([ox, oy]))
            for d in dets]


class CardReader:
    def __init__(self, table: TableFrame, model, zones: Dict[str, Zone], cfg: dict):
        self.table = table
        self.model = model
        self.zones = zones
        self.class_map = cfg.get("class_map") or None
        self.pair_m = float(cfg.get("pair_max_distance_m", 0.12))
        self.pair_px = float(cfg.get("pair_max_distance_px", 500))
        self.merge_conflicts = bool(cfg.get("merge_conflicting_corners", True))
        self.reject_conflicts = cfg.get("conflict_policy", "best") == "reject"
        self.crop = bool(cfg.get("crop_to_zone", True))
        self.margin = float(cfg.get("crop_margin_m", 0.03))
        self.zone_tol = float(cfg.get("zone_tolerance_m", 0.02))
        self.min_conf = float(cfg.get("min_card_conf", 0.0))

    def read(self, frames: Sequence[np.ndarray], zone_name: str, expected: int = 0) -> CardRead:
        t0 = time.perf_counter()
        zone = self.zones.get(zone_name)
        if zone is None:
            return CardRead(False, f"unknown zone {zone_name!r} (define it under zones: in vision.yaml)")
        for f in frames:                       # markers may be partly covered: try every shot
            self.table.update(f)
            if self.table.ready:
                break
        if not self.table.ready:
            return CardRead(False, "no ArUco table marker visible")
        shots: List[List[Card]] = []
        box, dets = None, []
        for frame in frames:
            img, (ox, oy) = frame, (0, 0)
            if self.crop:
                box = crop_box(zone, self.table.to_pixel, frame.shape, self.margin)
                if box is None:
                    return CardRead(False, f"zone {zone_name} is outside the camera image")
                x0, y0, x1, y1 = box
                img, (ox, oy) = frame[y0:y1, x0:x1], (x0, y0)
            dets = _shift(self.model(img, offset=(ox, oy)), ox, oy)
            cards = pair_corners(dets, self.class_map, self.table.to_table, self.pair_m, self.pair_px,
                                 self.merge_conflicts)
            shots.append([c for c in cards if c.conf >= self.min_conf and c.table is not None
                          and zone.contains(c.table[0], c.table[1], self.zone_tol)])
        cards = self._vote(shots)
        ms = (time.perf_counter() - t0) * 1000.0
        problems = []
        if expected and len(cards) != expected:
            problems.append(f"expected {expected} card(s), saw {len(cards)}")
        conflicts = [c.conflict for c in cards if c.conflict]
        if conflicts:
            note = f"corner disagreement {conflicts} (kept higher confidence)"
            if self.reject_conflicts:
                problems.append(note)
            else:
                return CardRead(not problems, "; ".join(problems + [note]), cards, box, dets, ms)
        return CardRead(not problems, "; ".join(problems) or "ok", cards, box, dets, ms)

    @staticmethod
    def _vote(shots: List[List[Card]]) -> List[Card]:
        if len(shots) == 1:
            return shots[0]
        votes = Counter(c.code for shot in shots for c in shot)
        need = len(shots) // 2 + 1
        best: Dict[str, Card] = {}
        for shot in shots:
            for c in shot:
                if votes[c.code] >= need and (c.code not in best or c.conf > best[c.code].conf):
                    best[c.code] = c
        return sorted(best.values(), key=lambda c: c.code)


class PotReader:
    def __init__(self, model, cfg: dict):
        self.model = model
        self.estimator = PotEstimator(cfg)
        roi = cfg.get("roi") or []
        self.roi = tuple(int(v) for v in roi) if len(roi) == 4 else None    # x0, y0, x1, y1 pixels

    def read(self, frame: np.ndarray) -> PotRead:
        t0 = time.perf_counter()
        img, (ox, oy) = frame, (0, 0)
        box = None
        if self.roi is not None:
            h, w = frame.shape[:2]
            x0, y0, x1, y1 = self.roi
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if x1 - x0 < 32 or y1 - y0 < 32:
                return PotRead(False, "pot roi is outside the side-camera image")
            box = (x0, y0, x1, y1)
            img, (ox, oy) = frame[y0:y1, x0:x1], (x0, y0)
        dets = _shift(self.model(img, offset=(ox, oy)), ox, oy)
        reading = self.estimator.estimate(dets)
        ms = (time.perf_counter() - t0) * 1000.0
        return PotRead(True, "ok", reading, box, dets, ms)


# ---------------------------------------------------------------------- drawing
def draw_cards(frame: np.ndarray, read: CardRead, table: Optional[TableFrame], zones: Dict[str, Zone]) -> np.ndarray:
    img = frame.copy()
    if table is not None:
        table.draw(img)
        if table.ready:
            for z in zones.values():
                pts = table.to_pixel(z.corners()).astype(np.int32)
                cv2.polylines(img, [pts], True, (255, 160, 0), 2)
                cv2.putText(img, z.name, tuple(int(v) for v in pts[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 160, 0), 2)
    if read.crop:
        x0, y0, x1, y1 = read.crop
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 255), 2)
    for d in read.detections:
        x0, y0, x1, y1 = (int(v) for v in d.xyxy)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 0), 1)
    for c in read.cards:
        u, v = int(c.pixel[0]), int(c.pixel[1])
        colour = (0, 165, 255) if c.conflict else (0, 255, 255)
        cv2.circle(img, (u, v), 6, colour, -1)
        cv2.putText(img, c.code + ("?" if c.conflict else ""), (u + 8, v - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    cv2.putText(img, read.message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if read.success else (0, 0, 255), 2)
    return img


def draw_pot(frame: np.ndarray, read: PotRead) -> np.ndarray:
    img = frame.copy()
    if read.crop:
        x0, y0, x1, y1 = read.crop
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
    for d in read.detections:
        if d.polygon is not None and len(d.polygon) >= 3:
            cv2.polylines(img, [d.polygon.astype(np.int32)], True, (0, 255, 0), 2)
        else:
            x0, y0, x1, y1 = (int(v) for v in d.xyxy)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
    if read.reading is not None:
        counts = " ".join(f"{k}:{v}" for k, v in sorted(read.reading.counts.items()))
        cv2.putText(img, f"POT {read.reading.value:.0f}  {counts}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 3)
    return img
