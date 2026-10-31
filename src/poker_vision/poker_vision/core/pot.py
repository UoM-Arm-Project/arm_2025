"""Chip segmentation (side camera) -> total pot value.

How masks become chips is set in ``vision.yaml`` (``pot_model.count_mode``),
because it depends on what the segmentation model was trained to label:

``stack_height`` (default)
    Each mask is a stack of chips of one denomination seen from the side.
    chips = round(mask height in px / ``px_per_chip``); value = chips x class value.
``instance``
    Each mask is one chip.
``area``
    chips = mask area / ``px2_per_chip``; for models trained on loose piles.

``class_values`` maps model class names to chip values (e.g. ``chip_5: 5``);
``default_value`` is used for unknown classes (set 0 to ignore them).
``roi`` (optional pixel polygon) restricts counting to the pot so the bot's
trays and players' stacks are not counted.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .detector import Detection


@dataclass
class PotReading:
    value: float
    confidence: float
    stacks: List[Tuple[str, int, float]] = field(default_factory=list)   # (class, chips, value)


class PotEstimator:
    def __init__(self, cfg: dict):
        self.mode = cfg.get("count_mode", "stack_height")
        if self.mode not in ("stack_height", "instance", "area"):
            raise ValueError("pot_model.count_mode must be stack_height | instance | area")
        self.class_values: Dict[str, float] = {str(k): float(v) for k, v in cfg.get("class_values", {}).items()}
        self.default_value = float(cfg.get("default_value", 0.0))
        self.px_per_chip = float(cfg.get("px_per_chip", 6.0))
        self.px2_per_chip = float(cfg.get("px2_per_chip", 900.0))
        roi = cfg.get("roi")
        self.roi = np.asarray(roi, dtype=np.float32).reshape(-1, 1, 2) if roi else None
        self.round_to = float(cfg.get("round_to", 0))      # e.g. 5 -> snap total to the chip unit

    def in_roi(self, pt) -> bool:
        if self.roi is None:
            return True
        return cv2.pointPolygonTest(self.roi, (float(pt[0]), float(pt[1])), False) >= 0

    def estimate(self, dets: List[Detection]) -> PotReading:
        total, confs, stacks = 0.0, [], []
        for d in dets:
            value = self.class_values.get(d.name, self.default_value)
            if value <= 0:
                continue
            centre = d.polygon.mean(axis=0) if d.polygon is not None and len(d.polygon) else d.center
            if not self.in_roi(centre):
                continue
            if self.mode == "instance":
                chips = 1
            elif self.mode == "stack_height":
                if d.polygon is not None and len(d.polygon):
                    height = float(d.polygon[:, 1].max() - d.polygon[:, 1].min())
                else:
                    height = float(d.xyxy[3] - d.xyxy[1])
                chips = max(1, int(round(height / self.px_per_chip)))
            else:
                if d.polygon is not None and len(d.polygon) >= 3:
                    area = float(cv2.contourArea(d.polygon.astype(np.float32)))
                else:
                    area = float((d.xyxy[2] - d.xyxy[0]) * (d.xyxy[3] - d.xyxy[1]))
                chips = max(1, int(round(area / self.px2_per_chip)))
            total += chips * value
            confs.append(d.conf)
            stacks.append((d.name, chips, chips * value))
        if self.round_to > 0:
            total = round(total / self.round_to) * self.round_to
        # An empty pot seen clearly is a confident zero; the logic layer only needs a number
        confidence = float(np.mean(confs)) if confs else 1.0
        return PotReading(total, confidence, stacks)
