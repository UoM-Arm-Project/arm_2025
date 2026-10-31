"""Chip segmentation (side camera) -> total pot value.

``count_mode`` in vision.yaml (``pot_model``) says what one mask means:

``instance`` (default, matches the trained model)
    Each mask is one chip; value = sum of the class values.
``stack_height``
    Each mask is a whole stack seen from the side;
    chips = round(mask height px / ``px_per_chip``).
``area``
    chips = mask area / ``px2_per_chip``, for loose piles.

``class_values`` maps model class names to chip values (White 1, Red 5,
Blue 10); classes with no value (e.g. Black) are ignored, and ``classes`` in
the model config can stop the model producing them at all.
"""
from dataclasses import dataclass, field
from typing import Dict, List

import cv2
import numpy as np

from .detector import Detection


@dataclass
class PotReading:
    value: float
    confidence: float
    counts: Dict[str, int] = field(default_factory=dict)      # class -> chips counted


class PotEstimator:
    def __init__(self, cfg: dict):
        self.mode = cfg.get("count_mode", "instance")
        if self.mode not in ("stack_height", "instance", "area"):
            raise ValueError("pot_model.count_mode must be instance | stack_height | area")
        self.class_values: Dict[str, float] = {str(k): float(v) for k, v in cfg.get("class_values", {}).items()}
        self.default_value = float(cfg.get("default_value", 0.0))
        self.px_per_chip = float(cfg.get("px_per_chip", 6.0))
        self.px2_per_chip = float(cfg.get("px2_per_chip", 900.0))
        self.round_to = float(cfg.get("round_to", 0))

    def chips_in(self, d: Detection) -> int:
        if self.mode == "instance":
            return 1
        if self.mode == "stack_height":
            if d.polygon is not None and len(d.polygon):
                height = float(d.polygon[:, 1].max() - d.polygon[:, 1].min())
            else:
                height = float(d.xyxy[3] - d.xyxy[1])
            return max(1, int(round(height / self.px_per_chip)))
        if d.polygon is not None and len(d.polygon) >= 3:
            area = float(cv2.contourArea(d.polygon.astype(np.float32)))
        else:
            area = float((d.xyxy[2] - d.xyxy[0]) * (d.xyxy[3] - d.xyxy[1]))
        return max(1, int(round(area / self.px2_per_chip)))

    def estimate(self, dets: List[Detection]) -> PotReading:
        total, confs, counts = 0.0, [], {}
        for d in dets:
            value = self.class_values.get(d.name, self.default_value)
            if value <= 0:
                continue
            chips = self.chips_in(d)
            total += chips * value
            confs.append(d.conf)
            counts[d.name] = counts.get(d.name, 0) + chips
        if self.round_to > 0:
            total = round(total / self.round_to) * self.round_to
        confidence = float(np.mean(confs)) if confs else 1.0     # a clearly empty pot is a confident 0
        return PotReading(total, confidence, counts)
