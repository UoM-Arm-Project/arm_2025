"""Debounce noisy per-frame vision into stable facts the game can act on."""
from collections import Counter, deque
from dataclasses import dataclass
from statistics import median
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

from .equity import normalise


@dataclass(frozen=True)
class Zone:
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


def assign_zone(zones: Iterable[Zone], x: float, y: float) -> str:
    for z in zones:
        if z.contains(x, y):
            return z.name
    return ""


class CardTracker:
    """A card is *stable* in a zone once seen there in >= ``ratio`` of the last ``window`` frames."""

    def __init__(self, window: int = 10, ratio: float = 0.7, min_confidence: float = 0.5):
        self.window = window
        self.ratio = ratio
        self.min_conf = min_confidence
        self.frames: Deque[Set[Tuple[str, str]]] = deque(maxlen=window)

    def update(self, detections: Iterable[Tuple[str, str, float]]) -> None:
        """``detections``: (card code, zone name, confidence) for one camera frame."""
        frame = set()
        for code, zone, conf in detections:
            if conf < self.min_conf or not zone:
                continue
            try:
                frame.add((zone, normalise(code)))
            except ValueError:
                continue
        self.frames.append(frame)

    def clear(self) -> None:
        self.frames.clear()

    def stable(self, zone: str) -> List[str]:
        if len(self.frames) < max(1, int(self.window * self.ratio)):
            return []
        counts = Counter(item for f in self.frames for item in f)
        need = self.ratio * len(self.frames)
        # a code is attributed to the zone where it is seen most
        best: Dict[str, Tuple[int, str]] = {}
        for (z, code), n in counts.items():
            if code not in best or n > best[code][0]:
                best[code] = (n, z)
        return sorted(code for code, (n, z) in best.items() if z == zone and n >= need)

    def stable_prefix(self, prefix: str) -> Dict[str, List[str]]:
        zones = {z for f in self.frames for z, _ in f if z.startswith(prefix)}
        return {z: self.stable(z) for z in zones}


class PotTracker:
    """Median of recent pot readings, used once a player says their turn is done."""

    def __init__(self, min_confidence: float = 0.3, history: int = 200):
        self.min_conf = min_confidence
        self.readings: Deque[Tuple[float, float]] = deque(maxlen=history)  # (t, value)

    def add(self, t: float, value: float, confidence: float = 1.0) -> None:
        if confidence >= self.min_conf and value >= 0:
            self.readings.append((t, float(value)))

    def latest(self) -> Optional[float]:
        return self.readings[-1][1] if self.readings else None

    def value_since(self, t0: float, min_samples: int = 3) -> Optional[float]:
        values = [v for t, v in self.readings if t >= t0]
        if len(values) < min_samples:
            return None
        return float(median(values))
