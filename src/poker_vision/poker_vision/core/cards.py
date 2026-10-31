"""Card detections -> one entry per physical card.

The card model detects the rank/suit *corner indices* (class names like
``AC``, ``5D``, ``10H``), so a face-up card usually yields two boxes, one per
corner. Same-class boxes close together are merged into one card at their
midpoint, generalising ``process_card_centers`` from the notebook. A lone
corner (the other one hidden by a hand or chips) still counts as the card,
positioned at that corner.

Pairing distance is checked in table metres when the table frame is known
(``pair_max_distance_m``, card diagonal ~0.10 m), otherwise in pixels.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .detector import Detection

RANKS = "23456789TJQKA"
SUIT_ALIASES = {"c": "c", "d": "d", "h": "h", "s": "s", "♣": "c", "♦": "d", "♥": "h", "♠": "s",
                "clubs": "c", "diamonds": "d", "hearts": "h", "spades": "s"}
RANK_ALIASES = {"10": "T", "ace": "A", "king": "K", "queen": "Q", "jack": "J", "ten": "T"}


def class_to_code(name: str, class_map: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Model class name -> 'Ah' style code used by poker_logic; None if not a card."""
    if class_map and name in class_map:
        return class_map[name]
    n = name.strip()
    low = n.lower()
    for sep in ("_of_", " of ", "-", "_", " "):
        if sep in low:                                  # e.g. "ace_of_spades", "10-hearts"
            a, b = low.split(sep, 1)
            rank = RANK_ALIASES.get(a, a.upper())
            suit = SUIT_ALIASES.get(b)
            return rank + suit if suit and len(rank) == 1 and rank in RANKS else None
    if len(n) < 2:
        return None
    rank, suit = n[:-1].upper(), SUIT_ALIASES.get(n[-1].lower())
    rank = RANK_ALIASES.get(rank.lower(), rank)
    if suit is None or len(rank) != 1 or rank not in RANKS:
        return None
    return rank + suit


@dataclass
class Card:
    code: str
    pixel: np.ndarray            # centre in the image
    conf: float
    corners: int                 # 1 or 2 corner detections merged
    table: Optional[np.ndarray] = None   # centre in the table frame (metres)
    conflict: str = ""           # e.g. "9S/9C": the two corners disagreed, higher confidence kept


def pair_corners(dets: List[Detection], class_map=None,
                 to_table: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 max_dist_m: float = 0.12, max_dist_px: float = 500.0,
                 merge_conflicts: bool = True, min_diag_m: float = 0.02) -> List[Card]:
    """Merge corner detections of the same card. ``to_table`` maps Nx2 pixels -> Nx2 metres.

    With ``merge_conflicts``, two lone corners that share a rank *or* a suit and
    sit diagonally from each other at card-corner distance (e.g. 9S top-left +
    9C bottom-right: one corner misread) are merged into one card with the
    higher-confidence code and ``conflict`` set. The diagonal test (both |dx|
    and |dy| >= ``min_diag_m``) keeps two neighbouring cards in a row apart.
    """
    by_code: Dict[str, List[Detection]] = {}
    for d in dets:
        code = class_to_code(d.name, class_map)
        if code:
            by_code.setdefault(code, []).append(d)
    cards: List[Card] = []
    for code, group in by_code.items():
        px = np.array([d.center for d in group])
        tb = to_table(px) if to_table is not None else None
        pts, limit = (tb, max_dist_m) if tb is not None else (px, max_dist_px)
        used = set()
        # greedy: closest same-code pairs first
        pairs = sorted(((float(np.linalg.norm(pts[i] - pts[j])), i, j)
                        for i in range(len(group)) for j in range(i + 1, len(group))))
        for dist, i, j in pairs:
            if dist > limit or i in used or j in used:
                continue
            used.update((i, j))
            cards.append(Card(code, (px[i] + px[j]) / 2, max(group[i].conf, group[j].conf), 2,
                              None if tb is None else (tb[i] + tb[j]) / 2))
        for i in range(len(group)):
            if i not in used:
                cards.append(Card(code, px[i], group[i].conf, 1, None if tb is None else tb[i]))
    if merge_conflicts:
        cards = _merge_conflicting(cards, max_dist_m if to_table is not None else max_dist_px,
                                   min_diag_m if to_table is not None else 0.0)
    # A real deck has one of each card: if a code appears twice (misdetection), keep the best
    best: Dict[str, Card] = {}
    for c in cards:
        if c.code not in best or (c.corners, c.conf) > (best[c.code].corners, best[c.code].conf):
            best[c.code] = c
    return sorted(best.values(), key=lambda c: c.code)


def _merge_conflicting(cards: List[Card], limit: float, min_diag: float) -> List[Card]:
    lone = [c for c in cards if c.corners == 1]
    done = [c for c in cards if c.corners != 1]
    cand = []
    for i in range(len(lone)):
        for j in range(i + 1, len(lone)):
            a, b = lone[i], lone[j]
            if a.code == b.code or (a.code[0] != b.code[0] and a.code[1] != b.code[1]):
                continue                                  # need same rank or same suit
            pa = a.table if a.table is not None else a.pixel
            pb = b.table if b.table is not None else b.pixel
            dx, dy = abs(pa[0] - pb[0]), abs(pa[1] - pb[1])
            dist = float(np.hypot(dx, dy))
            if dist > limit or dx < min_diag or dy < min_diag:
                continue
            cand.append((dist, i, j))
    used = set()
    for _dist, i, j in sorted(cand):
        if i in used or j in used:
            continue
        used.update((i, j))
        a, b = lone[i], lone[j]
        keep, other = (a, b) if a.conf >= b.conf else (b, a)
        done.append(Card(keep.code, (a.pixel + b.pixel) / 2, keep.conf, 2,
                         None if a.table is None else (a.table + b.table) / 2,
                         conflict=f"{keep.code}/{other.code}"))
    done += [c for k, c in enumerate(lone) if k not in used]
    return done
