"""Compile poker skills into the primitives the existing control stack accepts.

The control layer exposes ``/move_pose`` (x, y, z, pitch, roll, duration in the
arm base frame, IK solved by ``poker_controller``) and ``/pump_control``
(Int32 1/0). Every skill here becomes an ordered list of those primitives, so
nothing in ``poker_control`` has to change.

All coordinates come from ``config/arm_layout.yaml`` (arm base frame, metres,
radians). The face-up "flip" release reuses the motion taught in
``poker_demo/pick_demo_node.py``.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

from .chips import make_change


@dataclass(frozen=True)
class MovePose:
    x: float
    y: float
    z: float
    pitch: float = 0.0
    roll: float = 0.0
    duration: float = 3.0
    label: str = ""


@dataclass(frozen=True)
class Pump:
    on: bool
    label: str = ""


@dataclass(frozen=True)
class Wait:
    seconds: float
    label: str = ""


Step = Union[MovePose, Pump, Wait]


@dataclass
class ArmState:
    """Things the executor must remember between skills."""
    tray_counts: Dict[int, int] = field(default_factory=dict)
    deck_count: int = 52


class SkillCompiler:
    def __init__(self, layout: dict):
        self.L = layout
        m = layout.get("motion", {})
        self.travel_z = float(m.get("travel_z", 0.15))
        self.d_travel = float(m.get("travel_duration", 3.0))
        self.d_fine = float(m.get("fine_duration", 2.0))
        flip = layout.get("cards", {}).get("face_up_release", {})
        self.flip_approach = flip.get("approach", {"dx": 0.0, "dy": 0.0, "z": 0.20, "pitch": -0.4, "roll": 3.13})
        self.flip_release = flip.get("release", {"dx": 0.0, "dy": -0.065, "z": 0.20, "pitch": -1.2, "roll": 3.13})

    # ------------------------------------------------------------------ primitives
    def _xy(self, p) -> Tuple[float, float]:
        return float(p["x"]), float(p["y"])

    def _hover(self, x, y, label):
        return MovePose(x, y, self.travel_z, 0.0, 0.0, self.d_travel, label)

    def pick(self, x, y, z, label) -> List[Step]:
        return [self._hover(x, y, f"{label}: above"),
                MovePose(x, y, z, 0.0, 0.0, self.d_fine, f"{label}: down"),
                Pump(True, f"{label}: suction on"),
                MovePose(x, y, self.travel_z, 0.0, 0.0, self.d_fine, f"{label}: lift")]

    def place(self, x, y, z, label) -> List[Step]:
        return [self._hover(x, y, f"{label}: above"),
                MovePose(x, y, z, 0.0, 0.0, self.d_fine, f"{label}: down"),
                Pump(False, f"{label}: release"),
                MovePose(x, y, self.travel_z, 0.0, 0.0, self.d_fine, f"{label}: retreat")]

    def place_flipped(self, x, y, label) -> List[Step]:
        """Release a card so it lands the other way up (pick_demo flip motion)."""
        a, r = self.flip_approach, self.flip_release
        return [self._hover(x, y, f"{label}: above"),
                MovePose(x + a.get("dx", 0.0), y + a.get("dy", 0.0), a["z"], a["pitch"], a["roll"],
                         self.d_travel, f"{label}: flip"),
                MovePose(x + r.get("dx", 0.0), y + r.get("dy", 0.0), r["z"], r["pitch"], r["roll"],
                         self.d_travel, f"{label}: tilt"),
                Pump(False, f"{label}: release"),
                MovePose(x + r.get("dx", 0.0), y + r.get("dy", 0.0), self.travel_z, 0.0, 0.0,
                         self.d_travel, f"{label}: restore")]

    def home(self) -> List[Step]:
        h = self.L["poses"]["home"]
        return [MovePose(h["x"], h["y"], h["z"], h.get("pitch", 0.0), h.get("roll", 0.0),
                         float(h.get("duration", self.d_travel)), "home")]

    # ------------------------------------------------------------------ chips
    def pay_chips(self, amount: int, state: ArmState) -> Tuple[List[Step], ArmState, Dict[int, int]]:
        chips = self.L["chips"]
        trays = {int(k): v for k, v in chips["trays"].items()}
        inventory = {d: state.tray_counts.get(d, int(trays[d].get("count", 0))) for d in trays}
        change = make_change(int(amount), list(trays), inventory)
        drop = chips["pot_drop"]
        spread = float(chips.get("pot_spread", 0.0))
        offsets = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        steps: List[Step] = []
        counts = dict(inventory)
        n = 0
        for denom, k in change.items():
            tray = trays[denom]
            tx, ty = self._xy(tray)
            for _ in range(k):
                z = float(tray["base_z"]) + max(0, counts[denom] - 1) * float(tray.get("thickness", 0.0033))
                counts[denom] -= 1
                ox, oy = offsets[n % len(offsets)]
                n += 1
                steps += self.pick(tx, ty, z, f"chip {denom}")
                steps += self.place(drop["x"] + ox * spread, drop["y"] + oy * spread,
                                    float(drop["z"]), f"chip {denom} to pot")
        steps += self.home()
        return steps, ArmState(counts, state.deck_count), change

    # ------------------------------------------------------------------ bot actions
    def bot_action(self, kind: str, chips: int, state: ArmState):
        """kind: fold | check | call | raise | post_blind -> (steps, new_state, info)."""
        if kind in ("call", "raise", "bet", "post_blind"):
            return self.pay_chips(chips, state)
        if kind == "check":
            tap = self.L["poses"]["check_tap"]
            x, y = self._xy(tap)
            return ([self._hover(x, y, "check: above"),
                     MovePose(x, y, float(tap["z"]), 0.0, 0.0, self.d_fine, "check: tap"),
                     MovePose(x, y, self.travel_z, 0.0, 0.0, self.d_fine, "check: up")] + self.home(),
                    state, {})
        if kind == "fold":
            cards = self.L["cards"]
            mx, my = self._xy(cards["muck"])
            steps: List[Step] = []
            for i, slot in enumerate(cards["bot_hole_slots"]):
                sx, sy = self._xy(slot)
                steps += self.pick(sx, sy, float(cards["pick_z"]), f"fold card {i}")
                steps += self.place_flipped(mx, my, f"fold card {i} to muck")
            return steps + self.home(), state, {}
        raise ValueError(f"Unknown bot action {kind!r}")

    # ------------------------------------------------------------------ dealing
    def _deck_pick(self, state: ArmState, label: str) -> List[Step]:
        deck = self.L["cards"]["deck"]
        x, y = self._xy(deck)
        z = float(deck["z_bottom"]) + max(0, state.deck_count - 1) * float(deck.get("thickness", 0.0003))
        return self.pick(x, y, z, label)

    def dealer_action(self, kind: str, state: ArmState, seat: int = 0, index: int = 0,
                      face_up: bool = False):
        """kind: deal_hole | burn | deal_board | reset_deck | home -> (steps, new_state, info)."""
        cards = self.L["cards"]
        if kind == "reset_deck":
            return [], ArmState(dict(state.tray_counts), 52), {}
        if kind == "home":
            return self.home(), state, {}
        if state.deck_count <= 0:
            raise ValueError("Deck holder is empty")
        steps = self._deck_pick(state, f"{kind}: deck")
        new_state = ArmState(dict(state.tray_counts), state.deck_count - 1)
        if kind == "deal_hole":
            if face_up:
                slot = cards["bot_hole_slots"][index]
                steps += self.place_flipped(*self._xy(slot), f"bot card {index}")
            else:
                target = cards["seat_targets"][int(seat)]
                dx = float(cards.get("second_card_offset", 0.03)) * index
                steps += self.place(target["x"] + dx, target["y"], float(cards["drop_z"]), f"seat {seat} card {index}")
        elif kind == "burn":
            b = cards["burn"]
            steps += self.place(b["x"], b["y"], float(cards["drop_z"]), "burn")
        elif kind == "deal_board":
            slot = cards["board_slots"][index]
            steps += self.place_flipped(*self._xy(slot), f"board slot {index}")
        else:
            raise ValueError(f"Unknown dealer action {kind!r}")
        return steps + self.home(), new_state, {}


def validate_layout(layout: dict, seats: int) -> List[str]:
    """Return a list of problems (empty = OK)."""
    problems = []
    for key in ("motion", "poses", "chips", "cards"):
        if key not in layout:
            problems.append(f"missing section '{key}'")
    if problems:
        return problems
    for p in ("home", "check_tap"):
        if p not in layout["poses"]:
            problems.append(f"poses.{p} missing")
    cards = layout["cards"]
    for k in ("deck", "burn", "muck", "board_slots", "bot_hole_slots", "seat_targets", "pick_z", "drop_z"):
        if k not in cards:
            problems.append(f"cards.{k} missing")
    if len(cards.get("board_slots", [])) < 5:
        problems.append("cards.board_slots needs 5 entries")
    if len(cards.get("bot_hole_slots", [])) < 2:
        problems.append("cards.bot_hole_slots needs 2 entries")
    missing = [s for s in range(seats) if s not in {int(k) for k in cards.get("seat_targets", {})}]
    if missing:
        problems.append(f"cards.seat_targets missing chairs {missing}")
    if not layout["chips"].get("trays"):
        problems.append("chips.trays is empty")
    return problems
