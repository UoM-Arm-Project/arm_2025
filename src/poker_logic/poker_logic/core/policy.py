"""Explainable heuristic policies (not GTO), generalised from the demo to 2-6 players.

What changed versus the heads-up demo's ``bot_policy.py``:

* equity is estimated against every *active* opponent at once;
* value thresholds are re-centred on the fair share ``1 / (opponents + 1)``
  so that the hand-tuned heads-up margins carry over to multiway pots;
* "late position" (BTN, CO, SB) replaces the heads-up button flag;
* bet sizes are rounded to the smallest physical chip so the arm can pay them.

The pot-odds rule is unchanged: the pot already contains every opponent's chips.
"""
import math
from dataclasses import dataclass
from typing import Optional

from .equity import EquityResult, estimate_equity, opponent_pressures
from .observation import BotObservation, position_label


@dataclass(frozen=True)
class Decision:
    action: str                     # fold, check, call, raise
    reason: str
    raise_to: Optional[int] = None  # street total, as in PokerKit / the demo
    equity: Optional[float] = None
    required_equity: Optional[float] = None
    standard_error: Optional[float] = None
    range_model: str = ""
    method: str = ""


@dataclass(frozen=True)
class BotConfig:
    samples: int = 600
    range_model: str = "public_actions"
    style: str = "balanced"

    def __post_init__(self):
        if not isinstance(self.samples, int) or isinstance(self.samples, bool) or not 50 <= self.samples <= 5000:
            raise ValueError("Sample count must be between 50 and 5000")
        if self.range_model not in ("uniform", "public_actions"):
            raise ValueError("Invalid range model")
        if self.style not in ("cautious", "balanced", "active"):
            raise ValueError("Invalid playing style")


class BaselineBot:
    def choose(self, observation):
        legal = observation.legal_actions
        if legal.can_check:
            return Decision("check", "Baseline: no chips to call; check")
        if legal.can_call:
            return Decision("call", "Baseline: call")
        if legal.can_fold:
            return Decision("fold", "No check/call action is available")
        raise ValueError("The baseline policy has no legal action")


def contestable_pot(o: BotObservation) -> int:
    """Exclude chips above what our call can match (short-stack all-in calls)."""
    matched_total = o.my_bet + o.legal_actions.call_amount
    uncalled = sum(max(0, opp.bet - matched_total) for opp in o.opponents)
    return max(0, o.pot - uncalled)


def scale_threshold(heads_up_threshold: float, opponents: int) -> float:
    """Map a heads-up equity threshold onto a pot with ``opponents`` live players.

    Heads-up the fair share is 0.5; a threshold of 0.63 means "13 points above
    fair". We keep the same *relative* margin over the multiway fair share.
    """
    fair = 1.0 / (opponents + 1)
    return fair + (heads_up_threshold - 0.5) * (1.0 - fair) / 0.5


def quantise_raise(target: int, low: int, high: int, unit: int) -> int:
    """Round to the chip unit while staying inside [low, high]; all-in stays exact."""
    if target >= high:
        return high
    unit = max(1, unit)
    rounded = int(math.ceil(target / unit) * unit)
    if rounded > high:
        return high
    return max(low, rounded)


class EquityBot:
    def __init__(self, config=None):
        self.config = config or BotConfig()

    def choose(self, observation: BotObservation) -> Decision:
        result = estimate_equity(
            observation.hole_cards, observation.board_cards, self.config.samples,
            opponent_pressures(observation, self.config.range_model) or (0,),
        )
        return self.choose_with_equity(observation, result)

    def choose_with_equity(self, o: BotObservation, result: EquityResult) -> Decision:
        legal = o.legal_actions
        call = legal.call_amount
        pot = contestable_pot(o)
        n_opp = max(1, len(o.active_opponents))
        required = call / (pot + call) if call else 0.0
        style_shift = {"cautious": 0.025, "balanced": 0.0, "active": -0.015}[self.config.style]
        realization = {"preflop": 0.90, "flop": 0.86, "turn": 0.93, "river": 1.0}[o.stage]
        # Multiway pots are harder to realise (more players left to act behind).
        realization -= 0.02 * (n_opp - 1) if o.stage != "river" else 0.0
        if call >= o.my_stack or all(opp.all_in for opp in o.active_opponents):
            realization = 1.0
        adjusted = result.equity * realization
        buffer = max(0.0, 0.01 + style_shift) + 0.5 * result.standard_error
        my_raises = sum(a.player_index == o.player_index and a.stage == o.stage and a.action == "raise"
                        for a in o.public_history)
        unopened_preflop = o.stage == "preflop" and max(o.my_bet, o.max_opponent_bet) <= o.big_blind
        late = o.position in ("BTN", "CO", "SB")
        if unopened_preflop:
            hu = 0.57 if late else 0.63
        else:
            hu = 0.68 if o.stage == "preflop" else 0.67
            if call:
                hu += 0.07
        value_threshold = scale_threshold(hu + style_shift, n_opp)
        reason_base = (
            f"{n_opp} opponent(s); equity {result.equity:.1%} (fair {1 / (n_opp + 1):.1%}); "
            f"pot odds need {required:.1%}; value threshold {value_threshold:.1%}; "
            f"range {result.range_model}; {result.method}"
        )

        def decision(action, reason, target=None):
            return Decision(action, reason, target, result.equity, required,
                            result.standard_error, result.range_model, result.method)

        if legal.can_raise and my_raises < 1 and result.equity >= value_threshold:
            low, high = legal.min_raise_to, legal.max_raise_to
            effective_cap = min(high, max(o.max_effective_opponent_total, low))
            if low <= effective_cap:
                if unopened_preflop:
                    # 2.5bb open plus one bb per limper, a common live-game sizing
                    limpers = sum(1 for opp in o.opponents
                                  if opp.active and opp.bet >= o.big_blind
                                  and position_label(opp.seat, o.player_count) != "BB")
                    target = math.ceil((2.5 + limpers) * o.big_blind)
                else:
                    fraction = 0.80 if result.equity >= scale_threshold(0.85, n_opp) else 0.55
                    target = max(o.my_bet, o.max_opponent_bet) + math.ceil(fraction * (pot + call))
                if result.equity > scale_threshold(0.82, n_opp) and o.my_stack - call <= max(o.big_blind, pot + call):
                    target = effective_cap
                target = quantise_raise(int(max(low, min(effective_cap, target))), low, effective_cap, o.chip_unit)
                return decision("raise", reason_base + "; strong-equity value bet/raise", target)

        if legal.can_check:
            return decision("check", reason_base + "; nothing to call; check")
        if legal.can_call and (adjusted >= required + buffer or not legal.can_fold):
            return decision("call", reason_base + f"; adjusted equity {adjusted:.1%} beats price; call")
        if legal.can_fold:
            return decision("fold", reason_base + f"; adjusted equity {adjusted:.1%} too low for the price; fold")
        if legal.can_call:
            return decision("call", reason_base + "; only remaining legal action")
        raise ValueError("The policy could not find a legal action")


def make_policy(mode: str, config: BotConfig):
    if mode == "baseline":
        return BaselineBot()
    if mode == "equity":
        return EquityBot(config)
    raise ValueError(f"Unknown bot mode {mode!r}")
