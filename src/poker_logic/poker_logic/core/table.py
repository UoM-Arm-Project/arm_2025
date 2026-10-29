"""A PokerKit hand driven by the *physical* table instead of a random deck.

Replaces the demo's ``PokerMatch`` (which dealt random cards to a heads-up
match) with a 2-6 seat table where:

* cards come from outside (vision / dealer arm). Cards nobody can see are
  dealt into PokerKit as unknown ``??`` cards, so the engine's rules,
  pot and side-pot logic still run, but nothing hidden is ever invented;
* showdown cards are revealed into the engine when players turn them over;
* physical chairs (0..N-1, fixed around the table) are mapped to PokerKit's
  per-hand seat order (seat 0 = small blind ... last seat = button);
* busted players sit out and the button rotates clockwise (increasing chair).
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

from pokerkit import Automation, NoLimitTexasHoldem

from .equity import DECK, normalise
from .observation import (BotObservation, LegalActions, PublicAction, build_observation,
                          legal_actions_of, position_label, stage_name)

AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)


@dataclass(frozen=True)
class Phase:
    kind: str                    # deal_hole, burn, board, bet, showdown, over
    seat: Optional[int] = None   # physical chair for deal_hole / bet / showdown
    count: int = 0               # board cards to deal
    hole_index: int = 0          # 0 or 1 for deal_hole (first / second round)


@dataclass
class TableConfig:
    seats: int = 5                       # physical chairs incl. the bot
    bot_seat: Optional[int] = 0          # None -> dealer-only, the bot does not play
    small_blind: int = 5
    big_blind: int = 10
    starting_stack: int = 1000
    chip_unit: int = 5                   # smallest physical chip value
    first_button: int = 0

    def __post_init__(self):
        if not 2 <= self.seats <= 6:
            raise ValueError("Supported table size is 2-6 seats")
        if self.bot_seat is not None and not 0 <= self.bot_seat < self.seats:
            raise ValueError("bot_seat outside the table")
        if not 0 < self.small_blind <= self.big_blind:
            raise ValueError("Invalid blinds")
        if self.small_blind % self.chip_unit or self.big_blind % self.chip_unit:
            raise ValueError("Blinds must be multiples of the chip unit")


class PhysicalTable:
    def __init__(self, config: TableConfig, stacks: Optional[Dict[int, int]] = None):
        self.cfg = config
        self.stacks: Dict[int, int] = dict(stacks) if stacks else {
            s: config.starting_stack for s in range(config.seats)}
        self.total_chips = sum(self.stacks.values())
        self.button = config.first_button
        self.hand_number = 0
        self.state = None
        self.order: List[int] = []           # engine seat -> physical chair
        self.public_actions: List[PublicAction] = []
        self.log: List[str] = []
        self.results: List[dict] = []
        self.revealed: Dict[int, Sequence[str]] = {}
        self.refunded: Dict[int, int] = {}          # uncalled chips handed back this hand
        self.refund_events: List[tuple] = []        # (hand, chair, amount), append-only

    # ------------------------------------------------------------------ mapping
    def engine_seat(self, chair: int) -> Optional[int]:
        return self.order.index(chair) if chair in self.order else None

    def chair(self, engine_seat: Optional[int]) -> Optional[int]:
        return None if engine_seat is None else self.order[engine_seat]

    @property
    def bot_engine_seat(self) -> Optional[int]:
        return None if self.cfg.bot_seat is None else self.engine_seat(self.cfg.bot_seat)

    @property
    def bot_in_hand(self) -> bool:
        return self.bot_engine_seat is not None

    def seats_with_chips(self) -> List[int]:
        return [s for s in range(self.cfg.seats) if self.stacks[s] > 0]

    @property
    def match_over(self) -> bool:
        return len(self.seats_with_chips()) < 2

    # ------------------------------------------------------------------ hand lifecycle
    def start_hand(self) -> None:
        if self.state is not None and self.state.status:
            raise ValueError("The current hand is still in progress")
        live = self.seats_with_chips()
        if len(live) < 2:
            raise ValueError("Fewer than two players have chips")
        if self.hand_number > 0:
            self.button = self._next_live(self.button, live)
        elif self.button not in live:
            self.button = self._next_live(self.button, live)
        # engine seat 0 = first live chair clockwise from the button; button last
        start = live.index(self.button)
        self.order = live[start + 1:] + live[:start + 1]
        self.hand_number += 1
        starting = tuple(self.stacks[c] for c in self.order)
        self.state = NoLimitTexasHoldem.create_state(
            AUTOMATIONS, True, 0, (self.cfg.small_blind, self.cfg.big_blind),
            self.cfg.big_blind, starting, len(self.order), divmod=self._chip_divmod)
        self.revealed = {}
        self.refunded = {}
        self.public_actions = []
        for e, chair in enumerate(self.order):
            paid = starting[e] - int(self.state.stacks[e])
            if paid:
                label = "post_bb" if position_label(e, len(self.order)) == "BB" else "post_sb"
                self.public_actions.append(PublicAction(e, "preflop", label, paid, paid))
        self.hand_start = dict(self.stacks)
        self.log = [f"Hand {self.hand_number}: button chair {self.button}; "
                    f"blinds {self.cfg.small_blind}/{self.cfg.big_blind}; seats {self.order}"]

    def _chip_divmod(self, dividend: int, divisor: int):
        """Split pots in whole physical chips; leftover chips follow PokerKit's odd-chip rule."""
        unit = self.cfg.chip_unit
        quotient = (dividend // (divisor * unit)) * unit
        return quotient, dividend - quotient * divisor

    @staticmethod
    def _next_live(chair, live):
        later = [s for s in live if s > chair]
        return later[0] if later else live[0]

    # ------------------------------------------------------------------ phase
    def phase(self) -> Phase:
        s = self.state
        if s is None:
            return Phase("over")
        if s.can_deal_hole():
            dealee = s.hole_dealee_index
            return Phase("deal_hole", self.chair(dealee), hole_index=len(s.hole_cards[dealee]))
        if s.can_burn_card():
            return Phase("burn")
        if s.can_deal_board():
            return Phase("board", count=int(s.board_dealing_count))
        if s.actor_index is not None:
            return Phase("bet", self.chair(s.actor_index))
        if s.showdown_index is not None:
            return Phase("showdown", self.chair(s.showdown_index))
        if not s.status:
            return Phase("over")
        raise RuntimeError("Unhandled engine state")

    @property
    def stage(self) -> str:
        return stage_name(self.state) if self.state is not None else "none"

    @property
    def board(self) -> List[str]:
        return [repr(c) for c in self.state.get_board_cards(0)] if self.state else []

    @property
    def pot(self) -> int:
        """Chips physically in the middle (collected pot + current street bets)."""
        return int(self.state.total_pot_amount) if self.state and self.state.status else 0

    def bet_of(self, chair: int) -> int:
        e = self.engine_seat(chair)
        return int(self.state.bets[e]) if e is not None else 0

    def stack_of(self, chair: int) -> int:
        e = self.engine_seat(chair)
        if e is None or not self.state.status:
            return self.stacks[chair]
        return int(self.state.stacks[e])

    def in_hand(self, chair: int) -> bool:
        e = self.engine_seat(chair)
        return e is not None and bool(self.state.statuses[e])

    # ------------------------------------------------------------------ dealing
    def deal_hole(self, card: Optional[str] = None) -> int:
        """Deal the next hole card (``None`` = face-down unknown). Returns the chair."""
        ph = self.phase()
        if ph.kind != "deal_hole":
            raise ValueError("Not dealing hole cards now")
        self.state.deal_hole(normalise(card) if card else "??")
        return ph.seat

    def deal_all_holes(self, bot_cards: Sequence[str] = ()) -> None:
        """Player mode: humans dealt; we only know the bot's own two cards."""
        bot_cards = list(bot_cards)
        while self.phase().kind == "deal_hole":
            ph = self.phase()
            known = ph.seat == self.cfg.bot_seat and len(bot_cards) > ph.hole_index
            self.deal_hole(bot_cards[ph.hole_index] if known else None)

    def burn(self) -> None:
        if self.phase().kind != "burn":
            raise ValueError("Not burning now")
        self.state.burn_card("??")

    def deal_board(self, cards: Sequence[str]) -> None:
        ph = self.phase()
        if ph.kind != "board":
            raise ValueError("Not dealing the board now")
        cards = [normalise(c) for c in cards]
        if len(cards) != ph.count:
            raise ValueError(f"Expected {ph.count} board card(s), got {len(cards)}")
        known = set(self.board) | set(self.bot_hole_cards)
        dup = [c for c in cards if c in known]
        if dup or len(set(cards)) != len(cards):
            raise ValueError(f"Duplicate card(s) {dup or cards}")
        self.state.deal_board("".join(cards))
        self.log.append(f"Board: {' '.join(self.board)}")
        self._maybe_finish()   # an all-in run-out settles on the last board card

    @property
    def bot_hole_cards(self) -> List[str]:
        e = self.bot_engine_seat
        if e is None or self.state is None:
            return []
        return [repr(c) for c in self.state.hole_cards[e] if repr(c) != "??"]

    # ------------------------------------------------------------------ betting
    def legal(self, chair: int) -> LegalActions:
        if self.phase() != Phase("bet", chair):
            raise ValueError(f"It is not chair {chair}'s turn")
        return legal_actions_of(self.state)

    def act(self, chair: int, action: str, raise_to: Optional[int] = None) -> PublicAction:
        s = self.state
        e = self.engine_seat(chair)
        if not s.status or s.actor_index != e:
            raise ValueError(f"It is not chair {chair}'s turn")
        before = int(s.bets[e])
        stacks_before = [int(x) for x in s.stacks]
        stage = stage_name(s)
        if action == "fold":
            if not s.can_fold():
                raise ValueError("Folding is not currently allowed")
            s.fold()
            paid, total, label = 0, before, "Fold"
        elif action in ("check", "call"):
            cost = s.checking_or_calling_amount
            if not s.can_check_or_call() or (action == "check" and cost != 0):
                raise ValueError("This check/call action is not currently allowed")
            op = s.check_or_call()
            paid = int(op.amount)
            total = before + paid
            action = "check" if paid == 0 else "call"
            label = "Check" if paid == 0 else f"Call {paid}"
        elif action == "raise":
            if not isinstance(raise_to, int) or isinstance(raise_to, bool):
                raise ValueError("The bet amount must be an integer")
            if not s.can_complete_bet_or_raise_to(raise_to):
                raise ValueError(f"Raise to {raise_to} is outside the legal range")
            op = s.complete_bet_or_raise_to(raise_to)
            total = int(op.amount)
            paid = total - before
            label = f"Bet/raise to {total} (+{paid})"
        else:
            raise ValueError(f"Unknown action {action!r}")
        pa = PublicAction(e, stage, action, paid, total)
        self.public_actions.append(pa)
        who = "Bot" if chair == self.cfg.bot_seat else f"Chair {chair}"
        self.log.append(f"{who}: {label}")
        self.check_chips()
        if s.status:
            # PokerKit hands uncalled chips back when a betting round closes
            for i, chair_i in enumerate(self.order):
                gained = int(s.stacks[i]) - stacks_before[i] + (paid if i == e else 0)
                if gained > 0:
                    self.refunded[chair_i] = self.refunded.get(chair_i, 0) + gained
                    self.refund_events.append((self.hand_number, chair_i, gained))
                    self.log.append(f"Uncalled {gained} returned to chair {chair_i}")
        self._maybe_finish()
        return pa

    def observation(self) -> BotObservation:
        return build_observation(self.state, self.bot_engine_seat, self.public_actions,
                                 self.cfg.small_blind, self.cfg.big_blind, self.cfg.chip_unit)

    # ------------------------------------------------------------------ showdown
    def reveal(self, chair: int, cards: Optional[Sequence[str]]) -> None:
        """Show ``cards`` for ``chair`` at showdown, or muck with ``None``."""
        ph = self.phase()
        if ph != Phase("showdown", chair):
            raise ValueError(f"Chair {chair} is not due to show")
        if chair == self.cfg.bot_seat and cards is None:
            cards = self.bot_hole_cards
        if cards is None and len(self.board) < 5 and chair != self.cfg.bot_seat:
            raise ValueError("All-in showdown before the run-out: hands must be tabled, not mucked")
        if cards is None and sum(bool(x) for x in self.state.statuses) == 1:
            # Everyone else mucked: the last player wins without showing. PokerKit
            # still needs concrete cards, so use two unused placeholders (they
            # cannot change the result with no opponent left).
            e = self.engine_seat(chair)
            current = [repr(c) for c in self.state.hole_cards[e]]
            if "??" in current:
                used = set(self.board) | set(self.bot_hole_cards) | {
                    c for v in self.revealed.values() for c in v}
                spare = [c for c in DECK if c not in used][:2]
                self.state.show_or_muck_hole_cards("".join(spare))
            else:
                self.state.show_or_muck_hole_cards(True)
            self.log.append(f"Chair {chair}: wins uncontested (cards not shown)")
        elif cards is None:
            self.state.show_or_muck_hole_cards(False)
            self.log.append(f"Chair {chair}: mucks")
        else:
            cards = [normalise(c) for c in cards]
            e = self.engine_seat(chair)
            current = [repr(c) for c in self.state.hole_cards[e]]
            if "??" in current:
                self.state.show_or_muck_hole_cards("".join(cards))
            else:
                self.state.show_or_muck_hole_cards(True)
            self.revealed[chair] = tuple(cards)
            self.log.append(f"Chair {chair}: shows {' '.join(cards)}")
        self._maybe_finish()

    # ------------------------------------------------------------------ settlement
    def _maybe_finish(self) -> None:
        if self.state.status or (self.results and self.results[-1]["hand"] == self.hand_number):
            return
        for e, chair in enumerate(self.order):
            self.stacks[chair] = int(self.state.stacks[e])
        net = {c: self.stacks[c] - self.hand_start[c] for c in self.order}
        self.results.append({"hand": self.hand_number, "button": self.button, "net": net,
                             "board": self.board, "revealed": dict(self.revealed)})
        winners = [c for c, v in net.items() if v > 0]
        self.log.append(f"Hand over: winners {winners}; stacks {self.stacks}")
        if sum(self.stacks.values()) != self.total_chips:
            raise RuntimeError("Chip conservation check failed after settlement")

    def payouts(self) -> Dict[int, int]:
        """Chips each chair receives from the pot this hand (for pushing the pot)."""
        if self.state is None or self.state.status:
            return {}
        out = {}
        for c in self.order:
            physical = self.hand_start[c] - self._contributed(c) + self.refunded.get(c, 0)
            if self.stacks[c] - physical > 0:
                out[c] = self.stacks[c] - physical
        return out

    def _contributed(self, chair: int) -> int:
        return self.public_contribution(chair)

    def public_contribution(self, chair: int) -> int:
        """Chips ``chair`` has put in the middle this hand (blinds included)."""
        e = self.engine_seat(chair)
        return sum(a.paid for a in self.public_actions if a.player_index == e)

    def check_chips(self) -> None:
        live = sum(int(x) for x in self.state.stacks) + int(self.state.total_pot_amount)
        out = sum(self.stacks[c] for c in range(self.cfg.seats) if c not in self.order)
        if self.state.status and live + out != self.total_chips:
            raise RuntimeError("Chip conservation check failed")

    def snapshot(self) -> dict:
        ph = self.phase()
        return {
            "hand": self.hand_number, "stage": self.stage, "button": self.button,
            "phase": asdict(ph), "board": self.board, "pot": self.pot,
            "stacks": {c: self.stack_of(c) for c in range(self.cfg.seats)},
            "bets": {c: self.bet_of(c) for c in range(self.cfg.seats)},
            "in_hand": {c: self.in_hand(c) for c in range(self.cfg.seats)},
            "bot_seat": self.cfg.bot_seat,
        }
