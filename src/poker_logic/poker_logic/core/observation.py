"""N-player observations: immutable copies of private-to-self and public data only.

Generalised from the heads-up demo's ``bot_observation.py``. The bot never sees
other players' hole cards (they are unknown ``??`` cards in the engine anyway),
the deck order or burn cards.
"""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class PublicAction:
    player_index: int
    stage: str
    action: str          # post_sb, post_bb, fold, check, call, raise
    paid: int            # chips added by this action
    round_total: int     # player's total contribution on this street afterwards


@dataclass(frozen=True)
class LegalActions:
    can_fold: bool
    can_check: bool
    can_call: bool
    call_amount: int
    can_raise: bool
    min_raise_to: Optional[int]
    max_raise_to: Optional[int]


@dataclass(frozen=True)
class OpponentView:
    seat: int
    stack: int
    bet: int
    active: bool          # still holding cards (not folded)
    all_in: bool


@dataclass(frozen=True)
class BotObservation:
    player_index: int
    player_count: int
    hole_cards: Tuple[str, ...]
    board_cards: Tuple[str, ...]
    stage: str
    position: str         # BTN, SB, BB, UTG, MP, CO (see position_label)
    my_stack: int
    my_bet: int
    pot: int              # total chips in the middle incl. current street bets
    legal_actions: LegalActions
    opponents: Tuple[OpponentView, ...]
    public_history: Tuple[PublicAction, ...]
    small_blind: int = 5
    big_blind: int = 10
    chip_unit: int = 5

    @property
    def active_opponents(self) -> Tuple[OpponentView, ...]:
        return tuple(o for o in self.opponents if o.active)

    @property
    def max_opponent_bet(self) -> int:
        return max((o.bet for o in self.opponents), default=0)

    @property
    def max_effective_opponent_total(self) -> int:
        """Largest street total any live opponent could reach (bet + stack)."""
        return max((o.bet + o.stack for o in self.active_opponents), default=0)


STAGES = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}


def stage_name(state) -> str:
    return STAGES.get(len(tuple(state.get_board_cards(0))), "unknown")


def position_label(seat: int, player_count: int) -> str:
    """PokerKit seat order: seat 0 = SB, seat 1 = BB, last seat = button.

    Heads-up is special: PokerKit puts the button/SB at seat 1 and BB at 0
    (``post_bb`` comes from seat 0 in the demo); we mirror the demo labels.
    """
    if player_count == 2:
        return "BB" if seat == 0 else "BTN"
    if seat == 0:
        return "SB"
    if seat == 1:
        return "BB"
    if seat == player_count - 1:
        return "BTN"
    if seat == player_count - 2 and player_count >= 5:
        return "CO"
    return "UTG" if seat == 2 else "MP"


def legal_actions_of(state) -> LegalActions:
    can_call = state.can_check_or_call()
    amount = int(state.checking_or_calling_amount or 0) if can_call else 0
    can_raise = state.can_complete_bet_or_raise_to()
    return LegalActions(
        can_fold=state.can_fold(),
        can_check=can_call and amount == 0,
        can_call=can_call and amount > 0,
        call_amount=amount,
        can_raise=can_raise,
        min_raise_to=int(state.min_completion_betting_or_raising_to_amount) if can_raise else None,
        max_raise_to=int(state.max_completion_betting_or_raising_to_amount) if can_raise else None,
    )


def build_observation(state, bot_index, public_history=(), small_blind=5, big_blind=10,
                      chip_unit=5) -> BotObservation:
    n = len(state.stacks)
    if not 0 <= bot_index < n:
        raise ValueError("Invalid bot seat")
    if not state.status or state.actor_index != bot_index:
        raise ValueError("Observations can only be built on this player's betting turn")
    opponents = tuple(
        OpponentView(
            seat=i,
            stack=int(state.stacks[i]),
            bet=int(state.bets[i]),
            active=bool(state.statuses[i]),
            all_in=bool(state.statuses[i]) and int(state.stacks[i]) == 0,
        )
        for i in range(n) if i != bot_index
    )
    history = tuple(PublicAction(int(a.player_index), str(a.stage), str(a.action),
                                 int(a.paid), int(a.round_total)) for a in public_history)
    return BotObservation(
        player_index=bot_index,
        player_count=n,
        hole_cards=tuple(repr(c) for c in state.hole_cards[bot_index]),
        board_cards=tuple(repr(c) for c in state.get_board_cards(0)),
        stage=stage_name(state),
        position=position_label(bot_index, n),
        my_stack=int(state.stacks[bot_index]),
        my_bet=int(state.bets[bot_index]),
        pot=int(state.total_pot_amount),
        legal_actions=legal_actions_of(state),
        opponents=opponents,
        public_history=history,
        small_blind=small_blind,
        big_blind=big_blind,
        chip_unit=chip_unit,
    )
