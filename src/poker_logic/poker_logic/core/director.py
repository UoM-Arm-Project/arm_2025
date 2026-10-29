"""GameDirector: the ROS-free state machine between perception, the engine and the arm.

Inputs (all non-blocking):
    * stable cards from :class:`CardTracker` (zones ``board``, ``bot_hole``, ``seat_<n>``)
    * pot readings from :class:`PotTracker`
    * ``turn_done`` / ``fold`` / ``next_hand`` signals (buttons, console, dashboard)
    * ``arm_finished`` results for commands it issued

Outputs (``self.outbox``):
    * :class:`ArmCommand` - one physical skill for the arm (bot betting or dealing)
    * :class:`Announce`   - text for the operator / table display

Modes:
    ``player``         the bot holds a seat, humans deal
    ``dealer``         the arm deals, the bot does not play
    ``dealer_player``  the arm deals and the bot plays a seat
"""
import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from .inference import Inferred, infer_action
from .policy import Decision
from .table import Phase, PhysicalTable
from .tracker import CardTracker, PotTracker

MODES = ("player", "dealer", "dealer_player")


@dataclass
class ArmCommand:
    id: int
    skill: str                    # bet, post_blind, check, fold, deal_card, burn, deal_board
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Announce:
    text: str
    level: str = "info"           # info, warn, error


@dataclass
class DirectorConfig:
    mode: str = "player"
    pot_settle_s: float = 0.8           # wait after a press before reading the pot
    pot_min_samples: int = 3
    pot_tolerance: Optional[float] = None  # default: half the chip unit
    track_pot_bias: bool = True
    max_pot_bias: float = 60.0          # beyond this the reading is treated as wrong
    gesture_on_check: bool = True       # arm taps the table when the bot checks
    board_zone: str = "board"
    bot_zone: str = "bot_hole"
    seat_zone_prefix: str = "seat_"

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")

    @property
    def bot_deals(self) -> bool:
        return self.mode in ("dealer", "dealer_player")


class GameDirector:
    def __init__(self, table: PhysicalTable, config: DirectorConfig, policy,
                 cards: Optional[CardTracker] = None, pot: Optional[PotTracker] = None):
        if config.mode == "dealer" and table.cfg.bot_seat is not None:
            raise ValueError("dealer mode: set bot_seat to None")
        if config.mode in ("player", "dealer_player") and table.cfg.bot_seat is None:
            raise ValueError(f"{config.mode} mode needs a bot_seat")
        self.table = table
        self.cfg = config
        self.policy = policy
        self.cards = cards or CardTracker()
        self.pot = pot or PotTracker()
        self.outbox: Deque[Any] = deque()
        self.manual_cards: Dict[str, List[str]] = {}
        self._ids = itertools.count(1)
        self._pending_arm: Optional[ArmCommand] = None
        self._arm_failed: Optional[ArmCommand] = None
        self._press_at: Optional[float] = None
        self._fold_pressed = False
        self._next_hand_requested = True   # first hand starts on its own
        self._deal_queue: Deque[int] = deque()
        self._cards_dealt: Dict[int, int] = {}
        self._dealt_physically = False
        self._board_cmds_sent = 0
        self._burn_sent = False
        self._decision: Optional[Decision] = None
        self._bias = 0.0
        self._refunds_seen = 0
        self._verify: Optional[tuple] = None
        self.waiting = "starting"
        self.audit: List[dict] = []
        self.last_inference: Optional[Inferred] = None

    # ================================================================ inputs
    def press_turn_done(self, now: float) -> None:
        if self._arm_failed is not None:
            self._accept_failed_arm()
            return
        ph = self.table.phase()
        if ph.kind == "over":
            self._next_hand_requested = True
        elif self._human_can_press(ph):
            self._press_at = now
            self._fold_pressed = False
            self._verify = None      # next player's chips would pollute the bot-bet check
        else:
            self._say("Button ignored: not waiting for a player right now", "warn")

    def press_fold(self, now: float) -> None:
        ph = self.table.phase()
        if ph.kind == "bet" and self._human_can_press(ph):
            self._press_at = now
            self._fold_pressed = True
            self._verify = None
        else:
            self._say("Fold ignored: not a human player's turn", "warn")

    def _human_can_press(self, ph: Phase) -> bool:
        return (ph.kind in ("bet", "showdown") and ph.seat != self.table.cfg.bot_seat
                and self._pending_arm is None)

    def request_next_hand(self) -> None:
        self._next_hand_requested = True

    def set_manual_cards(self, zone: str, codes: List[str]) -> None:
        """Operator override for a zone (testing without vision, or misreads)."""
        if codes:
            self.manual_cards[zone] = list(codes)
        else:
            self.manual_cards.pop(zone, None)

    def manual_action(self, chair: int, action: str, raise_to: Optional[int] = None) -> None:
        """Operator override: apply a human player's action directly."""
        self._apply_human(chair, action, raise_to, note="manual override")

    def arm_finished(self, cmd_id: int, ok: bool, message: str, now: float) -> None:
        cmd = self._pending_arm
        if cmd is None or cmd.id != cmd_id:
            return
        self._pending_arm = None
        if not ok:
            self._arm_failed = cmd
            self._say(f"Arm failed on '{cmd.skill}' ({message}). Fix it by hand, then press "
                      "turn-done to continue as if it succeeded.", "error")
            return
        self._on_arm_success(cmd, now)

    # ================================================================ main loop
    def poll(self, now: float) -> None:
        for _ in range(50):           # advance as far as the inputs allow
            if not self._step(now):
                return

    def _step(self, now: float) -> bool:
        if self._pending_arm is not None:
            self.waiting = f"arm: {self._pending_arm.skill}"
            return False
        if self._arm_failed is not None:
            self.waiting = "operator: arm failed, press turn-done when fixed"
            return False
        self._check_verify(now)
        t = self.table
        ph = t.phase()

        if ph.kind == "over":
            return self._step_between_hands()
        if ph.kind == "deal_hole":
            return self._step_hole_cards()
        if ph.kind in ("burn", "board"):
            return self._step_board()
        if ph.kind == "bet":
            if ph.seat == t.cfg.bot_seat:
                return self._step_bot_turn()
            return self._step_human_turn(ph.seat, now)
        if ph.kind == "showdown":
            return self._step_showdown(ph.seat, now)
        return False

    # ---------------------------------------------------------------- between hands
    def _step_between_hands(self) -> bool:
        t = self.table
        if t.match_over:
            self.waiting = "match over"
            return False
        if not self._next_hand_requested:
            self.waiting = "press turn-done (or next hand) to start the next hand"
            return False
        if not self._zones_clear():
            self.waiting = "clear the board and the bot's card holder"
            return False
        self._next_hand_requested = False
        t.start_hand()
        self.cards.clear()
        self.manual_cards.clear()
        self._bias = 0.0
        self._press_at = None
        self._deal_queue.clear()
        self._cards_dealt = {}
        self._dealt_physically = False
        self._board_cmds_sent = 0
        self._burn_sent = False
        self._say(t.log[0])
        bot = t.cfg.bot_seat
        if bot is not None and t.bot_in_hand:
            blind = t.bet_of(bot)
            if blind:
                self._arm("post_blind", chips=blind, chair=bot)
        return True

    def _zones_clear(self) -> bool:
        if not self.cards.frames:
            return True       # no vision running yet
        if len(self.cards.frames) < self.cards.window:
            return False
        return not (self._stable(self.cfg.board_zone) or self._stable(self.cfg.bot_zone))

    # ---------------------------------------------------------------- hole cards
    def _step_hole_cards(self) -> bool:
        t = self.table
        if self.cfg.bot_deals and not self._dealt_physically:
            if not self._deal_queue:
                self._deal_queue.extend(t.order * 2)    # two rounds, SB first
                self._arm("reset_deck")                 # fresh shuffled deck in the holder
                return True
            chair = self._deal_queue.popleft()
            if not self._deal_queue:
                self._dealt_physically = True
            to_bot = chair == t.cfg.bot_seat
            params = {"chair": chair, "face_up": to_bot,
                      "card_index": self._cards_dealt.get(chair, 0)}
            self._cards_dealt[chair] = params["card_index"] + 1
            self._arm("deal_card", **params)
            return True
        if t.bot_in_hand:
            mine = self._stable(self.cfg.bot_zone)
            if len(mine) < 2:
                self.waiting = "bot hole cards (place them face-up in the bot holder)"
                return False
            if len(mine) > 2:
                self.waiting = f"too many cards in the bot holder: {mine}"
                return False
            t.deal_all_holes(mine)
            self._say(f"Bot hole cards read ({len(mine)} cards).")
        else:
            t.deal_all_holes(())
        return True

    # ---------------------------------------------------------------- board
    def _step_board(self) -> bool:
        t = self.table
        count = 3 if not t.board else 1
        slot0 = len(t.board)
        if self.cfg.bot_deals:
            if not self._burn_sent:
                self._burn_sent = True
                self._arm("burn")
                return True
            if self._board_cmds_sent < count:
                slot = slot0 + self._board_cmds_sent
                self._board_cmds_sent += 1
                self._arm("deal_board", slot=slot)
                return True
        seen = self._stable(self.cfg.board_zone)
        new = [c for c in seen if c not in t.board]
        missing = [c for c in t.board if c not in seen]
        if len(new) < count:
            street = {0: "flop", 3: "turn", 4: "river"}.get(slot0, "board")
            self.waiting = f"{street} card(s) on the board ({len(new)}/{count} seen)"
            return False
        if len(new) > count:
            self.waiting = f"board shows {len(new)} new cards, expected {count}: {new}"
            return False
        blocked = set(t.bot_hole_cards) & set(new)
        if blocked:
            self.waiting = f"card(s) {sorted(blocked)} seen on board and in bot hand"
            return False
        if missing:
            self._say(f"Board cards {missing} no longer visible; continuing", "warn")
        if t.phase().kind == "burn":
            t.burn()
        t.deal_board(new)
        self._say(f"Board: {' '.join(t.board)}")
        self._after_hand()
        self._burn_sent = False
        self._board_cmds_sent = 0
        return True

    # ---------------------------------------------------------------- bot turn
    def _step_bot_turn(self) -> bool:
        t = self.table
        obs = t.observation()
        decision = self.policy.choose(obs)
        self._decision = decision
        self.audit.append({"hand": t.hand_number, "stage": obs.stage, "decision": decision})
        bot = t.cfg.bot_seat
        if decision.action == "fold":
            self._arm("fold", chair=bot)
        elif decision.action == "check":
            if self.cfg.gesture_on_check:
                self._arm("check", chair=bot)
            else:
                self._apply_bot(decision)
        else:
            if decision.action == "call":
                chips = obs.legal_actions.call_amount
                raise_to = None
            else:
                chips = decision.raise_to - obs.my_bet
                raise_to = decision.raise_to
            self._arm("bet", chair=bot, action=decision.action, chips=int(chips),
                      raise_to=raise_to, all_in=int(chips) >= obs.my_stack)
        label = decision.action if decision.action != "raise" else f"raise to {decision.raise_to}"
        self._say(f"Bot decides: {label}")
        return True

    def _apply_bot(self, decision: Decision) -> None:
        t = self.table
        t.act(t.cfg.bot_seat, decision.action, decision.raise_to)
        self._decision = None
        self._announce_refunds()

    # ---------------------------------------------------------------- human turn
    def _step_human_turn(self, chair: int, now: float) -> bool:
        t = self.table
        legal = t.legal(chair)
        owe = f"call {legal.call_amount}" if legal.call_amount else "check"
        self.waiting = f"chair {chair} to act ({owe}); press turn-done when finished"
        if self._press_at is None or now - self._press_at < self.cfg.pot_settle_s:
            return False
        if self._fold_pressed:
            reading = self.pot.value_since(self._press_at, 1)
            if reading is None:
                reading = t.pot + self._bias
        else:
            reading = self.pot.value_since(self._press_at, self.cfg.pot_min_samples)
            if reading is None:
                self.waiting = f"chair {chair}: waiting for a pot reading"
                return False
        before = t.pot + self._bias
        inf = infer_action(legal, t.bet_of(chair), t.stack_of(chair), before, reading,
                           t.cfg.chip_unit, self.cfg.pot_tolerance, self._fold_pressed)
        self.last_inference = inf
        self._press_at = None
        self._fold_pressed = False
        if inf.action is None:
            self._say(f"Chair {chair}: {inf.message}", "warn")
            return True
        if inf.message:
            self._say(f"Chair {chair}: {inf.message}", "warn" if not inf.confident else "info")
        self._apply_human(chair, inf.action, inf.raise_to, note="inferred", observed_pot=reading if inf.confident else None)
        return True

    def _apply_human(self, chair, action, raise_to, note="", observed_pot=None) -> None:
        t = self.table
        try:
            t.act(chair, action, raise_to)
        except ValueError as e:
            self._say(f"Chair {chair}: {action} rejected by engine ({e})", "error")
            return
        self._press_at = None
        label = action if action != "raise" else f"raise to {raise_to}"
        self._say(f"Chair {chair}: {label} ({note})")
        refund = self._announce_refunds()
        if observed_pot is not None and self.cfg.track_pot_bias and t.state.status:
            bias = observed_pot - (t.pot + refund)
            if abs(bias) <= self.cfg.max_pot_bias:
                self._bias = bias
            else:
                self._say(f"Pot reading off by {bias:+.0f}; ignoring it for calibration", "warn")

    # ---------------------------------------------------------------- showdown
    def _step_showdown(self, chair: int, now: float) -> bool:
        t = self.table
        if chair == t.cfg.bot_seat:
            t.reveal(chair, t.bot_hole_cards)
            self._say(f"Bot shows {' '.join(t.bot_hole_cards)}")
            self._after_hand()
            return True
        zone = f"{self.cfg.seat_zone_prefix}{chair}"
        cards = [c for c in self._stable(zone) if c not in t.board and c not in t.bot_hole_cards]
        if len(cards) == 2:
            t.reveal(chair, cards)
            self._after_hand()
            return True
        all_in_runout = len(t.board) < 5
        if self._press_at is not None and now - self._press_at >= self.cfg.pot_settle_s:
            self._press_at = None
            if all_in_runout:
                self._say(f"Chair {chair}: all-in hands must be turned face-up, not mucked", "warn")
                return True
            t.reveal(chair, None)        # press with no cards turned over = muck
            self._after_hand()
            return True
        self.waiting = f"showdown: chair {chair} turn cards face-up in their zone, or press to muck"
        return False

    def _after_hand(self) -> None:
        t = self.table
        if t.phase().kind == "over":
            pay = t.payouts()
            self._say("Hand over. Pot goes to: " + ", ".join(f"chair {c}: {v}" for c, v in pay.items()))

    # ---------------------------------------------------------------- arm plumbing
    def _arm(self, skill: str, **params) -> None:
        cmd = ArmCommand(next(self._ids), skill, params)
        self._pending_arm = cmd
        self.outbox.append(cmd)
        self.waiting = f"arm: {skill}"

    def _on_arm_success(self, cmd: ArmCommand, now: float) -> None:
        if cmd.skill in ("bet", "check", "fold"):
            decision = self._decision
            if decision is not None:
                self._apply_bot(decision)
                if cmd.skill == "bet" and self.table.state.status:
                    self._verify = (now, self.table.pot)
        # deal_card / burn / deal_board / post_blind: vision confirms the result

    def _accept_failed_arm(self) -> None:
        cmd, self._arm_failed = self._arm_failed, None
        self._say(f"Operator confirmed '{cmd.skill}' was completed by hand")
        self._on_arm_success(cmd, 0.0)

    def _check_verify(self, now: float) -> None:
        if self._verify is None:
            return
        t0, expected = self._verify
        if now - t0 < self.cfg.pot_settle_s:
            return
        reading = self.pot.value_since(t0, self.cfg.pot_min_samples)
        if reading is None:
            if now - t0 > 5 * self.cfg.pot_settle_s:
                self._verify = None
            return
        self._verify = None
        err = reading - (expected + self._bias)
        tol = self.cfg.pot_tolerance or self.table.cfg.chip_unit / 2
        if abs(err) > max(tol, self.table.cfg.chip_unit):
            self._say(f"After the bot's bet the pot reads {reading:.0f}, expected "
                      f"{expected + self._bias:.0f}; check the bot's chips", "warn")
        elif self.cfg.track_pot_bias and abs(reading - expected) <= self.cfg.max_pot_bias:
            self._bias = reading - expected

    def _announce_refunds(self) -> int:
        """Tell the table about uncalled chips PokerKit handed back; returns the total."""
        events = self.table.refund_events[self._refunds_seen:]
        self._refunds_seen = len(self.table.refund_events)
        for _hand, chair, amount in events:
            who = "the bot" if chair == self.table.cfg.bot_seat else f"chair {chair}"
            self._say(f"Return {amount} uncalled chips from the pot to {who}")
        return sum(e[2] for e in events)

    # ---------------------------------------------------------------- helpers
    def _stable(self, zone: str) -> List[str]:
        if zone in self.manual_cards:
            return list(self.manual_cards[zone])
        return self.cards.stable(zone)

    def _say(self, text: str, level: str = "info") -> None:
        self.outbox.append(Announce(text, level))

    def status(self) -> dict:
        snap = self.table.snapshot() if self.table.state is not None else {}
        snap.update({"mode": self.cfg.mode, "waiting": self.waiting,
                     "pot_bias": self._bias,
                     "arm_busy": self._pending_arm is not None})
        return snap
