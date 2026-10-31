"""GameDirector: the ROS-free state machine between perception, the engine and the arm.

Perception is *on demand*: the director asks for a reading only when it needs
one, and only after the right person has pressed the button (or the arm has
finished dealing), so no model runs while nothing is happening.

Inputs (all non-blocking):
    * ``press_turn_done`` / ``press_fold`` / ``request_next_hand`` (buttons, console)
    * ``vision_done``   results of :class:`VisionRequest` it issued
    * ``arm_finished``  results of :class:`ArmCommand` it issued

Outputs (``self.outbox``):
    * :class:`VisionRequest` - read cards in a zone, or read the pot
    * :class:`ArmCommand`    - one physical skill for the arm
    * :class:`Announce`      - text for the table; every "press the button" moment
                               is announced once as a clear instruction

``self.expect`` says which press the director is waiting for:
    ``hole``      dealer has dealt; bot's cards are in its holder
    ``board``     dealer has burned and dealt the next street
    ``turn``      the current player has put chips in (or not)
    ``showdown``  the current player has turned their cards face-up
    ``next``      table cleared, start the next hand
    ``fixed``     operator fixed a failed arm move by hand
    ``""``        nothing: the system is busy (arm / vision / deciding)

Modes:
    ``player``         the bot holds a seat, humans deal
    ``dealer``         the arm deals, the bot does not play
    ``dealer_player``  the arm deals and the bot plays a seat
"""
import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from .equity import normalise
from .inference import Inferred, infer_action
from .policy import Decision
from .table import PhysicalTable

MODES = ("player", "dealer", "dealer_player")
STREETS = {0: "flop", 3: "turn", 4: "river"}


@dataclass
class ArmCommand:
    id: int
    skill: str                    # bet, post_blind, check, fold, deal_card, burn, deal_board, reset_deck
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisionRequest:
    id: int
    kind: str                     # cards | pot
    zone: str = ""                # cards: board | bot_hole | seat_<n>
    expected: int = 0             # cards: how many cards should be in the zone
    purpose: str = ""             # hole | board | showdown | turn | verify (for the director)


@dataclass
class Announce:
    text: str
    level: str = "info"           # info, warn, error


@dataclass
class DirectorConfig:
    mode: str = "player"
    pot_settle_s: float = 0.8           # wait after a turn press before photographing the pot
    pot_tolerance: Optional[float] = None  # default: half the chip unit
    track_pot_bias: bool = True
    max_pot_bias: float = 30.0          # beyond this the reading is treated as wrong
    gesture_on_check: bool = True       # arm taps the table when the bot checks
    verify_bot_chips: bool = True       # read the pot after the bot's chips land
    read_retries: int = 1               # automatic re-reads before asking the operator
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
    def __init__(self, table: PhysicalTable, config: DirectorConfig, policy):
        if config.mode == "dealer" and table.cfg.bot_seat is not None:
            raise ValueError("dealer mode: set bot_seat to None")
        if config.mode in ("player", "dealer_player") and table.cfg.bot_seat is None:
            raise ValueError(f"{config.mode} mode needs a bot_seat")
        self.table = table
        self.cfg = config
        self.policy = policy
        self.outbox: Deque[Any] = deque()
        self.manual_cards: Dict[str, List[str]] = {}
        self.manual_pot: Optional[float] = None
        self._ids = itertools.count(1)
        self._pending_arm: Optional[ArmCommand] = None
        self._arm_failed: Optional[ArmCommand] = None
        self._pending_vision: Optional[VisionRequest] = None
        self._press_at: Optional[float] = None     # a turn press waiting for the pot to settle
        self._read_now = False                      # a card read is due (press or arm finished)
        self._retries = 0
        self._next_hand_requested = True            # the first hand starts on its own
        self._deal_queue: Deque[int] = deque()
        self._cards_dealt: Dict[int, int] = {}
        self._dealt_physically = False
        self._board_cmds_sent = 0
        self._burn_sent = False
        self._decision: Optional[Decision] = None
        self._bias = 0.0
        self._refunds_seen = 0
        self.expect = ""
        self.waiting = "starting"
        self._last_prompt = ""
        self.audit: List[dict] = []
        self.last_inference: Optional[Inferred] = None

    # ================================================================ inputs
    def press_turn_done(self, now: float) -> None:
        if self._arm_failed is not None:
            self._accept_failed_arm()
            return
        if self._busy():
            self._say("Button ignored: still busy (arm or camera); press again when asked", "warn")
            return
        if self.expect == "next":
            self._next_hand_requested = True
        elif self.expect == "turn":
            self._press_at = now
        elif self.expect in ("hole", "board", "showdown"):
            self._read_now = True
            self._retries = 0
        else:
            self._say("Button ignored: not waiting for a press right now", "warn")

    def press_fold(self, now: float) -> None:
        if self._busy():
            self._say("Fold ignored: still busy", "warn")
            return
        ph = self.table.phase()
        if self.expect == "turn":
            self._apply_human(ph.seat, "fold", None, note="fold button")
        elif self.expect == "showdown":
            if len(self.table.board) < 5:
                self._say(f"Chair {ph.seat}: all-in hands must be turned face-up, not mucked", "warn")
                return
            self.table.reveal(ph.seat, None)
            self._after_hand()
        else:
            self._say("Fold ignored: not a human player's turn", "warn")

    def request_next_hand(self) -> None:
        self._next_hand_requested = True

    def set_manual_cards(self, zone: str, codes: List[str]) -> None:
        """Operator override for a zone: used instead of the camera when that zone is read."""
        if codes:
            self.manual_cards[zone] = list(codes)
        else:
            self.manual_cards.pop(zone, None)

    def set_manual_pot(self, value: Optional[float]) -> None:
        """Operator override for the pot reading (None = use the camera again)."""
        self.manual_pot = value

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
                      "the button to continue as if it succeeded.", "error")
            return
        self._on_arm_success(cmd, now)

    def vision_done(self, req_id: int, ok: bool, message: str = "", cards: Optional[List[str]] = None,
                    value: Optional[float] = None, now: float = 0.0) -> None:
        req = self._pending_vision
        if req is None or req.id != req_id:
            return
        self._pending_vision = None
        try:
            norm = [normalise(c) for c in (cards or [])]
        except ValueError as e:
            ok, message, norm = False, str(e), []
        handler = {"hole": self._got_hole, "board": self._got_board, "showdown": self._got_showdown,
                   "turn": self._got_pot, "verify": self._got_verify}[req.purpose]
        handler(req, ok, message, norm, value, now)

    # ================================================================ main loop
    def poll(self, now: float) -> None:
        for _ in range(50):           # advance as far as the inputs allow
            if not self._step(now):
                return

    def _busy(self) -> bool:
        return self._pending_arm is not None or self._pending_vision is not None

    def _step(self, now: float) -> bool:
        if self._pending_arm is not None:
            self._wait(f"arm: {self._pending_arm.skill}")
            return False
        if self._arm_failed is not None:
            self._prompt("fixed", "OPERATOR: the arm move failed. Finish it by hand, then press the button.")
            return False
        if self._pending_vision is not None:
            self._wait(f"camera: reading {self._pending_vision.zone or 'pot'}")
            return False
        ph = self.table.phase()
        if ph.kind == "over":
            return self._step_between_hands()
        if ph.kind == "deal_hole":
            return self._step_hole_cards()
        if ph.kind in ("burn", "board"):
            return self._step_board()
        if ph.kind == "bet":
            if ph.seat == self.table.cfg.bot_seat:
                return self._step_bot_turn()
            return self._step_human_turn(ph.seat, now)
        if ph.kind == "showdown":
            return self._step_showdown(ph.seat)
        return False

    # ---------------------------------------------------------------- between hands
    def _step_between_hands(self) -> bool:
        t = self.table
        if t.match_over:
            self._prompt("", "Match over.")
            return False
        if not self._next_hand_requested:
            self._prompt("next", f"DEALER: push the pot to the winner and collect the cards, "
                                 f"then press the button to start hand {t.hand_number + 1}.")
            return False
        self._next_hand_requested = False
        t.start_hand()
        self.manual_cards.clear()
        self._bias = 0.0
        self._press_at = None
        self._read_now = False
        self._retries = 0
        self._deal_queue.clear()
        self._cards_dealt = {}
        self._dealt_physically = False
        self._board_cmds_sent = 0
        self._burn_sent = False
        self._say(t.log[0])
        bot = t.cfg.bot_seat
        if bot is not None and t.bot_in_hand and t.bet_of(bot):
            self._arm("post_blind", chips=t.bet_of(bot), chair=bot)
        return True

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
                self._read_now = True                   # arm finished dealing: read straight away
            to_bot = chair == t.cfg.bot_seat
            params = {"chair": chair, "face_up": to_bot, "card_index": self._cards_dealt.get(chair, 0)}
            self._cards_dealt[chair] = params["card_index"] + 1
            self._arm("deal_card", **params)
            return True
        if not t.bot_in_hand:
            t.deal_all_holes(())
            return True
        if not self._read_now:
            if self.cfg.bot_deals:
                self._prompt("hole", "OPERATOR: make sure the bot's two cards lie face-up in its holder, "
                                     "then press the button.")
            else:
                self._prompt("hole", "DEALER: deal two cards to every player and put the bot's two cards "
                                     "face-up in its holder, then press the button.")
            return False
        self._read_cards(self.cfg.bot_zone, 2, "hole")
        return True

    def _got_hole(self, req, ok, message, cards, value, now):
        t = self.table
        cards = sorted(set(cards))
        if ok and len(cards) == 2:
            t.deal_all_holes(cards)
            self._read_now = False
            self._say(f"Bot hole cards read ({len(cards)} cards).")
            return
        self._read_failed("hole", f"Couldn't read the bot's two cards (saw {cards or 'none'}; {message}). "
                                  "DEALER: straighten them in the holder, then press the button.")

    # ---------------------------------------------------------------- board
    def _step_board(self) -> bool:
        t = self.table
        count = 3 if not t.board else 1
        if self.cfg.bot_deals:
            if not self._burn_sent:
                self._burn_sent = True
                self._arm("burn")
                return True
            if self._board_cmds_sent < count:
                slot = len(t.board) + self._board_cmds_sent
                self._board_cmds_sent += 1
                if self._board_cmds_sent == count:
                    self._read_now = True
                self._arm("deal_board", slot=slot)
                return True
        if not self._read_now:
            street = STREETS.get(len(t.board), "board")
            what = "the flop (3 cards)" if count == 3 else f"the {street} (1 card)"
            if self.cfg.bot_deals:
                self._prompt("board", f"OPERATOR: make sure {what} lies face-up on the board, "
                                      "then press the button.")
            else:
                self._prompt("board", f"DEALER: burn one card and deal {what} face-up on the board, "
                                      "then press the button.")
            return False
        self._read_cards(self.cfg.board_zone, len(t.board) + count, "board")
        return True

    def _got_board(self, req, ok, message, cards, value, now):
        t = self.table
        count = 3 if not t.board else 1
        new = [c for c in dict.fromkeys(cards) if c not in t.board]
        clash = set(new) & set(t.bot_hole_cards)
        problem = ""
        if not ok and not cards:
            problem = message or "no cards read"
        elif len(new) != count:
            problem = f"expected {count} new card(s), read {new or 'none'}"
        elif clash:
            problem = f"{sorted(clash)} are the bot's hole cards"
        if problem:
            self._read_failed("board", f"Board read failed ({problem}). DEALER: check the board cards "
                                       "are face-up and fully in the board area, then press the button.")
            return
        missing = [c for c in t.board if c not in cards]
        if missing:
            self._say(f"Board cards {missing} were not seen this time; keeping them", "warn")
        if t.phase().kind == "burn":
            t.burn()
        t.deal_board(new)
        self._read_now = False
        self._burn_sent = False
        self._board_cmds_sent = 0
        self._say(f"Board: {' '.join(t.board)}")
        self._after_hand()

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
                chips, raise_to = obs.legal_actions.call_amount, None
            else:
                chips, raise_to = decision.raise_to - obs.my_bet, decision.raise_to
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
        if legal.call_amount:
            owe = (f"to call put in {legal.call_amount}, to raise put in more, "
                   "or put in nothing / press fold to fold")
        else:
            owe = "put in chips to bet, or nothing to check"
        if self._press_at is None:
            self._prompt("turn", f"CHAIR {chair}: your turn ({owe}). Then press the button.")
            return False
        if now - self._press_at < self.cfg.pot_settle_s:
            self._wait(f"chair {chair}: letting the pot settle")
            return False
        self._press_at = None
        self._read_pot("turn")
        return True

    def _got_pot(self, req, ok, message, cards, value, now):
        t = self.table
        ph = t.phase()
        chair = ph.seat
        if not ok or value is None:
            self._read_failed("turn", f"Pot read failed ({message}). CHAIR {chair}: press the button again.",
                              retry=lambda: setattr(self, "_press_at", now - self.cfg.pot_settle_s))
            return
        self._retries = 0
        legal = t.legal(chair)
        before = t.pot + self._bias
        inf = infer_action(legal, t.bet_of(chair), t.stack_of(chair), before, value,
                           t.cfg.chip_unit, self.cfg.pot_tolerance, False)
        self.last_inference = inf
        if inf.action is None:
            self._say(f"Chair {chair}: {inf.message}", "warn")
            self._last_prompt = ""      # re-announce the turn prompt
            return
        if inf.message:
            self._say(f"Chair {chair}: {inf.message}", "warn" if not inf.confident else "info")
        self._apply_human(chair, inf.action, inf.raise_to, note=f"pot read {value:.0f}",
                          observed_pot=value if inf.confident else None)

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
    def _step_showdown(self, chair: int) -> bool:
        t = self.table
        if chair == t.cfg.bot_seat:
            t.reveal(chair, t.bot_hole_cards)
            self._say(f"Bot shows {' '.join(t.bot_hole_cards)}")
            self._after_hand()
            return True
        if not self._read_now:
            extra = "" if len(t.board) < 5 else " (fold button = muck)"
            self._prompt("showdown", f"SHOWDOWN, CHAIR {chair}: turn your two cards face-up in your "
                                     f"seat area, then press the button{extra}.")
            return False
        self._read_cards(f"{self.cfg.seat_zone_prefix}{chair}", 2, "showdown")
        return True

    def _got_showdown(self, req, ok, message, cards, value, now):
        t = self.table
        chair = t.phase().seat
        known = set(t.board) | set(t.bot_hole_cards) | {c for v in t.revealed.values() for c in v}
        mine = [c for c in dict.fromkeys(cards) if c not in known]
        if len(mine) == 2:
            t.reveal(chair, mine)
            self._read_now = False
            self._after_hand()
            return
        self._read_failed("showdown", f"Couldn't read chair {chair}'s cards (saw {mine or 'none'}). "
                                      f"CHAIR {chair}: lay both cards flat in your seat area and press "
                                      "the button, or press fold to muck.")

    def _after_hand(self) -> None:
        t = self.table
        if t.phase().kind == "over":
            pay = t.payouts()
            self._say("Hand over. Pot goes to: " + ", ".join(f"chair {c}: {v}" for c, v in pay.items()))

    # ---------------------------------------------------------------- perception plumbing
    def _read_cards(self, zone: str, expected: int, purpose: str) -> None:
        req = VisionRequest(next(self._ids), "cards", zone, expected, purpose)
        if zone in self.manual_cards:                # operator override: answer immediately
            self._pending_vision = req
            self.vision_done(req.id, True, "manual", list(self.manual_cards[zone]))
            return
        self._pending_vision = req
        self.outbox.append(req)
        self._wait(f"camera: reading {zone}")

    def _read_pot(self, purpose: str, expected: int = 0) -> None:
        req = VisionRequest(next(self._ids), "pot", expected=expected, purpose=purpose)
        if self.manual_pot is not None:
            self._pending_vision = req
            self.vision_done(req.id, True, "manual", value=self.manual_pot)
            return
        self._pending_vision = req
        self.outbox.append(req)
        self._wait("camera: reading pot")

    def _read_failed(self, purpose: str, operator_msg: str, retry=None) -> None:
        if self._retries < self.cfg.read_retries:
            self._retries += 1
            self._say(f"Read failed, retrying ({self._retries}/{self.cfg.read_retries})", "warn")
            if retry is not None:
                retry()
            else:
                self._read_now = True
            return
        self._retries = 0
        self._read_now = False
        self._press_at = None
        self._say(operator_msg, "warn")
        self._last_prompt = ""            # make sure the next prompt is announced again

    def _got_verify(self, req, ok, message, cards, value, now):
        if not ok or value is None:
            return
        expected = req.expected
        err = value - (expected + self._bias)
        tol = self.cfg.pot_tolerance or self.table.cfg.chip_unit / 2
        if abs(err) > max(tol, self.table.cfg.chip_unit):
            self._say(f"After the bot's bet the pot reads {value:.0f}, expected "
                      f"{expected + self._bias:.0f}; check the bot's chips", "warn")
        elif self.cfg.track_pot_bias and abs(value - expected) <= self.cfg.max_pot_bias:
            self._bias = value - expected

    # ---------------------------------------------------------------- arm plumbing
    def _arm(self, skill: str, **params) -> None:
        cmd = ArmCommand(next(self._ids), skill, params)
        self._pending_arm = cmd
        self.outbox.append(cmd)
        self._wait(f"arm: {skill}")

    def _on_arm_success(self, cmd: ArmCommand, now: float) -> None:
        if cmd.skill in ("bet", "check", "fold"):
            decision = self._decision
            if decision is not None:
                self._apply_bot(decision)
                if cmd.skill == "bet" and self.cfg.verify_bot_chips and self.table.state.status:
                    self._read_pot("verify", expected=self.table.pot)

    def _accept_failed_arm(self) -> None:
        cmd, self._arm_failed = self._arm_failed, None
        self._say(f"Operator confirmed '{cmd.skill}' was completed by hand")
        self._on_arm_success(cmd, 0.0)

    def _announce_refunds(self) -> int:
        """Tell the table about uncalled chips PokerKit handed back; returns the total."""
        events = self.table.refund_events[self._refunds_seen:]
        self._refunds_seen = len(self.table.refund_events)
        for _hand, chair, amount in events:
            who = "the bot" if chair == self.table.cfg.bot_seat else f"chair {chair}"
            self._say(f"Return {amount} uncalled chips from the pot to {who}")
        return sum(e[2] for e in events)

    # ---------------------------------------------------------------- prompts
    def _prompt(self, expect: str, text: str) -> None:
        """Wait for a specific press; announce the instruction once."""
        self.expect = expect
        self.waiting = text
        if text != self._last_prompt:
            self._last_prompt = text
            self._say(text, "info")

    def _wait(self, text: str) -> None:
        self.expect = ""
        self.waiting = text

    def _say(self, text: str, level: str = "info") -> None:
        self.outbox.append(Announce(text, level))

    def status(self) -> dict:
        snap = self.table.snapshot() if self.table.state is not None else {}
        snap.update({"mode": self.cfg.mode, "waiting": self.waiting, "expect": self.expect,
                     "pot_bias": self._bias, "arm_busy": self._pending_arm is not None})
        return snap
