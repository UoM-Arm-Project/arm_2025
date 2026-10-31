"""Offline table simulator: exercises the whole decision stack without ROS or hardware.

A fake "world" plays the human players and the human dealer, deals real
shuffled cards, answers the director's on-demand camera requests (with
occasional misreads and pot noise), presses the button when prompted, and
executes the arm commands instantly. Use it to test modes, inference and
policy changes:

    python3 -m poker_logic.sim --mode dealer_player --seats 5 --hands 50
"""
import argparse
import random
from typing import Dict, List

from .core.chips import make_change
from .core.director import Announce, ArmCommand, DirectorConfig, GameDirector, VisionRequest
from .core.equity import DECK
from .core.policy import BotConfig, make_policy
from .core.table import PhysicalTable, TableConfig


class FakeWorld:
    def __init__(self, director: GameDirector, rng: random.Random, pot_noise=0.0,
                 miss_rate=0.1, verbose=False, denominations=(1, 5, 10)):
        self.d = director
        self.t = director.table
        self.rng = rng
        self.pot_noise = pot_noise
        self.miss_rate = miss_rate
        self.verbose = verbose
        self.denoms = list(denominations)
        self.now = 0.0
        self.physical_stacks: Dict[int, int] = dict(self.t.stacks)
        self.messages: List[Announce] = []
        self.arm_log: List[ArmCommand] = []
        self.reads = {"cards": 0, "pot": 0}
        self._refunds_done = 0
        self.hand_seen = 0
        self._new_hand_world()

    # ------------------------------------------------------------- world state
    def _new_hand_world(self):
        self.deck = list(DECK)
        self.rng.shuffle(self.deck)
        self.board: List[str] = []
        self.bot_cards: List[str] = []
        self.hands: Dict[int, List[str]] = {}
        self.shown: Dict[int, List[str]] = {}
        self.pot = 0.0
        self.contrib: Dict[int, int] = {}
        self.holes_dealt = False
        self.paid_turn = None

    def _pay(self, chair, amount):
        if amount <= 0:
            return
        self.physical_stacks[chair] -= amount
        self.contrib[chair] = self.contrib.get(chair, 0) + amount
        self.pot += amount

    # ------------------------------------------------------------- camera
    def answer(self, req: VisionRequest):
        self.reads[req.kind] += 1
        if req.kind == "pot":
            value = max(0.0, self.pot + self.rng.gauss(0, self.pot_noise))
            self.d.vision_done(req.id, True, "", value=value, now=self.now)
            return
        if req.zone == "board":
            cards = list(self.board)
        elif req.zone == "bot_hole":
            cards = list(self.bot_cards)
        else:
            cards = list(self.shown.get(int(req.zone.split("_")[1]), []))
        if cards and self.rng.random() < self.miss_rate:      # a corner hidden / misdetected
            cards.pop(self.rng.randrange(len(cards)))
        ok = req.expected == 0 or len(cards) == req.expected
        self.d.vision_done(req.id, ok, "" if ok else "count mismatch", cards=cards, now=self.now)

    # ------------------------------------------------------------- arm
    def run_arm(self, cmd: ArmCommand):
        self.arm_log.append(cmd)
        p = cmd.params
        bot = self.t.cfg.bot_seat
        if cmd.skill in ("bet", "post_blind"):
            make_change(p["chips"], self.denoms)          # must be payable in chips
            self._pay(bot, p["chips"])
        elif cmd.skill == "fold":
            self.bot_cards = []
        elif cmd.skill == "deal_card":
            card = self.deck.pop()
            if p["face_up"]:
                self.bot_cards.append(card)
            else:
                self.hands.setdefault(p["chair"], []).append(card)
        elif cmd.skill == "reset_deck":
            assert len(self.deck) == 52, "deck reset mid-hand"
        elif cmd.skill == "burn":
            self.deck.pop()
        elif cmd.skill == "deal_board":
            assert p["slot"] == len(self.board), (p, self.board)
            self.board.append(self.deck.pop())
        self.d.arm_finished(cmd.id, True, "ok", self.now)

    # ------------------------------------------------------------- humans
    def deal_holes(self):
        if self.holes_dealt:
            return
        self.holes_dealt = True
        for _ in range(2):
            for chair in self.t.order:
                card = self.deck.pop()
                if chair == self.t.cfg.bot_seat:
                    self.bot_cards.append(card)
                else:
                    self.hands.setdefault(chair, []).append(card)

    def deal_street(self):
        want = len(self.t.board) + (3 if not self.t.board else 1)
        if len(self.board) < want:
            self.deck.pop()                                  # burn
            while len(self.board) < want:
                self.board.append(self.deck.pop())

    def human_turn(self, chair):
        key = (self.t.hand_number, len(self.t.public_actions))
        if self.paid_turn == key:                            # re-prompted after a bad read
            self.d.press_turn_done(self.now)
            return
        self.paid_turn = key
        legal = self.t.legal(chair)
        bet, stack = self.t.bet_of(chair), self.t.stack_of(chair)
        r = self.rng.random()
        if legal.call_amount and r < 0.15:
            if self.rng.random() < 0.5:
                self.d.press_fold(self.now)
                return
            amount = 0
        elif legal.can_raise and r > 0.8:
            lo, hi = legal.min_raise_to, legal.max_raise_to
            target = self.rng.choice([lo, lo, lo, 2 * lo, 3 * lo, hi if self.rng.random() < 0.1 else lo])
            amount = min(hi, max(lo, target)) - bet
        else:
            amount = min(legal.call_amount, stack)
        self._pay(chair, amount)
        self.d.press_turn_done(self.now)

    def human_showdown(self, chair):
        if chair not in self.shown:
            if self.rng.random() < 0.9 or len(self.t.board) < 5:
                self.shown[chair] = list(self.hands[chair])
            else:
                self.shown[chair] = []
                self.d.press_fold(self.now)                  # muck
                return
        self.d.press_turn_done(self.now)

    def settle_hand(self):
        pay = self.t.payouts()
        for chair, amount in pay.items():
            self.physical_stacks[chair] += amount
            self.pot -= amount
        assert abs(self.pot) < 1e-9, f"pot not empty after payouts: {self.pot}"
        for chair in self.t.order:
            assert self.physical_stacks[chair] == self.t.stacks[chair], (
                f"hand {self.t.hand_number}: chair {chair} physical {self.physical_stacks[chair]} "
                f"!= engine {self.t.stacks[chair]}\n" + "\n".join(self.t.log))

    # ------------------------------------------------------------- loop
    def run(self, hands: int, max_ticks: int = 400000) -> int:
        played = settled = 0
        for _ in range(max_ticks):
            self.now += 0.1
            self.d.poll(self.now)
            while self.d.outbox:
                msg = self.d.outbox.popleft()
                if isinstance(msg, ArmCommand):
                    self.run_arm(msg)
                elif isinstance(msg, VisionRequest):
                    self.answer(msg)
                else:
                    self.messages.append(msg)
                    if self.verbose:
                        print(f"[{msg.level}] {msg.text}")
                    if msg.level == "error":
                        raise RuntimeError(msg.text)
            t = self.t
            for _h, chair, amount in t.refund_events[self._refunds_done:]:
                self.physical_stacks[chair] += amount       # dealer hands back uncalled chips
                self.pot -= amount
            self._refunds_done = len(t.refund_events)
            if self.hand_seen != t.hand_number:             # new hand: human blinds go in
                self.hand_seen = t.hand_number
                for chair in t.order:
                    if chair != t.cfg.bot_seat:
                        self._pay(chair, t.public_contribution(chair))
            if t.state is not None and t.phase().kind == "over" and t.hand_number > settled:
                settled = t.hand_number
                self.settle_hand()
                played += 1
                if played >= hands or t.match_over:
                    return played
                self._new_hand_world()
            expect = self.d.expect
            ph = t.phase()
            if expect == "hole":
                if not self.d.cfg.bot_deals:
                    self.deal_holes()
                self.d.press_turn_done(self.now)
            elif expect == "board":
                if not self.d.cfg.bot_deals:
                    self.deal_street()
                self.d.press_turn_done(self.now)
            elif expect == "turn":
                self.human_turn(ph.seat)
            elif expect == "showdown":
                self.human_showdown(ph.seat)
            elif expect in ("next", "fixed"):
                self.d.press_turn_done(self.now)
        raise RuntimeError(f"simulation stalled: {self.d.waiting}\n" + "\n".join(self.t.log[-15:]))


def build(mode="player", seats=5, bot_seat=0, policy="equity", samples=300, seed=0,
          pot_noise=0.0, miss_rate=0.1, verbose=False):
    cfg = TableConfig(seats=seats, bot_seat=None if mode == "dealer" else bot_seat,
                      small_blind=1, big_blind=2, starting_stack=200, chip_unit=1)
    table = PhysicalTable(cfg)
    director = GameDirector(table, DirectorConfig(mode=mode), make_policy(policy, BotConfig(samples=samples)))
    return FakeWorld(director, random.Random(seed), pot_noise, miss_rate, verbose)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="player", choices=["player", "dealer", "dealer_player"])
    ap.add_argument("--seats", type=int, default=5)
    ap.add_argument("--bot-seat", type=int, default=0)
    ap.add_argument("--hands", type=int, default=20)
    ap.add_argument("--policy", default="equity", choices=["equity", "baseline"])
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pot-noise", type=float, default=0.0, help="std-dev of pot readings (chips)")
    ap.add_argument("--miss-rate", type=float, default=0.1, help="chance a card read misses a card")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    world = build(a.mode, a.seats, a.bot_seat, a.policy, a.samples, a.seed, a.pot_noise, a.miss_rate, a.verbose)
    played = world.run(a.hands)
    t = world.t
    print(f"{played} hands, mode={a.mode}; final stacks {t.stacks}")
    print(f"camera reads: {world.reads['cards']} card, {world.reads['pot']} pot "
          f"({(world.reads['cards'] + world.reads['pot']) / max(1, played):.1f} per hand)")
    if t.cfg.bot_seat is not None:
        print(f"bot (chair {t.cfg.bot_seat}) net: {t.stacks[t.cfg.bot_seat] - t.cfg.starting_stack:+d}; "
              f"arm commands issued: {len(world.arm_log)}")


if __name__ == "__main__":
    main()
