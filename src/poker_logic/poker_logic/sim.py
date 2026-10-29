"""Offline table simulator: exercises the whole decision stack without ROS or hardware.

A fake "world" plays the human players, deals real shuffled cards, produces
noisy per-frame card detections and pot readings, and executes the director's
arm commands instantly. Use it to test modes, inference and policy changes:

    python3 -m poker_logic.sim --mode dealer_player --seats 5 --hands 50
"""
import argparse
import random
from typing import Dict, List

from .core.chips import make_change
from .core.director import Announce, ArmCommand, DirectorConfig, GameDirector
from .core.equity import DECK
from .core.policy import BotConfig, make_policy
from .core.table import PhysicalTable, TableConfig
from .core.tracker import CardTracker, PotTracker


class FakeWorld:
    def __init__(self, director: GameDirector, rng: random.Random, pot_noise=1.0,
                 miss_rate=0.1, verbose=False, denominations=(5, 25, 100)):
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
        self._refunds_done = 0
        self.hand_seen = 0
        self.arm_log: List[ArmCommand] = []
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
        self.humans_dealt = False
        self.street_dealt = 0

    def _pay(self, chair, amount):
        if amount <= 0:
            return
        self.physical_stacks[chair] -= amount
        self.contrib[chair] = self.contrib.get(chair, 0) + amount
        self.pot += amount

    def frame(self):
        dets = []
        for c in self.board:
            dets.append((c, "board", 0.9))
        for c in self.bot_cards:
            dets.append((c, "bot_hole", 0.9))
        for chair, cards in self.shown.items():
            for c in cards:
                dets.append((c, f"seat_{chair}", 0.9))
        dets = [d for d in dets if self.rng.random() > self.miss_rate]
        self.d.cards.update(dets)
        self.d.pot.add(self.now, max(0.0, self.pot + self.rng.gauss(0, self.pot_noise)), 0.9)

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
    def human_dealer(self):
        """Player mode: humans deal hole cards and the board."""
        ph = self.t.phase()
        if self.d.cfg.bot_deals:
            return
        if ph.kind == "deal_hole" and not self.humans_dealt:
            self.humans_dealt = True
            for _ in range(2):
                for chair in self.t.order:
                    card = self.deck.pop()
                    if chair == self.t.cfg.bot_seat:
                        self.bot_cards.append(card)
                    else:
                        self.hands.setdefault(chair, []).append(card)
        if ph.kind in ("burn", "board"):
            want = len(self.t.board) + (3 if not self.t.board else 1)
            if len(self.board) < want and self.street_dealt != want:
                self.street_dealt = want
                self.deck.pop()
                while len(self.board) < want:
                    self.board.append(self.deck.pop())

    def human_turn(self, chair):
        legal = self.t.legal(chair)
        bet = self.t.bet_of(chair)
        stack = self.t.stack_of(chair)
        r = self.rng.random()
        fold_button = False
        if legal.call_amount and r < 0.15:
            amount = 0
            fold_button = self.rng.random() < 0.5
        elif legal.can_raise and r > 0.8:
            lo, hi = legal.min_raise_to, legal.max_raise_to
            target = self.rng.choice([lo, lo, lo, 2 * lo, 3 * lo, hi if self.rng.random() < 0.1 else lo])
            target = hi if target >= hi else min(hi, max(lo, (target // 5) * 5))
            amount = target - bet
        else:
            amount = min(legal.call_amount, stack)
        self._pay(chair, amount)
        if fold_button:
            self.d.press_fold(self.now)
        else:
            self.d.press_turn_done(self.now)

    def human_showdown(self, chair):
        if chair not in self.shown:
            if self.rng.random() < 0.9 or len(self.t.board) < 5:
                self.shown[chair] = list(self.hands[chair])
            else:
                self.shown[chair] = []
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
        played = 0
        settled = 0
        acted_key = None
        for _ in range(max_ticks):
            self.now += 0.1
            self.frame()
            self.d.poll(self.now)
            while self.d.outbox:
                msg = self.d.outbox.popleft()
                if isinstance(msg, ArmCommand):
                    self.run_arm(msg)
                else:
                    self.messages.append(msg)
                    if self.verbose:
                        print(f"[{msg.level}] {msg.text}")
                    if msg.level == "error":
                        raise RuntimeError(msg.text)
            t = self.t
            for _h, chair, amount in t.refund_events[self._refunds_done:]:
                self.physical_stacks[chair] += amount   # dealer hands back uncalled chips
                self.pot -= amount
            self._refunds_done = len(t.refund_events)
            if self.hand_seen != t.hand_number:
                # new hand: human blinds are placed physically
                self.hand_seen = t.hand_number
                for chair in t.order:
                    if chair != t.cfg.bot_seat:
                        self._pay(chair, t.public_contribution(chair))
            ph = t.phase()
            self.human_dealer()
            if ph.kind == "bet" and ph.seat != t.cfg.bot_seat:
                key = (t.hand_number, len(t.public_actions))
                if acted_key != key and self.d._press_at is None:
                    acted_key = key
                    self.human_turn(ph.seat)
            elif ph.kind == "showdown" and ph.seat != t.cfg.bot_seat:
                self.human_showdown(ph.seat)
            elif ph.kind == "over" and t.hand_number > settled:
                settled = t.hand_number
                self.settle_hand()
                played += 1
                if played >= hands or t.match_over:
                    return played
                self._new_hand_world()
                self.d.request_next_hand()
        raise RuntimeError(f"simulation stalled: {self.d.waiting}\n" + "\n".join(self.t.log[-15:]))


def build(mode="player", seats=5, bot_seat=0, policy="equity", samples=300, seed=0,
          pot_noise=1.0, miss_rate=0.1, verbose=False):
    cfg = TableConfig(seats=seats, bot_seat=None if mode == "dealer" else bot_seat)
    table = PhysicalTable(cfg)
    director = GameDirector(table, DirectorConfig(mode=mode), make_policy(policy, BotConfig(samples=samples)),
                            CardTracker(window=5, ratio=0.6), PotTracker())
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
    ap.add_argument("--pot-noise", type=float, default=1.0, help="std-dev of pot readings (chips)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    world = build(a.mode, a.seats, a.bot_seat, a.policy, a.samples, a.seed, a.pot_noise, 0.1, a.verbose)
    played = world.run(a.hands)
    t = world.t
    print(f"{played} hands, mode={a.mode}; final stacks {t.stacks}")
    if t.cfg.bot_seat is not None:
        print(f"bot (chair {t.cfg.bot_seat}) net: {t.stacks[t.cfg.bot_seat] - t.cfg.starting_stack:+d}; "
              f"arm commands issued: {len(world.arm_log)}")


if __name__ == "__main__":
    main()
