"""Unit + integration tests for the ROS-free core.  Run:  python3 -m pytest src/poker_logic/test"""
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poker_logic.core.chips import make_change                                  # noqa: E402
from poker_logic.core.equity import estimate_equity                             # noqa: E402
from poker_logic.core.inference import infer_action                             # noqa: E402
from poker_logic.core.observation import LegalActions                           # noqa: E402
from poker_logic.core.policy import EquityBot, BotConfig, quantise_raise, scale_threshold  # noqa: E402
from poker_logic.core.skills import ArmState, MovePose, Pump, SkillCompiler, validate_layout  # noqa: E402
from poker_logic.core.table import PhysicalTable, TableConfig                   # noqa: E402
from poker_logic.core.tracker import CardTracker, PotTracker                    # noqa: E402
from poker_logic.sim import build                                               # noqa: E402

CONFIG = os.path.join(os.path.dirname(__file__), "..", "config")


def legal(call=0, lo=20, hi=1000, can_raise=True):
    return LegalActions(call > 0, call == 0, call > 0, call, can_raise, lo if can_raise else None,
                        hi if can_raise else None)


# ---------------------------------------------------------------- inference
@pytest.mark.parametrize("call,before,after,fold,expect,to", [
    (0, 100, 100, False, "check", None),
    (10, 100, 101, False, "fold", None),       # noise below half a chip, owed 10
    (10, 100, 110, False, "call", None),
    (10, 100, 111.5, False, "call", None),     # noisy but snaps to 10
    (10, 100, 130, False, "raise", 30),
    (10, 100, 100, True, "fold", None),
    (0, 100, 140, False, "raise", 40),         # opening bet
])
def test_inference(call, before, after, fold, expect, to):
    inf = infer_action(legal(call, lo=20 if call else 10), 0, 990, before, after, 5, fold_signal=fold)
    assert inf.action == expect
    assert inf.raise_to == to


def test_inference_under_raise_and_short_call():
    # owes 20, min raise to 40: +25 is under half a raise increment -> call; +30 -> min raise
    assert infer_action(legal(20, lo=40), 0, 990, 0, 25, 5).action == "call"
    under = infer_action(legal(20, lo=40), 0, 990, 0, 30, 5)
    assert under.action == "raise" and under.raise_to == 40 and not under.confident
    assert infer_action(legal(10, lo=20), 0, 990, 0, 5, 5).action is None        # incomplete call
    allin = infer_action(legal(100, lo=200, hi=40), 0, 40, 0, 40, 5)
    assert allin.action == "call" and allin.delta == 40                          # short all-in call


# ---------------------------------------------------------------- chips
def test_make_change():
    assert make_change(135, [5, 25, 100]) == {100: 1, 25: 1, 5: 2}
    assert make_change(100, [5, 25, 100], {100: 0, 25: 4, 5: 10}) == {25: 4}
    with pytest.raises(ValueError):
        make_change(7, [5, 25])


# ---------------------------------------------------------------- equity / policy
def test_equity_sane():
    hu = estimate_equity(["As", "Ah"], [], 800, (0,))
    multi = estimate_equity(["As", "Ah"], [], 800, (0, 0, 0, 0))
    assert 0.80 < hu.equity < 0.90
    assert 0.48 < multi.equity < 0.62          # AA vs 4 random hands ~ 0.56
    river = estimate_equity(["As", "Ah"], ["Ad", "Ac", "2c", "7h", "9s"], 200, (0,))
    assert river.method == "river_exact" and river.equity > 0.99


def test_threshold_scaling():
    assert scale_threshold(0.63, 1) == pytest.approx(0.63)
    assert scale_threshold(0.5, 4) == pytest.approx(0.2)
    assert quantise_raise(27, 20, 1000, 5) == 30
    assert quantise_raise(996, 20, 997, 5) == 997      # rounds past the cap -> all-in


def test_bot_decisions_always_legal():
    t = PhysicalTable(TableConfig(seats=5, bot_seat=2))
    t.start_hand()
    t.deal_all_holes(["7c", "2d"])
    bot = EquityBot(BotConfig(samples=100))
    while t.phase().kind == "bet":
        chair = t.phase().seat
        if chair == 2:
            d = bot.choose(t.observation())
            t.act(2, d.action, d.raise_to)
        else:
            lg = t.legal(chair)
            t.act(chair, "raise", lg.min_raise_to) if lg.can_raise and chair == 3 else t.act(
                chair, "call" if lg.can_call else "check")
        if t.phase().kind in ("burn", "board"):
            break
    assert t.phase().kind in ("burn", "board", "over")


# ---------------------------------------------------------------- table
def test_seat_mapping_and_button_rotation():
    t = PhysicalTable(TableConfig(seats=5, bot_seat=0, first_button=4))
    t.start_hand()
    assert t.order == [0, 1, 2, 3, 4]                # chair 0 = SB, chair 4 = button
    assert t.bet_of(0) == 5 and t.bet_of(1) == 10
    t.deal_all_holes(["Ah", "Kh"])
    assert t.bot_hole_cards == ["Ah", "Kh"]
    assert t.phase().seat == 2                       # UTG acts first
    for c in (2, 3, 4, 0):
        t.act(c, "fold")
    assert t.phase().kind == "over" and t.stacks[1] == 1005
    t.start_hand()
    assert t.button == 0 and t.order == [1, 2, 3, 4, 0]


def test_showdown_with_revealed_cards():
    t = PhysicalTable(TableConfig(seats=3, bot_seat=0))
    t.start_hand()
    t.deal_all_holes(["As", "Ad"])
    while t.phase().kind == "bet":
        lg = t.legal(t.phase().seat)
        t.act(t.phase().seat, "call" if lg.can_call else "check")
    for street in (["2c", "7h", "9s"], ["Jd"], ["3c"]):
        t.burn()
        t.deal_board(street)
        while t.phase().kind == "bet":
            t.act(t.phase().seat, "check")
    while t.phase().kind == "showdown":
        c = t.phase().seat
        t.reveal(c, None if c == 0 else (["Kc", "Kd"] if c == 1 else ["Qc", "Qd"]))
    assert t.phase().kind == "over"
    assert t.stacks[0] == 1020 and sum(t.stacks.values()) == 3000


# ---------------------------------------------------------------- trackers
def test_card_tracker_debounce():
    from poker_logic.core.equity import normalise
    assert normalise("10h") == "Th" and normalise("qS") == "Qs"
    tr = CardTracker(window=5, ratio=0.6)
    for i in range(5):
        dets = [("Ah", "board", 0.9), ("Kd", "board", 0.9)] + ([("2c", "board", 0.9)] if i == 0 else [])
        tr.update(dets)
    assert tr.stable("board") == ["Ah", "Kd"]
    pt = PotTracker()
    for i, v in enumerate([10, 11, 9, 50, 10]):
        pt.add(i, v)
    assert pt.value_since(0) == 10


# ---------------------------------------------------------------- skills
def test_layout_and_skills():
    with open(os.path.join(CONFIG, "arm_layout.yaml")) as f:
        layout = yaml.safe_load(f)
    assert validate_layout(layout, 5) == []
    sc = SkillCompiler(layout)
    steps, state, change = sc.bot_action("raise", 135, ArmState())
    assert change == {100: 1, 25: 1, 5: 2}
    assert sum(isinstance(s, Pump) and s.on for s in steps) == 4
    assert state.tray_counts[5] == 28
    assert all(isinstance(s, (MovePose, Pump)) for s in steps)
    steps, state, _ = sc.dealer_action("deal_board", ArmState(deck_count=52), index=2)
    assert state.deck_count == 51
    flip = [s for s in steps if isinstance(s, MovePose) and s.roll == pytest.approx(3.13)]
    assert flip and flip[0].x == pytest.approx(0.0)        # board slot 2 is x = 0.0


# ---------------------------------------------------------------- full simulated games
@pytest.mark.parametrize("mode", ["player", "dealer", "dealer_player"])
@pytest.mark.parametrize("seats", [4, 5])
def test_simulated_games(mode, seats):
    world = build(mode, seats=seats, seed=seats * 7, samples=80)
    played = world.run(15)
    assert played >= 1
    assert sum(world.t.stacks.values()) == seats * 1000
    assert not [m for m in world.messages if m.level == "error"]
