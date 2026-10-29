"""Showdown equity against explicit hypothetical ranges, for 1..N-1 opponents.

Adapted from the heads-up demo's ``equity.py``:

* heads-up river spots are still enumerated exactly;
* multiway spots use Monte Carlo, sampling one hand per *active* opponent, each
  weighted by that opponent's own public preflop raises;
* hand evaluation uses ``treys`` when installed (~18x faster than PokerKit's
  evaluator, which matters on a Raspberry Pi) and falls back to PokerKit.

The bot never touches the real deck, burn cards or other players' hole cards.
"""
import hashlib
import itertools
import json
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = tuple(r + s for r in RANKS for s in SUITS)


@dataclass(frozen=True)
class EquityResult:
    equity: float          # share of the pot won on average (ties split)
    win_rate: float        # outright wins
    tie_rate: float
    standard_error: float
    trials: int
    method: str
    range_model: str
    opponents: int


# --------------------------------------------------------------------------- evaluators
class _TreysEval:
    name = "treys"

    def __init__(self):
        from treys import Card, Evaluator
        self._ev = Evaluator()
        self._id = {c: Card.new(c) for c in DECK}

    def score(self, hole, board):
        # treys: lower is better -> negate so higher is better
        return -self._ev.evaluate([self._id[c] for c in board], [self._id[c] for c in hole])


class _PokerkitEval:
    name = "pokerkit"

    def __init__(self):
        from pokerkit import Card, StandardHighHand
        self._hand = StandardHighHand
        self._card = {c: next(iter(Card.parse(c))) for c in DECK}

    def score(self, hole, board):
        return self._hand.from_game([self._card[c] for c in hole], [self._card[c] for c in board])


def _make_evaluator():
    try:
        return _TreysEval()
    except ImportError:
        return _PokerkitEval()


EVALUATOR = _make_evaluator()


# --------------------------------------------------------------------------- range model
def normalise(code: str) -> str:
    code = code.strip()
    if len(code) == 3 and code[:2] == "10":
        code = "T" + code[2]
    if len(code) != 2:
        raise ValueError(f"Bad card code {code!r}")
    rank, suit = code[0].upper(), code[1].lower()
    card = rank + suit
    if card not in DECK:
        raise ValueError(f"Bad card code {code!r}")
    return card


def starting_hand_score(combo) -> float:
    """Heuristic range weight, NOT a win probability (unchanged from the demo)."""
    a, b = combo
    high, low = sorted((RANKS.index(a[0]) + 2, RANKS.index(b[0]) + 2), reverse=True)
    if high == low:
        return min(1.0, 0.48 + high / 28)
    score = (high + low) / 34
    score += 0.06 if a[1] == b[1] else 0
    score += 0.04 if high - low == 1 else 0
    score -= 0.06 if high - low >= 5 else 0
    return max(0.0, min(1.0, score))


def range_weight(combo, pressure: int) -> float:
    if pressure == 0:
        return 1.0
    score = starting_hand_score(combo)
    if pressure == 1:
        return 1.0 + (3.0 if score >= 0.62 else 0) + (3.0 if score >= 0.80 else 0)
    return 1.0 + (3.0 if score >= 0.70 else 0) + (6.0 if score >= 0.86 else 0)


def opponent_pressures(observation, model: str):
    """Per active opponent: 0..2 depending on their public preflop raises."""
    if model == "uniform":
        return tuple(0 for _ in observation.active_opponents)
    if model != "public_actions":
        raise ValueError("Unknown opponent range model")
    out = []
    for opp in observation.active_opponents:
        raises = sum(a.stage == "preflop" and a.action == "raise" and a.player_index == opp.seat
                     for a in observation.public_history)
        out.append(min(2, raises))
    return tuple(out)


# --------------------------------------------------------------------------- estimation
def estimate_equity(hole_cards: Sequence[str], board_cards: Sequence[str] = (),
                    sample_count: int = 600, pressures: Sequence[int] = (0,), seed=None) -> EquityResult:
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or not 50 <= sample_count <= 5000:
        raise ValueError("Sample count must be an integer between 50 and 5000")
    pressures = tuple(sorted(int(p) for p in pressures))
    if not pressures or len(pressures) > 9 or any(p not in (0, 1, 2) for p in pressures):
        raise ValueError("Need 1..9 opponents with pressure 0, 1 or 2")
    hole = tuple(sorted(normalise(c) for c in hole_cards))
    board = tuple(sorted(normalise(c) for c in board_cards))
    return _estimate(hole, board, sample_count, pressures, seed)


@lru_cache(maxsize=256)
def _combo_table(unknown, pressure):
    combos = tuple(itertools.combinations(unknown, 2))
    weights = tuple(range_weight(c, pressure) for c in combos)
    cum = tuple(itertools.accumulate(weights))
    return combos, weights, cum


@lru_cache(maxsize=256)
def _estimate(hole, board, samples, pressures, seed) -> EquityResult:
    if len(hole) != 2 or len(board) not in (0, 3, 4, 5):
        raise ValueError("Expected two hole cards and 0, 3, 4, or 5 board cards")
    known = hole + board
    if len(set(known)) != len(known):
        raise ValueError("Duplicate cards in the hole cards or board")
    unknown = tuple(c for c in DECK if c not in known)
    n_opp = len(pressures)
    model = "uniform" if not any(pressures) else "preflop_raise_weighted_" + "".join(map(str, pressures))
    if seed is None:
        payload = json.dumps((hole, board, samples, pressures)).encode()
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    rng = random.Random(seed)
    score = EVALUATOR.score

    wins = ties = total = square_sum = 0.0
    trials = 0

    if n_opp == 1 and len(board) == 5:
        # Exact heads-up river enumeration (as in the demo).
        combos, weights, _ = _combo_table(unknown, pressures[0])
        hero = score(hole, board)
        for opp, w in zip(combos, weights):
            villain = score(opp, board)
            s = 1.0 if hero > villain else 0.5 if hero == villain else 0.0
            wins += w * (hero > villain)
            ties += w * (hero == villain)
            total += w
            square_sum += w * s * s
            trials += 1
        eq = (wins + ties / 2) / total
        return EquityResult(eq, wins / total, ties / total, 0.0, trials, "river_exact", model, 1)

    tables = [_combo_table(unknown, p) for p in pressures]
    need_board = 5 - len(board)
    for _ in range(samples):
        used = set()
        hands = []
        for combos, _weights, cum in tables:
            for _attempt in range(50):
                combo = rng.choices(combos, cum_weights=cum, k=1)[0]
                if combo[0] not in used and combo[1] not in used:
                    break
            else:  # pathological: fall back to uniform choice among remaining cards
                remaining = [c for c in unknown if c not in used]
                combo = tuple(rng.sample(remaining, 2))
            used.update(combo)
            hands.append(combo)
        remaining = [c for c in unknown if c not in used]
        full_board = board + tuple(rng.sample(remaining, need_board))
        hero = score(hole, full_board)
        villains = [score(h, full_board) for h in hands]
        best_villain = max(villains)
        if hero > best_villain:
            share, win, tie = 1.0, 1, 0
        elif hero == best_villain:
            tied = 1 + sum(v == hero for v in villains)
            share, win, tie = 1.0 / tied, 0, 1
        else:
            share, win, tie = 0.0, 0, 0
        wins += win
        ties += tie
        total += share
        square_sum += share * share
        trials += 1

    eq = total / trials
    variance = max(0.0, square_sum / trials - eq * eq)
    err = math.sqrt(variance / max(1, trials - 1))
    return EquityResult(eq, wins / trials, ties / trials, err, trials, "monte_carlo", model, n_opp)
