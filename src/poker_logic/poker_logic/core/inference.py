"""Turn a "turn done" press + pot reading into a legal poker action.

The engine knows whose turn it is and how much they owe, so one button press
(or any "turn finished" signal) is enough:

=====================================  =================================
chips added to the pot (snapped)       inferred action
=====================================  =================================
0, nothing owed                        check
0, chips owed                          fold
== amount owed (or all of the stack)   call
> amount owed                          bet / raise to (street bet + delta)
=====================================  =================================

Readings are snapped to the smallest chip and compared with a tolerance so a
noisy pot estimate does not flip a call into a raise. Anything that cannot be
resolved returns ``action=None`` with a message for the operator.
"""
from dataclasses import dataclass
from typing import Optional

from .observation import LegalActions


@dataclass(frozen=True)
class Inferred:
    action: Optional[str]          # fold / check / call / raise / None (unresolved)
    raise_to: Optional[int] = None
    delta: int = 0                 # chips the player added (snapped)
    confident: bool = True
    message: str = ""


def snap(value: float, unit: int) -> int:
    unit = max(1, int(unit))
    return int(round(value / unit) * unit)


def infer_action(legal: LegalActions, street_bet_before: int, stack_before: int,
                 pot_before: float, pot_now: float, chip_unit: int = 5,
                 tolerance: Optional[float] = None, fold_signal: bool = False) -> Inferred:
    tol = chip_unit / 2 if tolerance is None else float(tolerance)
    raw = float(pot_now) - float(pot_before)

    if fold_signal:
        if legal.can_fold:
            note = "" if abs(raw) <= tol else f" (pot moved by {raw:+.0f}; check the chips)"
            return Inferred("fold", delta=0, confident=abs(raw) <= tol, message="Fold signalled" + note)
        return Inferred("check", delta=0, confident=False,
                        message="Fold signalled but nothing is owed; treated as check")

    if raw < -tol:
        return Inferred(None, delta=int(raw), confident=False,
                        message=f"Pot shrank by {-raw:.0f}; recount the pot and press again")

    delta = max(0, snap(raw, chip_unit))
    off_grid = abs(raw - delta) > tol
    call = legal.call_amount

    if delta == 0:
        if legal.can_check:
            return Inferred("check", delta=0)
        return Inferred("fold", delta=0, message="Owed chips but added none: fold")

    # Short all-in: everything the player had, even if below the call price.
    if delta >= stack_before:
        delta = stack_before
        if legal.can_raise and street_bet_before + delta > street_bet_before + call:
            return Inferred("raise", raise_to=street_bet_before + delta, delta=delta,
                            confident=not off_grid, message="All-in")
        return Inferred("call", delta=delta, confident=not off_grid, message="All-in call")

    if call and abs(delta - call) <= max(tol, 0):
        return Inferred("call", delta=delta, confident=not off_grid)

    if delta < call:
        return Inferred(None, delta=delta, confident=False,
                        message=f"Added {delta} but {call} is needed to call; add chips or fold")

    # delta > call: a bet or raise
    target = street_bet_before + delta
    if not legal.can_raise:
        return Inferred("call", delta=call, confident=False,
                        message=f"Raising is not possible; treated as call, return {delta - call} chips")
    low, high = legal.min_raise_to, legal.max_raise_to
    if target > high:
        if target - high <= tol:
            return Inferred("raise", raise_to=high, delta=high - street_bet_before, confident=False)
        return Inferred("raise", raise_to=high, delta=high - street_bet_before, confident=False,
                        message=f"More than the maximum; treated as all-in to {high}, return {target - high}")
    if target < low:
        # Live-poker rule of thumb: under half a raise increment is a call.
        increment = low - street_bet_before - call
        extra = delta - call
        if extra * 2 < increment:
            return Inferred("call", delta=call, confident=False,
                            message=f"Under-raise; treated as call, return {extra} chips")
        return Inferred("raise", raise_to=low, delta=low - street_bet_before, confident=False,
                        message=f"Under-raise; treated as min raise to {low}, add {low - target} chips")
    return Inferred("raise", raise_to=target, delta=delta, confident=not off_grid)
