"""Break a chip amount into physical chips the arm can pick from its trays."""
from typing import Dict, List, Optional


def make_change(amount: int, denominations: List[int],
                inventory: Optional[Dict[int, int]] = None) -> Dict[int, int]:
    """Fewest chips paying exactly ``amount`` (largest first, inventory-aware).

    Uses a small DP so odd denomination sets (e.g. 5/25/100 with a short tray)
    still work. Raises ``ValueError`` if the amount cannot be paid exactly.
    """
    if amount < 0:
        raise ValueError("Negative amount")
    if amount == 0:
        return {}
    denoms = sorted({int(d) for d in denominations if d > 0}, reverse=True)
    inv = {d: (inventory.get(d, 0) if inventory is not None else amount // d + 1) for d in denoms}
    # best[a] = (chip count, {denom: n}) for exact amount a
    best = {0: (0, {})}
    for d in denoms:
        nxt = dict(best)
        for a, (count, combo) in best.items():
            for k in range(1, inv[d] + 1):
                tot = a + k * d
                if tot > amount:
                    break
                cand = (count + k, {**combo, d: k})
                if tot not in nxt or cand[0] < nxt[tot][0]:
                    nxt[tot] = cand
        best = nxt
    if amount not in best:
        raise ValueError(f"Cannot pay {amount} exactly with {denoms} and inventory {inv}")
    return dict(sorted(best[amount][1].items(), reverse=True))
