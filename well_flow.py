"""
Well-filling simulator.

Model
-----
- The well is a grid of 1 ft x 1 ft sections. depth[r][c] is how deep a section is
  (bigger = deeper). Water flows toward deeper neighbours.
- Water poured into `source` at `rate` cu ft/min passes instantly through every section
  it reaches: a section splits its water evenly across each orthogonally adjacent section
  whose water surface is lower than its own floor.
- A section with no lower neighbour is a basin. Water collects in it (a "pool") and the
  level rises. When the level reaches the lowest point of the pool's rim (the boundary
  section with the greatest depth), the pool spills over that rim and passes its inflow on.
  Pools that rise to the same level merge.

Algorithm (event-driven, exact arithmetic with Fraction)
--------------------------------------------------------
1. Every local minimum of elevation (elevation = -depth) starts as a pool.
2. Settle any pool that has reached its rim: absorb flat rim sections, merge with a
   neighbouring pool at the same level, or start spilling.
3. Route the source's flow downhill to find the inflow rate of every pool.
4. Stop if the target has received water. Otherwise advance time to the next event
   (the first non-spilling pool reaching its rim), add volume, and repeat.

Usage
-----
    python well_flow.py            # solves the 7x7 puzzle in this file
To solve another grid, call simulate_pour(depth, source, target).
If "lower depth" should mean shallower instead, pass negated depths.
"""

from collections import defaultdict
from fractions import Fraction


class Pool:
    def __init__(self, cells, volume=Fraction(0)):
        self.cells = set(cells)
        self.volume = volume
        self.full = False   # True while spilling over its rim
        self.spill = []     # rim sections it spills over (only when full)
        self.alive = True


def simulate_pour(depth, source, target, rate=1, verbose=False):
    """Return the time (minutes, as a Fraction) water first reaches `target`,
    or None if it never does."""
    rows, cols = len(depth), len(depth[0])
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    elev = {(r, c): -Fraction(depth[r][c]) for r, c in cells}
    nbrs = {
        (r, c): [(r + dr, c + dc)
                 for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                 if 0 <= r + dr < rows and 0 <= c + dc < cols]
        for r, c in cells
    }

    pools = []
    owner = {}  # section -> Pool, only for submerged sections

    def log(msg):
        if verbose:
            print(msg)

    def add_pool(cell_set, volume):
        p = Pool(cell_set, volume)
        pools.append(p)
        for c in p.cells:
            owner[c] = p
        return p

    def level(p):
        # all sections of a pool lie below its level, so volume fixes the level exactly
        return (p.volume + sum(elev[c] for c in p.cells)) / len(p.cells)

    def boundary(p):
        return {y for c in p.cells for y in nbrs[c] if y not in p.cells}

    def rim(p):
        b = boundary(p)
        return min(elev[y] for y in b) if b else None

    def room(p):
        """Volume still needed before the pool reaches its rim (None if it has no rim)."""
        m = rim(p)
        if m is None:
            return None
        return sum(m - elev[c] for c in p.cells) - p.volume

    def surface(y):
        p = owner.get(y)
        return level(p) if p else elev[y]

    def merge(p, q):
        p.cells |= q.cells
        p.volume += q.volume
        p.full, p.spill = False, []
        q.alive = False
        pools.remove(q)
        for c in q.cells:
            owner[c] = p
        log(f"  merge pools -> {len(p.cells)} sections at level {float(level(p)):.2f}")

    def has_lower_exit(y, p):
        """Does rim section y drain somewhere other than pool p?"""
        return any(surface(z) < elev[y] for z in nbrs[y] if z not in p.cells)

    def settle(p):
        """p has reached its rim level m: absorb, merge, or start spilling."""
        while True:
            m = level(p)
            other = next((owner[y] for y in boundary(p)
                          if y in owner and owner[y] is not p and level(owner[y]) == m),
                         None)
            if other is not None:
                merge(p, other)
                continue
            rim_cells = [y for y in boundary(p) if y not in owner and elev[y] == m]
            if not rim_cells:
                return
            spill = [y for y in rim_cells if has_lower_exit(y, p)]
            flat = [y for y in rim_cells if y not in spill]
            if flat:
                for y in flat:
                    p.cells.add(y)
                    owner[y] = p
                continue
            p.full, p.spill = True, spill
            log(f"  pool spills at depth {float(-m):g} over {spill}")
            return

    def route(src, amount):
        """Push `amount` into `src`; return (inflow per pool, set of touched pools/sections)."""
        inflow = defaultdict(Fraction)
        touched = set()

        def deliver(cell, amt):
            p = owner.get(cell)
            touched.add(p if p else cell)
            if p is None:
                drop(cell, amt)
            elif not p.full:
                inflow[p] += amt
            else:
                for s in p.spill:
                    drop(s, amt / len(p.spill))

        def drop(cell, amt):
            """Dry section: split evenly to every neighbour whose surface is lower."""
            outs = [z for z in nbrs[cell] if surface(z) < elev[cell]]
            if not outs:
                raise RuntimeError(f"water has nowhere to go from {cell}")
            for z in outs:
                deliver(z, amt / len(outs))

        deliver(src, amount)
        return inflow, touched

    # Step 1: every local minimum starts as a pool.
    for c in cells:
        if all(elev[y] >= elev[c] for y in nbrs[c]):
            add_pool({c}, Fraction(0))

    t = Fraction(0)
    while True:
        # Step 2: settle pools that have reached their rim.
        progress = True
        while progress:
            progress = False
            for p in pools:
                if not p.full and room(p) == 0:
                    log(f"t={float(t):.4f}: pool reached rim")
                    settle(p)
                    progress = True
                    break

        # Step 3: route flow.
        inflow, touched = route(source, Fraction(rate))
        if (owner.get(target) or target) in touched:
            return t

        # Step 4: advance to the next event.
        dt = None
        for p in pools:
            if not p.full and inflow[p] > 0 and room(p) is not None:
                d = room(p) / inflow[p]
                if dt is None or d < dt:
                    dt = d
        if dt is None:
            return None
        for p in pools:
            if not p.full:
                p.volume += inflow[p] * dt
        t += dt


if __name__ == "__main__":
    # Puzzle grid, row 1 at the top. Section numbers are depths in feet.
    WELL = [
        [1, 46, 9, 19, 45, 26, 18],
        [44, 30, 37, 5, 22, 40, 23],
        [31, 33, 16, 13, 43, 20, 41],
        [14, 12, 36, 2, 28, 38, 42],
        [49, 48, 17, 47, 11, 29, 34],
        [4, 32, 7, 8, 25, 15, 3],
        [21, 27, 6, 39, 10, 24, 35],
    ]
    # Water is poured above section 1 (row 0, col 0); watch section 35 (row 6, col 6).
    result = simulate_pour(WELL, source=(0, 0), target=(6, 6), rate=1)
    print("Minutes until water accumulates in section 35:", result)
