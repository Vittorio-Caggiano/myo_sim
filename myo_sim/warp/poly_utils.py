"""Small standalone helper for multivariate-polynomial muscle paths (OpenSim/Bolt-compatible term ordering)."""

from __future__ import annotations


def poly_exponents(dim: int, order: int) -> list[tuple[int, ...]]:
    """Enumerate exponent tuples for a degree-`order` polynomial in `dim` variables, in OpenSim/Bolt's own
    term order (recursive, first coordinate's exponent ascending fastest)."""

    def rec(d, o):
        if d == 0:
            yield ()
            return
        for i in range(o + 1):
            for tail in rec(d - 1, o - i):
                yield (i,) + tail

    return list(rec(dim, order))
