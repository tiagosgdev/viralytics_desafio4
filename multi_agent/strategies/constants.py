"""
Shared lookup tables for the scorer strategies.

These were previously defined inline inside the SPADE agents. They live here
so every personality of a given agent can share the same compatibility /
adjacency knowledge without duplicating it. The tables themselves are *data*
(not tunable params): the personalities differ in how they *reward* the buckets
these tables define, not in the buckets themselves.
"""

from __future__ import annotations

# ── Colour compatibility matrix ───────────────────────────────────────────────
# Symmetric: if A is compatible with B, B is compatible with A.
COLOUR_COMPATIBLE: dict[str, frozenset[str]] = {
    "black":     frozenset({"white", "gray", "red", "blue", "yellow", "green", "purple", "orange", "beige", "cream", "pink"}),
    "white":     frozenset({"black", "navy", "blue", "red", "green", "gray", "pink", "beige", "burgundy", "olive"}),
    "gray":      frozenset({"white", "black", "navy", "blue", "red", "pink", "purple", "yellow"}),
    "navy":      frozenset({"white", "beige", "gray", "red", "yellow", "cream", "burgundy"}),
    "blue":      frozenset({"white", "gray", "beige", "navy", "brown", "yellow", "orange"}),
    "red":       frozenset({"white", "black", "gray", "navy", "beige", "pink"}),
    "green":     frozenset({"white", "beige", "brown", "navy", "black", "cream", "olive"}),
    "yellow":    frozenset({"white", "black", "navy", "gray", "blue", "brown"}),
    "orange":    frozenset({"white", "black", "navy", "brown", "blue"}),
    "pink":      frozenset({"white", "gray", "navy", "black", "beige", "cream"}),
    "purple":    frozenset({"white", "black", "gray", "beige", "pink"}),
    "brown":     frozenset({"white", "beige", "green", "navy", "cream", "yellow", "orange"}),
    "beige":     frozenset({"white", "brown", "navy", "blue", "green", "black", "pink"}),
    "cream":     frozenset({"brown", "navy", "black", "beige", "green", "burgundy"}),
    "burgundy":  frozenset({"white", "beige", "gray", "black", "navy", "cream"}),
    "olive":     frozenset({"white", "beige", "brown", "black", "cream"}),
    "teal":      frozenset({"white", "beige", "gray", "navy", "black", "coral"}),
    "coral":     frozenset({"white", "beige", "navy", "gray", "teal"}),
    "multicolor": frozenset({"black", "white", "navy", "gray"}),
}

# ── Body-shape adjacency graph ────────────────────────────────────────────────
# Shapes considered "close enough" to get partial credit.
# Symmetric: if A is adjacent to B then B is adjacent to A.
BODY_ADJACENT: dict[str, frozenset[str]] = {
    "hourglass":         frozenset({"pear", "rectangle"}),
    "pear":              frozenset({"hourglass", "triangle"}),
    "triangle":          frozenset({"pear", "rectangle"}),
    "rectangle":         frozenset({"hourglass", "triangle", "trapezoid", "inverted_triangle"}),
    "inverted_triangle": frozenset({"trapezoid", "rectangle"}),
    "apple":             frozenset({"rectangle", "oval"}),
    "trapezoid":         frozenset({"rectangle", "inverted_triangle"}),
    "oval":              frozenset({"apple", "rectangle"}),
}


def item_key(item_id: int | str, size: str) -> str:
    """Canonical per-candidate key shared by all strategies: ``f"{item_id}:{size}"``."""
    return f"{item_id}:{size}"
