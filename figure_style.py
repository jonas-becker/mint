"""figure_style.py

Reusable, project-level figure styling helpers:
- Curated categorical palettes (2/3/4/5/6/8)
- Helpers for picking palettes per figure
- Sequential/diverging colormaps derived from the same palette family

This file is intentionally self-contained so it can be copied to other projects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# Categorical palettes (provided by user)
# ---------------------------------------------------------------------------

PALETTES: Dict[int, List[str]] = {
    2: ["#5e4c5f", "#ffbb6f"],
    3: ["#5e4c5f", "#999999", "#ffbb6f"],
    4: ["#5e4c5f", "#6f8fa6", "#999999", "#ffbb6f"],
    5: ["#5e4c5f", "#6f8fa6", "#7fa37a", "#999999", "#ffbb6f"],
    6: ["#5e4c5f", "#6f8fa6", "#7fa37a", "#c27a7a", "#999999", "#ffbb6f"],
    8: ["#5e4c5f", "#6f8fa6", "#7fa37a", "#c27a7a", "#a88fbf", "#d6a0c4", "#999999", "#ffbb6f"],
}


def get_palette(n: int) -> List[str]:
    """Return a categorical palette of size n (2/3/4/5/6/8)."""
    if n not in PALETTES:
        raise ValueError(f"Unsupported palette size {n}. Supported: {sorted(PALETTES)}")
    return list(PALETTES[n])


def cycle_palette(n: int, k: int) -> List[str]:
    """Return k colors by cycling through the n-color palette."""
    base = get_palette(n)
    if k <= 0:
        return []
    return [base[i % len(base)] for i in range(k)]


def categorical_map(labels: Sequence[str], palette_n: int) -> Dict[str, str]:
    """Assign colors to labels in-order using a palette."""
    cols = cycle_palette(palette_n, len(labels))
    return {lab: col for lab, col in zip(labels, cols)}


# ---------------------------------------------------------------------------
# Colormaps derived from the same family
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Colormaps:
    sequential: LinearSegmentedColormap
    diverging: LinearSegmentedColormap


def get_colormaps() -> Colormaps:
    """Return standard sequential + diverging colormaps for heatmaps.

    - sequential: dull purple -> neutral gray -> soft gold
    - diverging: muted rose (neg) -> neutral gray -> sage green (pos)
    """
    pal3 = get_palette(3)
    # purple -> gray -> gold
    sequential = LinearSegmentedColormap.from_list("proj_seq", [pal3[0], pal3[1], pal3[2]], N=256)

    pal6 = get_palette(6)
    # rose -> gray -> sage
    diverging = LinearSegmentedColormap.from_list("proj_div", [pal6[3], pal6[4], pal6[2]], N=256)
    return Colormaps(sequential=sequential, diverging=diverging)


def sample_cmap(cmap: LinearSegmentedColormap, n: int) -> List[str]:
    """Sample n colors evenly from a colormap."""
    if n <= 1:
        return ["#999999"]
    return [cmap(i / (n - 1)) for i in range(n)]

