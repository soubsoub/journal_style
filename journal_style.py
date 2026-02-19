"""
journal_style.py
================
Matplotlib style utilities for scientific journal figures.

Supported journals / presets
-----------------------------
    "nature"    : Nature, Nature Methods, Nature Communications
    "elsevier"  : Cell, Lancet, most Elsevier journals
    "acs"       : ACS Nano, JACS, Analytical Chemistry
    "ieee"      : IEEE Transactions, IEEE Access
    "plos"      : PLOS ONE, PLOS Biology

Usage
-----
    from journal_style import set_color_style, set_grayscale_style, reset_style, journal_style

    # Option A: direct call
    palette = set_color_style(journal="nature")
    fig, ax = plt.subplots(figsize=palette.figsize_single)

    # Option B: context manager (recommended for notebooks)
    with journal_style("nature", mode="color") as palette:
        fig, ax = plt.subplots(figsize=palette.figsize_single)
        ax.plot(x, y, color=palette.colors[0])

    # Option C: on top of SciencePlots (if installed)
    import matplotlib.pyplot as plt
    plt.style.use(["science", "nature"])
    palette = set_color_style(journal="nature")

Dependencies
------------
    Required : matplotlib
    Optional : SciencePlots  (pip install SciencePlots)
"""

from __future__ import annotations

import copy
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Journal specifications
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Column widths in inches (single, double) and max height per journal
_JOURNAL_SPECS: dict[str, dict] = {
    "nature":   {"single": 3.54, "double": 7.08, "max_height": 8.75},
    "elsevier": {"single": 3.54, "double": 7.48, "max_height": 9.45},
    "acs":      {"single": 3.33, "double": 7.00, "max_height": 9.17},
    "ieee":     {"single": 3.50, "double": 7.16, "max_height": 9.45},
    "plos":     {"single": 5.20, "double": 6.83, "max_height": 8.75},
}

_DEFAULT_ASPECT = 0.75  # height = width * aspect for default figsize


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Font availability check
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _resolve_fonts(preferred: list[str], fallback: str = "DejaVu Sans") -> list[str]:
    """
    Return preferred fonts that are actually installed, with fallback appended.
    Emits a warning if none of the preferred fonts are found.
    """
    available = {f.name for f in font_manager.fontManager.ttflist}
    resolved = [f for f in preferred if f in available]
    if not resolved:
        warnings.warn(
            f"None of the preferred fonts {preferred} found on this system. "
            f"Falling back to '{fallback}'.",
            UserWarning,
            stacklevel=3,
        )
    if fallback not in resolved:
        resolved.append(fallback)
    return resolved


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Shared base rcParams
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_BASE_RC: dict = {
    "pdf.fonttype": 42,       # embed fonts as Type 1 (required by most journals)
    "ps.fonttype":  42,
    "font.size":          7,
    "axes.labelsize":     7,
    "axes.titlesize":     7,
    "xtick.labelsize":    6,
    "ytick.labelsize":    6,
    "legend.fontsize":    6,
    "axes.linewidth":     0.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "lines.linewidth":    1.2,
    "lines.markersize":   4,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "axes.grid":          False,
    "legend.frameon":     False,
    "figure.dpi":         300,
    "savefig.dpi":        600,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Palette dataclass (replaces plain dict)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class Palette:
    """
    Structured container returned by set_color_style() and set_grayscale_style().

    Attributes
    ----------
    colors       : list of hex/named color strings
    linestyles   : list of linestyle strings (e.g. '-', '--')
    markers      : list of marker strings    (e.g. 'o', 's')
    hatches      : list of hatch patterns    (e.g. '/', 'x')
    fill_alpha   : default alpha for shaded/fill_between areas
    figsize_single : (width, height) tuple for single-column figures
    figsize_double : (width, height) tuple for double-column figures
    journal      : journal preset name used to build this palette
    mode         : 'color' or 'grayscale'
    """
    colors:         list[str]
    linestyles:     list[str]    = field(default_factory=lambda: ["-", "--", ":", "-."])
    markers:        list[str]    = field(default_factory=lambda: ["o", "s", "^", "D"])
    hatches:        list[str]    = field(default_factory=lambda: ["/", "\\", "x", "."])
    fill_alpha:     float        = 0.20
    figsize_single: tuple        = (3.54, 2.66)
    figsize_double: tuple        = (7.08, 5.31)
    journal:        str          = "nature"
    mode:           str          = "color"

    def cycle(self, n: int | None = None):
        """
        Yield (color, linestyle, marker) tuples â€” convenience for multi-series plots.

        Parameters
        ----------
        n : int, optional
            Number of series. If None, cycles infinitely (use with zip()).

        Example
        -------
            for i, (col, ls, mk) in enumerate(palette.cycle(3)):
                ax.plot(x, y[i], color=col, linestyle=ls, marker=mk)
        """
        from itertools import cycle, islice
        triples = zip(cycle(self.colors), cycle(self.linestyles), cycle(self.markers))
        if n is not None:
            triples = islice(triples, n)
        return triples


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5. Internal builder
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _build_rc(extra: dict | None = None) -> dict:
    """Return a deep copy of _BASE_RC merged with any extra keys."""
    rc = copy.deepcopy(_BASE_RC)
    fonts = _resolve_fonts(["Arial", "Helvetica"])
    rc["font.family"]     = "sans-serif"
    rc["font.sans-serif"] = fonts
    if extra:
        rc.update(extra)
    return rc


def _get_figsize(journal: str, aspect: float = _DEFAULT_ASPECT) -> tuple[tuple, tuple]:
    """Return (figsize_single, figsize_double) for the given journal."""
    spec = _JOURNAL_SPECS.get(journal, _JOURNAL_SPECS["nature"])
    ws, wd = spec["single"], spec["double"]
    return (ws, ws * aspect), (wd, wd * aspect)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6. Public style functions
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def set_color_style(
    journal: Literal["nature", "elsevier", "acs", "ieee", "plos"] = "nature",
    colors: list[str] | None = None,
    aspect: float = _DEFAULT_ASPECT,
) -> Palette:
    """
    Apply Nature-compatible color rcParams and return a Palette.

    Parameters
    ----------
    journal : str
        Target journal preset for figure width specs.
    colors  : list[str], optional
        Override the default colorblind-safe palette.
    aspect  : float
        Height / width ratio for default figsize.

    Returns
    -------
    Palette
    """
    rc = _build_rc()
    mpl.rcParams.update(rc)

    palette_colors = colors or [
        "#4C72B0",  # blue
        "#DD8452",  # orange
        "#55A868",  # green
        "#C44E52",  # red
        "#8172B3",  # purple
        "#937860",  # brown
    ]
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=palette_colors)

    fs, fd = _get_figsize(journal, aspect)
    return Palette(
        colors=palette_colors,
        figsize_single=fs,
        figsize_double=fd,
        journal=journal,
        mode="color",
    )


def set_grayscale_style(
    journal: Literal["nature", "elsevier", "acs", "ieee", "plos"] = "nature",
    aspect: float = _DEFAULT_ASPECT,
) -> Palette:
    """
    Apply grayscale rcParams suitable for BW / print journals.

    Returns
    -------
    Palette
        Includes colors, linestyles, markers, and hatches for disambiguation.
    """
    rc = _build_rc({"image.cmap": "gray"})
    mpl.rcParams.update(rc)

    gs_colors = ["black", "dimgray", "gray", "darkgray"]
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=gs_colors)

    fs, fd = _get_figsize(journal, aspect)
    return Palette(
        colors=gs_colors,
        linestyles=["-", "--", ":", "-."],
        markers=["o", "s", "^", "D"],
        hatches=["/", "\\", "x", "."],
        fill_alpha=0.35,     # higher alpha needed in grayscale
        figsize_single=fs,
        figsize_double=fd,
        journal=journal,
        mode="grayscale",
    )


def reset_style() -> None:
    """
    Restore matplotlib's default rcParams.
    Useful when mixing styles in the same session or notebook.
    """
    mpl.rcParams.update(mpl.rcParamsDefault)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 7. Context manager
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@contextmanager
def journal_style(
    journal: Literal["nature", "elsevier", "acs", "ieee", "plos"] = "nature",
    mode: Literal["color", "grayscale"] = "color",
    colors: list[str] | None = None,
    aspect: float = _DEFAULT_ASPECT,
) -> Generator[Palette, None, None]:
    """
    Context manager: apply journal style locally and restore defaults on exit.
    Prevents global rcParams pollution (recommended for Jupyter notebooks).

    Parameters
    ----------
    journal : str
        Target journal preset.
    mode    : 'color' or 'grayscale'
    colors  : list[str], optional
        Custom color list (only applied in color mode).
    aspect  : float
        Height / width ratio for default figsize.

    Yields
    ------
    Palette

    Example
    -------
        with journal_style("nature", mode="grayscale") as palette:
            fig, ax = plt.subplots(figsize=palette.figsize_single)
            for col, ls, mk in palette.cycle(3):
                ax.plot(x, y, color=col, linestyle=ls, marker=mk)
            fig.savefig("figure1.pdf")
    """
    prev_rc = copy.deepcopy(dict(mpl.rcParams))
    try:
        if mode == "color":
            palette = set_color_style(journal=journal, colors=colors, aspect=aspect)
        else:
            palette = set_grayscale_style(journal=journal, aspect=aspect)
        yield palette
    finally:
        mpl.rcParams.update(prev_rc)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 8. Quick-check / demo  (run as script)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    import numpy as np

    x = np.linspace(0, 2 * np.pi, 100)

    # --- Color demo ---
    with journal_style("nature", mode="color") as p:
        fig, ax = plt.subplots(figsize=p.figsize_single)
        for i, (col, ls, mk) in enumerate(p.cycle(3)):
            ax.plot(x, np.sin(x + i * 0.5), color=col, linestyle=ls,
                    marker=mk, markevery=20, label=f"Series {i + 1}")
        ax.set_xlabel("x (rad)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Color â€” Nature single column")
        ax.legend()
        fig.savefig("demo_color.pdf")
        print(f"Saved demo_color.pdf  |  figsize={p.figsize_single}")

    # --- Grayscale demo ---
    with journal_style("nature", mode="grayscale") as p:
        fig, ax = plt.subplots(figsize=p.figsize_single)
        for i, (col, ls, mk) in enumerate(p.cycle(3)):
            ax.plot(x, np.sin(x + i * 0.5), color=col, linestyle=ls,
                    marker=mk, markevery=20, label=f"Series {i + 1}")
        ax.set_xlabel("x (rad)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Grayscale â€” Nature single column")
        ax.legend()
        fig.savefig("demo_grayscale.pdf")
        print(f"Saved demo_grayscale.pdf  |  figsize={p.figsize_single}")

    print("\nJournal presets available:")
    for name, spec in _JOURNAL_SPECS.items():
        print(f"  {name:<10}  single={spec['single']}in  double={spec['double']}in")
