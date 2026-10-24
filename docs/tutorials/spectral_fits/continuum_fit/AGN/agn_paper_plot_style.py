"""Shared presentation style for the DC4 AGN notebook figures.

The colors and typography follow the Paper_Plots_DC4/Fluctuate_True notebooks.
Use ``paper_style`` around figure creation, then ``style_axis`` for each panel.
Both functions leave the numerical data and fit results untouched.
"""

from contextlib import contextmanager

import matplotlib.pyplot as plt


ORANGE = "#D55E00"
BLUE = "#2A8BC3"
GREEN = "#009E73"
PURPLE = "#CC79A7"
LIGHT_FACE = "white"
DARK_FACE = "#0B0B0B"


def ink(theme):
    """High-contrast color for observations and injected reference curves."""
    if theme not in {"light", "dark"}:
        raise ValueError("theme must be 'light' or 'dark'")
    return "#F5F5F5" if theme == "dark" else "#161616"


@contextmanager
def paper_style(theme="light", font_size=25):
    """Apply the paper typography to one plot without changing global rcParams."""
    foreground = ink(theme)
    face = DARK_FACE if theme == "dark" else LIGHT_FACE
    with plt.rc_context({
        "agg.path.chunksize": 10000,
        "font.family": "Times New Roman",
        "font.size": font_size,
        "font.weight": 550,
        "axes.linewidth": 1.5,
        "axes.labelsize": font_size,
        "axes.labelweight": 550,
        "axes.titleweight": 550,
        "axes.edgecolor": foreground,
        "axes.labelcolor": foreground,
        "axes.facecolor": face,
        "figure.facecolor": face,
        "savefig.facecolor": face,
        "text.color": foreground,
        "xtick.color": foreground,
        "ytick.color": foreground,
        "legend.labelcolor": foreground,
    }):
        yield


def style_axis(axis, theme="light", font_size=25):
    """Boxed axes and four-sided inward ticks matching the paper figures."""
    foreground = ink(theme)
    axis.set_facecolor(DARK_FACE if theme == "dark" else LIGHT_FACE)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color(foreground)
    axis.tick_params(axis="both", which="major", direction="in", top=True,
                     right=True, length=12, width=1.5, pad=8,
                     labelsize=font_size, colors=foreground)
    axis.tick_params(axis="both", which="minor", direction="in", top=True,
                     right=True, length=6, width=1.5, colors=foreground)
    axis.xaxis.label.set_color(foreground)
    axis.yaxis.label.set_color(foreground)
    axis.title.set_color(foreground)
    return axis
