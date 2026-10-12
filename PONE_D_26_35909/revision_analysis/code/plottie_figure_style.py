#!/usr/bin/env python3
"""Plottie-inspired shared style for the public-only submission figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.transforms import ScaledTranslation
import numpy as np


INK = "#252A2E"
MUTED = "#687177"
GRID = "#E3E7E9"
PALE = "#F2F5F6"
WHITE = "#FFFFFF"

TEAL = "#2D7F82"
TEAL_LIGHT = "#8FC7C5"
CORAL = "#D86F55"
CORAL_LIGHT = "#F0B39F"
BLUE = "#4B78A8"
BLUE_LIGHT = "#AFC7DD"
GOLD = "#C99A3D"
PURPLE = "#7B6A9D"
GREEN = "#5C8D72"
GRAY = "#AAB1B5"
GRAY_DARK = "#70787D"

COHORT_COLORS = {
    "GSE162631": PURPLE,
    "GSE211376": TEAL,
    "GSE273274": CORAL,
    "GSE274987": GOLD,
    "GSE276177": BLUE,
    "GSE286419": GREEN,
}

STATE_COLORS = {
    "AC-like": CORAL,
    "Other malignant": BLUE,
    "Myeloid": GREEN,
}

REGION_COLORS = {
    "Tumor": CORAL,
    "Core": CORAL,
    "Edge": GOLD,
    "Peritumoral": TEAL,
    "Peripheral": TEAL,
}

DIVERGING = LinearSegmentedColormap.from_list(
    "plottie_diverging",
    [BLUE, BLUE_LIGHT, WHITE, CORAL_LIGHT, CORAL],
)
TEAL_SEQ = LinearSegmentedColormap.from_list(
    "plottie_teal",
    ["#F0F5F5", TEAL_LIGHT, TEAL],
)
CORAL_SEQ = LinearSegmentedColormap.from_list(
    "plottie_coral",
    ["#F7F2EF", CORAL_LIGHT, CORAL],
)
PURPLE_SEQ = LinearSegmentedColormap.from_list(
    "plottie_purple",
    ["#F3F1F6", "#BFB5D2", PURPLE],
)


def configure_plottie_style() -> None:
    """Apply one restrained visual system to every manuscript figure."""
    mpl.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "font.weight": "normal",
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.5,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.65,
            "axes.grid": False,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "legend.fontsize": 6.8,
            "legend.frameon": False,
            "lines.linewidth": 1.0,
            "lines.markersize": 4.2,
            "patch.linewidth": 0.65,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.dpi": 600,
            "savefig.bbox": None,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    """Build one aligned panel header from the letter and existing title."""
    title = ax.get_title(loc="left")
    title_artist = getattr(ax, "_left_title", ax.title)
    title_size = title_artist.get_fontsize()
    title_weight = title_artist.get_fontweight()
    ax.set_title("", loc="left")

    elements = [
        TextArea(
            label,
            textprops={
                "fontsize": 10.0,
                "fontweight": "bold",
                "color": INK,
            },
        )
    ]
    if title:
        elements.append(
            TextArea(
                title,
                textprops={
                    "fontsize": title_size,
                    "fontweight": title_weight,
                    "color": INK,
                    "linespacing": 1.0,
                },
            )
        )

    header = HPacker(children=elements, align="top", pad=0, sep=5)
    transform = ax.transAxes + ScaledTranslation(
        0,
        5 / 72,
        ax.figure.dpi_scale_trans,
    )
    anchored = AnchoredOffsetbox(
        loc="lower left",
        child=header,
        frameon=False,
        bbox_to_anchor=(0, 1),
        bbox_transform=transform,
        borderpad=0,
        pad=0,
    )
    ax.add_artist(anchored)


def open_axes(ax: plt.Axes, grid_axis: str | None = None) -> None:
    """Use thin open axes with an optional light one-direction grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.65, length=2.6)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, zorder=0)


def clean_axes(ax: plt.Axes) -> None:
    """Remove the frame while retaining labels and ticks."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def median_bar(
    ax: plt.Axes,
    center: float,
    value: float,
    width: float = 0.34,
    color: str = INK,
    linewidth: float = 2.0,
) -> None:
    """Draw a compact Plottie-style median bar."""
    ax.plot(
        [center - width / 2, center + width / 2],
        [value, value],
        color=color,
        linewidth=linewidth,
        solid_capstyle="round",
        zorder=4,
    )


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    """Export fixed-canvas PNG and PDF without post-hoc tight cropping."""
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=600, facecolor=WHITE)
    fig.savefig(directory / f"{stem}.pdf", facecolor=WHITE)
    plt.close(fig)


def add_spatial_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    x,
    y,
    values,
    *,
    cmap,
    title: str,
    point_size: float = 3.8,
):
    """Draw a compact spatial panel with clipped robust color limits."""
    finite = mpl.cbook.safe_masked_invalid(values)
    valid = finite.compressed()
    vmin = float(valid.min()) if len(valid) else 0.0
    vmax = float(valid.max()) if len(valid) else 1.0
    if len(valid) > 10:
        vmin = float(np.quantile(valid, 0.01))
        vmax = float(np.quantile(valid, 0.99))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0
    scatter = ax.scatter(
        x,
        y,
        c=values,
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        rasterized=True,
    )
    ax.set_aspect("equal")
    ax.set_title(title, loc="left", pad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    clean_axes(ax)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.040, pad=0.018)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=5.8, width=0.5, length=2)
    return scatter
