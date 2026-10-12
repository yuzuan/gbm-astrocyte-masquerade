#!/usr/bin/env python3
"""Redraw all submission figures with one Plottie-inspired visual system.

The rendering layer is intentionally separated from statistical computation.
It reads frozen result tables and representative spatial caches only.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns

from plottie_figure_style import (
    BLUE,
    BLUE_LIGHT,
    COHORT_COLORS,
    CORAL,
    CORAL_LIGHT,
    CORAL_SEQ,
    DIVERGING,
    GOLD,
    GRAY,
    GRAY_DARK,
    GREEN,
    GRID,
    INK,
    MUTED,
    PALE,
    PURPLE,
    PURPLE_SEQ,
    REGION_COLORS,
    STATE_COLORS,
    TEAL,
    TEAL_LIGHT,
    TEAL_SEQ,
    clean_axes,
    configure_plottie_style,
    median_bar,
    open_axes,
    panel_label,
    save_figure,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
EXTERNAL_RESULTS = ROOT / "external_validation" / "results"

COHORT_ORDER = [
    "GSE211376",
    "GSE273274",
    "GSE274987",
    "GSE276177",
    "GSE286419",
]
SENSITIVITY_ORDER = ["GSE162631", *COHORT_ORDER]
LIGANDS = ["TGFB2", "TGFB1", "ANXA1", "SPP1", "CSF1", "CCL2"]
RNG = np.random.default_rng(20260717)


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen figure inputs:\n" + "\n".join(missing))


def json_frame(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(json.loads(row["patient_values"]))


def short_region(region: str) -> str:
    return {
        "Tumor": "T",
        "Core": "C",
        "Edge": "E",
        "Peritumoral": "P",
        "Peripheral": "P",
    }.get(region, region)


def cohort_display(cohort: str, *, short: bool = False) -> str:
    """Use a transparent display label for the joint regional source cohort."""
    if cohort == "GSE274987":
        return "274986/7" if short else "GSE274986/7"
    return cohort.replace("GSE", "") if short else cohort


def add_strip(
    ax: plt.Axes,
    values,
    *,
    title: str,
    ylabel: str,
    color: str = TEAL,
    center: float = 0.0,
) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    jitter = RNG.normal(center, 0.035, len(values))
    ax.scatter(
        jitter,
        values,
        s=22,
        color=color,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    if len(values):
        median_bar(ax, center, float(np.median(values)), color=CORAL)
    ax.axhline(0, color=GRAY_DARK, linewidth=0.65, linestyle="--", zorder=1)
    ax.set_xlim(center - 0.27, center + 0.27)
    ax.set_xticks([])
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=5)
    ax.spines["bottom"].set_visible(False)
    open_axes(ax)


def add_spatial(
    fig: plt.Figure,
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    value: str,
    title: str,
    cmap,
    point_size: float = 3.3,
) -> None:
    values = pd.to_numeric(frame[value], errors="coerce").to_numpy(float)
    valid = np.isfinite(values)
    if valid.sum() == 0:
        raise ValueError(f"No finite values for spatial field {value}")
    vmin, vmax = np.quantile(values[valid], [0.01, 0.99])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    scatter = ax.scatter(
        frame[x],
        -frame[y],
        c=values,
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        rasterized=True,
    )
    ax.set_aspect("equal")
    ax.set_title(title, pad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    clean_axes(ax)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.038, pad=0.018)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=5.6, width=0.45, length=1.8)


def add_flow_node(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    width: float = 0.18,
    height: float = 0.23,
) -> None:
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.7,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=6.6, color=INK)


def figure1() -> None:
    manifest = (
        pd.read_csv(DATA / "cohort_manifest.csv")
        .set_index("cohort")
        .loc[COHORT_ORDER]
        .reset_index()
    )
    fig = plt.figure(figsize=(7.1, 4.35))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.0, 0.70],
        width_ratios=[1.32, 0.68],
        hspace=0.48,
        wspace=0.38,
    )

    ax = fig.add_subplot(gs[0, 0])
    regions = ["Tumor/core", "Edge", "Peritumoral"]
    yvals = np.arange(len(manifest))[::-1]
    for y, row in zip(yvals, manifest.itertuples()):
        present_set = set(str(row.regions).split("; "))
        present = [
            bool(present_set & {"Tumor", "Core"}),
            "Edge" in present_set,
            bool(present_set & {"Peritumoral", "Peripheral"}),
        ]
        active = np.flatnonzero(present)
        if len(active) > 1:
            ax.plot(active, np.repeat(y, len(active)), color=GRID, linewidth=1.4)
        for x, available in enumerate(present):
            ax.scatter(
                x,
                y,
                s=28 + row.patients * 4 if available else 20,
                facecolor=COHORT_COLORS[row.cohort] if available else "white",
                edgecolor=COHORT_COLORS[row.cohort] if available else GRAY,
                linewidth=0.7,
                zorder=2,
            )
        ax.text(2.75, y, f"n={row.patients}", va="center", fontsize=6.5, color=MUTED)
    ax.set_xticks(range(3), regions)
    ax.set_yticks(yvals, [cohort_display(item) for item in manifest["cohort"]])
    ax.set_xlim(-0.42, 3.15)
    ax.set_ylim(-0.6, len(manifest) - 0.4)
    ax.set_title("Public cohorts and sampled regions", pad=6)
    clean_axes(ax)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    for y, row in zip(yvals, manifest.itertuples()):
        ax.scatter(
            row.included_cells,
            y,
            s=34,
            color=COHORT_COLORS[row.cohort],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.text(
            row.included_cells * 1.08,
            y,
            f"{row.included_cells / 1000:.1f}k",
            va="center",
            fontsize=6.2,
            color=MUTED,
        )
    ax.set_xscale("log")
    ax.set_yticks(
        yvals, [cohort_display(item, short=True) for item in manifest["cohort"]]
    )
    ax.set_xlabel("Included cells or nuclei")
    ax.set_title("Analyzed cell set", pad=6)
    open_axes(ax, grid_axis="x")
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    lane_specs = [
        (
            0.70,
            ["AC-like vs other states", "Patient contrasts", "Cohort tests", "Aggregate association"],
            CORAL_LIGHT,
            CORAL,
        ),
        (
            0.25,
            ["Regional co-availability", "Paired patient effects", "Exact sign flip", "Statistically unresolved"],
            TEAL_LIGHT,
            TEAL,
        ),
    ]
    xs = [0.14, 0.38, 0.62, 0.86]
    for y, labels, pale, accent in lane_specs:
        for index, (x, label) in enumerate(zip(xs, labels)):
            add_flow_node(
                ax,
                x,
                y,
                label,
                facecolor=PALE if index not in {0, 3} else pale,
                edgecolor=accent if index in {0, 3} else GRAY,
            )
            if index < len(xs) - 1:
                ax.annotate(
                    "",
                    xy=(xs[index + 1] - 0.105, y),
                    xytext=(x + 0.105, y),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY_DARK, linewidth=0.75),
                )
    panel_label(ax, "c")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.94, bottom=0.07)
    save_figure(fig, FIGURES, "Figure1")


def figure2() -> None:
    contrasts = pd.read_csv(DATA / "ac_vs_other_ligand_contrasts.csv")
    meta = pd.read_csv(DATA / "meta_ligand_results.csv")
    inferential = ["GSE211376", "GSE273274", "GSE274987", "GSE286419"]
    gene_order = ["TGFB2", "CSF1", "ANXA1", "CCL2", "TGFB1", "SPP1"]
    fig = plt.figure(figsize=(7.1, 5.45))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[0.92, 1.12],
        width_ratios=[1.35, 0.78],
        hspace=0.50,
        wspace=0.42,
    )
    top = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0, :], wspace=0.32)
    for index, cohort in enumerate(inferential):
        ax = fig.add_subplot(top[0, index])
        row = contrasts[
            (contrasts["cohort"] == cohort) & (contrasts["gene"] == "TGFB2")
        ].iloc[0]
        values = json_frame(row)
        for item in values.itertuples():
            ax.plot(
                [0, 1],
                [item.other_mean, item.ac_mean],
                color=GRAY,
                linewidth=0.65,
                alpha=0.8,
                zorder=1,
            )
            ax.scatter(
                [0, 1],
                [item.other_mean, item.ac_mean],
                s=16,
                color=[STATE_COLORS["Other malignant"], STATE_COLORS["AC-like"]],
                edgecolor="white",
                linewidth=0.4,
                zorder=2,
            )
        ax.plot(
            [0, 1],
            [values["other_mean"].median(), values["ac_mean"].median()],
            color=INK,
            linewidth=2.0,
            zorder=3,
        )
        ax.set_xticks([0, 1], ["Other", "AC-like"])
        ax.set_title(
            f"{cohort_display(cohort, short=True)}  n={len(values)}",
            fontsize=7.4,
        )
        ax.set_xlim(-0.24, 1.24)
        open_axes(ax, grid_axis="y")
        if index == 0:
            ax.set_ylabel("Mean log-normalized TGFB2")
            panel_label(ax, "a")
        else:
            ax.set_ylabel("")

    ax = fig.add_subplot(gs[1, 0])
    cohort_order = COHORT_ORDER
    xmap = {gene: i for i, gene in enumerate(gene_order)}
    ymap = {cohort: i for i, cohort in enumerate(cohort_order[::-1])}
    max_abs = float(np.nanmax(np.abs(contrasts["median_difference"])))
    norm = mpl.colors.Normalize(-max_abs, max_abs)
    for row in contrasts.itertuples():
        x, y = xmap[row.gene], ymap[row.cohort]
        significant = pd.notna(row.q_value) and row.q_value < 0.05
        ax.scatter(
            x,
            y,
            s=24 + 6 * row.n_patients,
            color=DIVERGING(norm(row.median_difference)),
            edgecolor=INK if significant else "white",
            linewidth=0.75 if significant else 0.4,
        )
    ax.set_xticks(range(len(gene_order)), gene_order, rotation=32, ha="right")
    ax.set_yticks(
        range(len(cohort_order)),
        [cohort_display(item) for item in cohort_order[::-1]],
    )
    ax.set_title("AC-like minus other malignant states", pad=6)
    ax.set_xlabel("Ligand")
    clean_axes(ax)
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=DIVERGING),
        ax=ax,
        fraction=0.035,
        pad=0.025,
    )
    colorbar.outline.set_visible(False)
    colorbar.set_label("Median expression difference", fontsize=6.5)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 1])
    ordered = meta.set_index("gene").loc[gene_order].reset_index()
    yvals = np.arange(len(ordered))[::-1]
    for y, row in zip(yvals, ordered.itertuples()):
        robust = row.loo_q_value < 0.05
        color = CORAL if robust else GRAY_DARK
        ax.plot([0, row.meta_z], [y, y], color=GRID, linewidth=1.4)
        ax.scatter(
            row.meta_z,
            y,
            s=42,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        ax.text(
            row.meta_z + 0.11,
            y,
            f"q={row.q_value:.2g}",
            va="center",
            fontsize=6.1,
            color=color,
        )
    ax.axvline(0, color=GRAY_DARK, linewidth=0.65)
    ax.set_yticks(yvals, ordered["gene"])
    ax.set_xlabel("Signed Stouffer z")
    ax.set_title("Cross-cohort evidence", pad=6)
    open_axes(ax)
    panel_label(ax, "c")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.11)
    save_figure(fig, FIGURES, "Figure2")


def figure3() -> None:
    ac = pd.read_csv(DATA / "patient_ac_fraction.csv")
    tests = pd.read_csv(DATA / "regional_ac_fraction_tests.csv")
    fig = plt.figure(figsize=(7.1, 4.90))
    gs = GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[0.92, 1.05],
        width_ratios=[1.0, 1.0, 0.88],
        hspace=0.53,
        wspace=0.40,
    )
    for index, cohort in enumerate(["GSE273274", "GSE276177", "GSE286419"]):
        ax = fig.add_subplot(gs[0, index])
        frame = ac[ac["cohort"] == cohort]
        wide = frame.pivot_table(
            index="patient",
            columns="region",
            values="fraction_of_malignant",
            aggfunc="mean",
        )
        wide = wide[[c for c in ["Peritumoral", "Tumor"] if c in wide]].dropna()
        for _, row in wide.iterrows():
            ax.plot([0, 1], row[["Peritumoral", "Tumor"]], color=GRAY, linewidth=0.7)
            ax.scatter(
                [0, 1],
                row[["Peritumoral", "Tumor"]],
                s=17,
                color=[REGION_COLORS["Peritumoral"], REGION_COLORS["Tumor"]],
                edgecolor="white",
                linewidth=0.4,
                zorder=2,
            )
        if len(wide):
            ax.plot(
                [0, 1],
                [wide["Peritumoral"].median(), wide["Tumor"].median()],
                color=INK,
                linewidth=2.0,
            )
        ax.set_xticks([0, 1], ["Peritumoral", "Tumor"], rotation=22, ha="right")
        ax.set_ylim(-0.04, 1.04)
        ax.set_title(f"{cohort_display(cohort, short=True)}  n={len(wide)}")
        open_axes(ax, grid_axis="y")
        if index == 0:
            ax.set_ylabel("AC-like fraction of malignant cells")
            panel_label(ax, "a")

    ax = fig.add_subplot(gs[1, :2])
    frame = ac[ac["cohort"] == "GSE274987"]
    wide = frame.pivot_table(
        index="patient",
        columns="region",
        values="fraction_of_malignant",
        aggfunc="mean",
    )
    region_order = ["Core", "Edge", "Peripheral"]
    for _, row in wide.iterrows():
        values = row.reindex(region_order).to_numpy(float)
        valid = np.isfinite(values)
        ax.plot(np.arange(3)[valid], values[valid], color=GRAY, linewidth=0.7)
        for index, value in enumerate(values):
            if np.isfinite(value):
                ax.scatter(
                    index,
                    value,
                    s=18,
                    color=REGION_COLORS[region_order[index]],
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=2,
                )
    medians = [wide.get(region, pd.Series(dtype=float)).median() for region in region_order]
    ax.plot(
        range(3),
        medians,
        color=INK,
        linewidth=2.0,
        marker="D",
        markersize=3.5,
    )
    ax.set_xticks(range(3), region_order)
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel("AC-like fraction of malignant cells")
    ax.set_title("274986/7 regional trajectories", pad=6)
    open_axes(ax, grid_axis="y")
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 2])
    labels = {
        "GSE273274": "273274 T-P",
        "GSE276177": "276177 T-P",
        "GSE286419": "286419 T-P",
        "GSE274987_core_peripheral": "274986/7 C-P",
        "GSE274987_core_edge": "274986/7 C-E",
    }
    rows = tests.set_index("comparison").loc[list(labels)].reset_index()
    yvals = np.arange(len(rows))[::-1]
    for y, row in zip(yvals, rows.itertuples()):
        differences = np.asarray(json.loads(row.patient_differences), dtype=float)
        ax.scatter(
            differences,
            np.repeat(y, len(differences)) + RNG.normal(0, 0.045, len(differences)),
            s=16,
            color=COHORT_COLORS[row.cohort],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.35,
        )
        ax.scatter(
            row.median_difference,
            y,
            s=38,
            marker="D",
            color=INK,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
    ax.axvline(0, color=GRAY_DARK, linewidth=0.65, linestyle="--")
    ax.set_yticks(yvals, [labels[item] for item in rows["comparison"]])
    ax.set_xlabel("Region A minus B")
    ax.set_title("Paired effects", pad=6)
    open_axes(ax)
    panel_label(ax, "c")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.95, bottom=0.11)
    save_figure(fig, FIGURES, "Figure3")


def figure4() -> None:
    regional = pd.read_csv(DATA / "regional_communication_tests.csv")
    meta = pd.read_csv(DATA / "meta_ligand_results.csv")
    comparisons = [
        "GSE273274",
        "GSE276177",
        "GSE286419",
        "GSE274987_core_peripheral",
        "GSE274987_core_edge",
    ]
    labels = ["273274 T-P", "276177 T-P", "286419 T-P", "274987 C-P", "274987 C-E"]
    pairs = [
        "TGFB2-TGFBR1",
        "TGFB2-TGFBR2",
        "TGFB1-TGFBR1",
        "ANXA1-FPR1",
        "SPP1-CD44",
        "CSF1-CSF1R",
        "CCL2-CCR2",
    ]
    fig = plt.figure(figsize=(7.1, 5.25))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[0.95, 1.05],
        width_ratios=[1.22, 0.78],
        hspace=0.55,
        wspace=0.43,
    )

    ax = fig.add_subplot(gs[0, :])
    max_abs = float(np.nanmax(np.abs(regional["median_difference"])))
    norm = mpl.colors.Normalize(-max_abs, max_abs)
    for row_index, (comparison, label) in enumerate(zip(comparisons, labels)):
        y = len(comparisons) - 1 - row_index
        frame = regional[regional["comparison"] == comparison].set_index("pair")
        for x, pair in enumerate(pairs):
            row = frame.loc[pair]
            size = 22 + 130 * abs(row["median_difference"]) / max_abs
            ax.scatter(
                x,
                y,
                s=size,
                color=DIVERGING(norm(row["median_difference"])),
                edgecolor="white",
                linewidth=0.45,
            )
        n = int(frame["n_patients"].iloc[0])
        floor = float(frame["minimum_attainable_p"].min())
        ax.text(len(pairs) - 0.1, y, f"n={n}  min p={floor:.3g}", va="center", fontsize=6.2, color=MUTED)
    ax.set_xticks(
        range(len(pairs)),
        [pair.replace("-", "\n") for pair in pairs],
    )
    ax.set_yticks(range(len(comparisons)), labels[::-1])
    ax.set_xlim(-0.55, len(pairs) + 1.10)
    ax.set_ylim(-0.6, len(comparisons) - 0.4)
    ax.set_title("Regional change in ligand-receptor co-availability", pad=6)
    clean_axes(ax)
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=DIVERGING),
        ax=ax,
        fraction=0.024,
        pad=0.02,
    )
    colorbar.outline.set_visible(False)
    colorbar.set_label("Median region A minus B", fontsize=6.3)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[1, 0])
    yvals = np.arange(len(comparisons))[::-1]
    for y, comparison in zip(yvals, comparisons):
        frame = regional[
            (regional["comparison"] == comparison)
            & regional["pair"].isin(["TGFB2-TGFBR1", "TGFB2-TGFBR2"])
        ].set_index("pair")
        left = float(frame.loc["TGFB2-TGFBR1", "median_difference"])
        right = float(frame.loc["TGFB2-TGFBR2", "median_difference"])
        ax.plot([left, right], [y, y], color=GRID, linewidth=1.2)
        ax.scatter(
            [left, right],
            [y, y],
            s=[34, 34],
            marker="o",
            color=[CORAL, TEAL],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
    ax.axvline(0, color=GRAY_DARK, linewidth=0.65, linestyle="--")
    ax.set_yticks(yvals, labels)
    ax.set_xlabel("Median regional difference")
    ax.set_title("TGFB2 receptor-specific effects", pad=6)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=CORAL, label="TGFBR1"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL, label="TGFBR2"),
        ],
        loc="lower right",
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    open_axes(ax)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 1])
    temp = regional.copy()
    temp["gene"] = temp["pair"].str.split("-").str[0]
    best_q = temp.groupby("gene", observed=True)["q_value"].min()
    label_offsets = {
        "CCL2": (0.05, 0.07),
        "TGFB1": (0.05, -0.08),
    }
    for row in meta.itertuples():
        if row.gene not in best_q:
            continue
        x = -np.log10(max(row.q_value, 1e-12))
        y = -np.log10(max(float(best_q.loc[row.gene]), 1e-12))
        robust = row.loo_q_value < 0.05
        color = CORAL if robust else GRAY_DARK
        ax.scatter(x, y, s=36, color=color, edgecolor="white", linewidth=0.5)
        dx, dy = label_offsets.get(row.gene, (0.04, 0.035))
        ax.text(x + dx, y + dy, row.gene, fontsize=6.2, color=color)
    limit = max(ax.get_xlim()[1], ax.get_ylim()[1], 1.45)
    ax.plot([0, limit], [0, limit], color=GRAY, linestyle="--", linewidth=0.65)
    ax.axhline(-np.log10(0.05), color=GRAY_DARK, linestyle=":", linewidth=0.65)
    ax.axvline(-np.log10(0.05), color=GRAY_DARK, linestyle=":", linewidth=0.65)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Intrinsic evidence, -log10(q)")
    ax.set_ylabel("Best regional evidence, -log10(q)")
    ax.set_title("Evidence layers", pad=6)
    open_axes(ax)
    panel_label(ax, "c")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.12)
    save_figure(fig, FIGURES, "Figure4")


def supplementary_figure_s1() -> None:
    expression = pd.read_csv(DATA / "patient_state_expression.csv")
    state_order = ["AC-like", "MES-like", "OPC-like", "NPC-like", "Myeloid"]
    expression = expression[expression["state"].isin(state_order)].copy()
    expression["cohort_order"] = expression["cohort"].map(
        {cohort: index for index, cohort in enumerate(COHORT_ORDER)}
    )
    samples = (
        expression[["cohort", "cohort_order", "patient", "region"]]
        .drop_duplicates()
        .sort_values(["cohort_order", "region", "patient"])
        .reset_index(drop=True)
    )
    row_lookup = {
        (row.cohort, row.patient, row.region): index
        for index, row in samples.iterrows()
    }
    height = min(10.3, max(7.4, 1.5 + 0.13 * len(samples)))
    fig, ax = plt.subplots(figsize=(7.1, height))
    for row in expression.itertuples():
        y = len(samples) - 1 - row_lookup[(row.cohort, row.patient, row.region)]
        x = state_order.index(row.state)
        eligible = row.n_cells >= 20
        ax.scatter(
            x,
            y,
            s=8 + 14 * np.log10(max(row.n_cells, 1)),
            facecolor=COHORT_COLORS[row.cohort] if eligible else "white",
            edgecolor=COHORT_COLORS[row.cohort],
            linewidth=0.65,
        )
    labels = []
    for row in samples.itertuples():
        patient = str(row.patient).replace(f"{row.cohort}_", "")
        labels.append(
            f"{cohort_display(row.cohort, short=True)}  "
            f"{patient} {short_region(row.region)}"
        )
    ax.set_xticks(range(len(state_order)), ["AC", "MES", "OPC", "NPC", "Myeloid"])
    ax.set_yticks(range(len(samples)), labels[::-1])
    ax.tick_params(axis="y", labelsize=5.5)
    ax.set_xlim(-0.55, len(state_order) - 0.45)
    ax.set_ylim(-0.8, len(samples) - 0.2)
    boundaries = samples.groupby("cohort_order", sort=True).size().cumsum().to_numpy()[:-1]
    for boundary in boundaries:
        ax.axhline(len(samples) - boundary - 0.5, color=GRID, linewidth=0.8)
    ax.set_title("Patient-region state counts and predefined eligibility", pad=7)
    clean_axes(ax)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL, markeredgecolor=TEAL, label="Eligible, n>=20"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=TEAL, label="Below threshold"),
        ],
        loc="upper right",
        ncol=2,
        bbox_to_anchor=(1.0, 1.035),
    )
    panel_label(ax, "a")
    fig.subplots_adjust(left=0.24, right=0.98, top=0.97, bottom=0.045)
    save_figure(fig, FIGURES, "SupplementaryFigureS1")


def supplementary_figure_s2() -> None:
    expression = pd.read_csv(DATA / "patient_state_expression.csv")
    genes = ["TGFB2", "CSF1", "ANXA1", "CCL2", "TGFB1", "SPP1"]
    rows = []
    for cohort in COHORT_ORDER:
        target = "Core" if cohort == "GSE274987" else "Tumor"
        frame = expression[
            (expression["cohort"] == cohort) & (expression["region"] == target)
        ]
        for state in ["AC-like", "MES-like", "OPC-like", "NPC-like"]:
            state_frame = frame[(frame["state"] == state) & (frame["n_cells"] >= 20)]
            row = {
                "label": (
                    f"{cohort_display(cohort, short=True)} | "
                    f"{state.replace('-like', '')}"
                )
            }
            for gene in genes:
                row[gene] = (
                    state_frame[f"mean_{gene}"].median()
                    if not state_frame.empty
                    else np.nan
                )
            rows.append(row)
    matrix = pd.DataFrame(rows).set_index("label")
    matrix = matrix.apply(
        lambda column: (column - column.mean()) / column.std(ddof=0), axis=0
    )
    fig, ax = plt.subplots(figsize=(7.1, 5.05))
    heatmap = sns.heatmap(
        matrix,
        cmap=DIVERGING,
        center=0,
        linewidths=0.55,
        linecolor="white",
        cbar_kws={"label": "Column z score", "shrink": 0.78},
        ax=ax,
    )
    ax.set_xlabel("Ligand")
    ax.set_ylabel("")
    ax.set_title("Median patient-level expression by malignant state", pad=7)
    ax.tick_params(axis="y", labelsize=6.2)
    heatmap.collections[0].colorbar.outline.set_visible(False)
    panel_label(ax, "a")
    fig.subplots_adjust(left=0.20, right=0.96, top=0.94, bottom=0.10)
    save_figure(fig, FIGURES, "SupplementaryFigureS2")


def supplementary_figure_s3() -> None:
    patient = pd.read_csv(DATA / "uniform_annotation_patient_contrasts.csv")
    uniform = pd.read_csv(DATA / "uniform_annotation_cohort_results.csv")
    primary = pd.read_csv(DATA / "ac_vs_other_ligand_contrasts.csv")
    primary = primary[primary["gene"] == "TGFB2"][
        ["cohort", "median_difference"]
    ].rename(columns={"median_difference": "primary_difference"})
    merged = uniform.merge(primary, on="cohort", how="left")
    summary = json.loads(
        (DATA / "uniform_annotation_summary.json").read_text(encoding="utf-8")
    )
    fig = plt.figure(figsize=(7.1, 3.15))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.96, 1.02, 1.12], wspace=0.48)

    ax = fig.add_subplot(gs[0, 0])
    ordered = merged.set_index("cohort").loc[SENSITIVITY_ORDER].reset_index()
    yvals = np.arange(len(ordered))[::-1]
    for y, row in zip(yvals, ordered.itertuples()):
        if pd.notna(row.primary_difference):
            ax.plot(
                [row.primary_difference, row.median_difference],
                [y, y],
                color=GRID,
                linewidth=1.1,
            )
            ax.scatter(
                row.primary_difference,
                y,
                s=28,
                facecolor="white",
                edgecolor=COHORT_COLORS[row.cohort],
                linewidth=0.8,
            )
        ax.scatter(
            row.median_difference,
            y,
            s=32,
            color=COHORT_COLORS[row.cohort],
            edgecolor="white",
            linewidth=0.45,
        )
    ax.axvline(0, color=GRAY_DARK, linewidth=0.65, linestyle="--")
    ax.set_yticks(
        yvals,
        [cohort_display(item, short=True) for item in ordered["cohort"]],
    )
    ax.set_xlabel("Median TGFB2 difference")
    ax.set_title("Annotation sensitivity", pad=6)
    open_axes(ax)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    for index, cohort in enumerate(SENSITIVITY_ORDER):
        values = patient.loc[
            patient["cohort"] == cohort, "tgfb2_difference"
        ].to_numpy(float)
        ax.scatter(
            np.repeat(index, len(values)) + RNG.normal(0, 0.045, len(values)),
            values,
            s=16,
            color=COHORT_COLORS[cohort],
            alpha=0.8,
            edgecolor="white",
            linewidth=0.35,
        )
        if len(values):
            ax.scatter(
                index,
                np.median(values),
                s=36,
                marker="D",
                color=INK,
                edgecolor="white",
                linewidth=0.5,
            )
    ax.axhline(0, color=GRAY_DARK, linewidth=0.65, linestyle="--")
    ax.set_xticks(
        range(len(SENSITIVITY_ORDER)),
        [cohort_display(item, short=True) for item in SENSITIVITY_ORDER],
        rotation=42,
        ha="right",
    )
    ax.set_ylabel("Uniform AC-like minus other")
    ax.set_title("Patient-level effects", pad=6)
    open_axes(ax)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    for cohort, frame in patient.groupby("cohort", observed=True):
        ax.scatter(
            frame["ac_score_difference"],
            frame["tgfb2_difference"],
            s=18,
            color=COHORT_COLORS[cohort],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.35,
        )
    x = patient["ac_score_difference"].to_numpy(float)
    y = patient["tgfb2_difference"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, intercept + slope * xx, color=INK, linewidth=1.0)
    lineage = summary["lineage_adjustment"]
    ax.text(
        0.03,
        0.97,
        f"rho={lineage['spearman_rho']:.2f}\nHC3 P={lineage['hc3_p_value']:.2g}",
        transform=ax.transAxes,
        va="top",
        fontsize=6.2,
        color=MUTED,
    )
    ax.axhline(0, color=GRAY, linewidth=0.6, linestyle="--")
    ax.set_xlabel("AC marker-score difference")
    ax.set_ylabel("TGFB2 difference")
    ax.set_title("Lineage-program sensitivity", pad=6)
    open_axes(ax)
    panel_label(ax, "c")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.19)
    save_figure(fig, FIGURES, "SupplementaryFigureS3")


def supplementary_figure_s4() -> None:
    grid = pd.read_csv(DATA / "parameter_sensitivity_grid.csv")
    cohort = pd.read_csv(DATA / "parameter_sensitivity_cohort_results.csv")
    score_thresholds = [0.0, 0.025, 0.05, 0.10]
    cell_thresholds = [10, 20, 30, 50]
    modes = ["Raw marker mean", "Within-cohort gene z-score"]
    fig = plt.figure(figsize=(7.1, 4.85))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.78], hspace=0.50, wspace=0.42)
    for index, mode in enumerate(modes):
        ax = fig.add_subplot(gs[0, index])
        frame = grid[grid["scoring_mode"] == mode]
        values = (
            frame.pivot(
                index="minimum_cells", columns="score_threshold", values="meta_z"
            )
            .reindex(index=cell_thresholds, columns=score_thresholds)
        )
        labels = frame.copy()
        labels["label"] = labels.apply(
            lambda row: f"{row.meta_z:.2f}\n{int(row.positive_cohorts)}/{int(row.n_cohorts)} +",
            axis=1,
        )
        annotations = (
            labels.pivot(
                index="minimum_cells", columns="score_threshold", values="label"
            )
            .reindex(index=cell_thresholds, columns=score_thresholds)
        )
        heatmap = sns.heatmap(
            values,
            ax=ax,
            annot=annotations,
            fmt="",
            cmap=TEAL_SEQ,
            vmin=3.0,
            vmax=4.2,
            linewidths=0.7,
            linecolor="white",
            cbar_kws={"label": "Signed Stouffer z", "shrink": 0.75},
            annot_kws={"fontsize": 6.2},
        )
        ax.set_xlabel("Score threshold")
        ax.set_ylabel("Minimum cells per group")
        ax.set_xticklabels([f"{value:g}" for value in values.columns])
        ax.set_yticklabels([str(value) for value in values.index], rotation=0)
        ax.set_title(mode, pad=6)
        heatmap.collections[0].colorbar.outline.set_visible(False)
        panel_label(ax, chr(ord("a") + index))

    ax = fig.add_subplot(gs[1, :])
    baseline = cohort[
        np.isclose(cohort["score_threshold"], 0.05)
        & cohort["minimum_cells"].eq(20)
    ]
    order = ["GSE162631", "GSE211376", "GSE273274", "GSE274987", "GSE286419"]
    wide = baseline.pivot(
        index="cohort", columns="scoring_mode", values="median_difference"
    ).reindex(order)
    yvals = np.arange(len(order))[::-1]
    raw = wide["Raw marker mean"].to_numpy(float)
    standardized = wide["Within-cohort gene z-score"].to_numpy(float)
    for y, left, right in zip(yvals, raw, standardized):
        ax.plot([left, right], [y, y], color=GRID, linewidth=1.2)
    ax.scatter(raw, yvals, color=CORAL, s=30, label="Raw marker mean")
    ax.scatter(standardized, yvals, color=TEAL, s=30, label="Gene z-score")
    ax.axvline(0, color=GRAY_DARK, linewidth=0.65, linestyle="--")
    ax.set_yticks(
        yvals, [cohort_display(item, short=True) for item in order]
    )
    ax.set_xlabel("Cohort median TGFB2 difference")
    ax.set_title("Baseline parameters", pad=6)
    ax.legend(loc="lower right", ncol=2)
    open_axes(ax)
    panel_label(ax, "c")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.94, bottom=0.11)
    save_figure(fig, FIGURES, "SupplementaryFigureS4")


def supplementary_figure_s5() -> None:
    results = pd.read_csv(DATA / "exact_meta_calibration.csv")
    tail = pd.read_csv(DATA / "exact_meta_null_tail.csv")
    gene_order = ["TGFB2", "CSF1", "ANXA1", "CCL2", "TGFB1", "SPP1"]
    colors = {"TGFB2": CORAL, "CSF1": GOLD, "ANXA1": BLUE}

    def slope_panel(
        ax: plt.Axes,
        normal_column: str,
        exact_column: str,
        title: str,
    ) -> None:
        ordered = results.set_index("gene").loc[gene_order].reset_index()
        for row in ordered.itertuples(index=False):
            normal = -np.log10(max(float(getattr(row, normal_column)), 1e-15))
            exact = -np.log10(max(float(getattr(row, exact_column)), 1e-15))
            color = colors.get(row.gene, GRAY_DARK)
            ax.plot([0, 1], [normal, exact], color=color, linewidth=0.9, alpha=0.8)
            ax.scatter(
                [0, 1],
                [normal, exact],
                color=color,
                s=24,
                edgecolor="white",
                linewidth=0.4,
                zorder=2,
            )
            if row.gene in colors:
                ax.text(1.04, exact, row.gene, va="center", fontsize=6.1, color=color)
        ax.axhline(-np.log10(0.05), color=GRAY_DARK, linestyle="--", linewidth=0.65)
        ax.set_xlim(-0.18, 1.45)
        ax.set_xticks([0, 1], ["Normal\napprox.", "Exact\nsign flip"])
        ax.set_ylabel("Evidence, -log10(q)")
        ax.set_title(title, pad=6)
        open_axes(ax)

    fig = plt.figure(figsize=(7.1, 4.55))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.95, 0.88], hspace=0.48, wspace=0.44)
    ax = fig.add_subplot(gs[0, 0])
    slope_panel(
        ax,
        "normal_approximation_q_value",
        "exact_signflip_q_value",
        "Full meta-analysis",
    )
    panel_label(ax, "a")
    ax = fig.add_subplot(gs[0, 1])
    slope_panel(
        ax,
        "normal_loo_q_value",
        "exact_loo_q_value",
        "Leave-one-cohort-out",
    )
    panel_label(ax, "b")
    ax = fig.add_subplot(gs[1, :])
    for gene in ["TGFB2", "CSF1"]:
        frame = tail[tail["gene"] == gene]
        color = colors[gene]
        ax.plot(
            frame["absolute_z_threshold"],
            frame["exact_tail_probability"],
            color=color,
            linewidth=1.5,
            label=gene,
        )
        observed_z = float(frame["observed_absolute_z"].iloc[0])
        observed_p = float(frame["observed_exact_p_value"].iloc[0])
        ax.scatter(
            observed_z,
            observed_p,
            color=color,
            s=30,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        ax.vlines(
            observed_z,
            1e-6,
            observed_p,
            color=color,
            linestyle=":",
            linewidth=0.75,
        )
    ax.axhline(0.05, color=GRAY_DARK, linestyle="--", linewidth=0.65)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.1)
    ax.set_xlabel("Absolute signed Stouffer z")
    ax.set_ylabel("Exact sign-flip tail probability")
    ax.set_title("Exhaustive patient-level null distribution", pad=6)
    ax.legend()
    open_axes(ax)
    panel_label(ax, "c")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.94, bottom=0.12)
    save_figure(fig, FIGURES, "SupplementaryFigureS5")


def figure5() -> None:
    result_dir = EXTERNAL_RESULTS / "gbmspace"
    patient_ligands = pd.read_csv(result_dir / "patient_ligand_metrics.csv")
    patient_spatial = pd.read_csv(result_dir / "patient_spatial_metrics.csv")
    representative = pd.read_csv(result_dir / "representative_plot.csv")
    tgfb2 = patient_ligands[patient_ligands["ligand"] == "TGFB2"].copy()
    fig = plt.figure(figsize=(7.1, 5.25))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[0.82, 1.12], hspace=0.42)
    top = GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=outer[0],
        width_ratios=[1.0, 1.35, 1.20],
        wspace=0.46,
    )
    bottom = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.32)

    ax = fig.add_subplot(top[0, 0])
    raw = tgfb2["rho_ac_fraction"].to_numpy(float)
    adjusted = tgfb2["rho_ac_fraction_adjusted"].to_numpy(float)
    for raw_value, adjusted_value in zip(raw, adjusted):
        if np.isfinite(raw_value) and np.isfinite(adjusted_value):
            ax.plot([0, 1], [raw_value, adjusted_value], color=GRID, linewidth=0.75)
    ax.scatter(
        RNG.normal(0, 0.03, len(raw)),
        raw,
        s=19,
        color=TEAL,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.scatter(
        RNG.normal(1, 0.03, len(adjusted)),
        adjusted,
        s=19,
        color=BLUE,
        edgecolor="white",
        linewidth=0.4,
    )
    median_bar(ax, 0, float(np.nanmedian(raw)), color=CORAL)
    median_bar(ax, 1, float(np.nanmedian(adjusted)), color=CORAL)
    ax.axhline(0, color=GRAY_DARK, linewidth=0.65, linestyle="--")
    ax.set_xticks([0, 1], ["Raw", "Adjusted"])
    ax.set_ylabel("Patient median rho")
    ax.set_title("TGFB2-AC association", pad=6)
    open_axes(ax)
    panel_label(ax, "a")

    ax = fig.add_subplot(top[0, 1])
    plot_data = patient_ligands.pivot(
        index="patient", columns="ligand", values="rho_ac_fraction_adjusted"
    ).reindex(columns=LIGANDS)
    for index, ligand in enumerate(LIGANDS):
        values = plot_data[ligand].dropna().to_numpy(float)
        ax.scatter(
            np.full(len(values), index) + RNG.normal(0, 0.05, len(values)),
            values,
            s=14,
            color=CORAL if ligand == "TGFB2" else GRAY_DARK,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.3,
        )
        if len(values):
            median_bar(
                ax,
                index,
                float(np.median(values)),
                color=CORAL if ligand == "TGFB2" else INK,
                width=0.30,
                linewidth=1.6,
            )
    ax.axhline(0, color=GRAY_DARK, linewidth=0.65, linestyle="--")
    ax.set_xticks(range(len(LIGANDS)), LIGANDS, rotation=38, ha="right")
    ax.set_ylabel("Adjusted patient median rho")
    ax.set_title("Fixed ligand panel", pad=6)
    open_axes(ax)
    panel_label(ax, "b")

    nested = GridSpecFromSubplotSpec(1, 2, subplot_spec=top[0, 2], wspace=0.62)
    ax = fig.add_subplot(nested[0, 0])
    add_strip(
        ax,
        tgfb2["mean_top_minus_bottom_ac_delta"],
        title="AC-rich contrast",
        ylabel="TGFB2 delta",
    )
    ax.set_title("AC-rich\ncontrast", fontsize=7.6, pad=4)
    panel_label(ax, "c")
    ax = fig.add_subplot(nested[0, 1])
    add_strip(
        ax,
        patient_spatial["tgfb2_neighbor_receiver_delta"],
        title="Neighbor receiver",
        ylabel="",
        color=PURPLE,
    )
    ax.set_title("Neighbor\nreceiver", fontsize=7.6, pad=4)
    panel_label(ax, "d")

    spatial_specs = [
        ("ac_fraction", "AC-like fraction", TEAL_SEQ),
        ("tgfb2", "TGFB2 expression", CORAL_SEQ),
        ("receiver", "Myeloid x TGFBR score", PURPLE_SEQ),
    ]
    for index, (field, title, cmap) in enumerate(spatial_specs):
        ax = fig.add_subplot(bottom[0, index])
        add_spatial(
            fig,
            ax,
            representative,
            x="x",
            y="y",
            value=field,
            title=title,
            cmap=cmap,
            point_size=3.0,
        )
        panel_label(ax, chr(ord("e") + index))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.07)
    save_figure(fig, FIGURES, "Figure5")


def figure6() -> None:
    result_dir = EXTERNAL_RESULTS / "cptac_gbm"
    comparisons = pd.read_csv(result_dir / "ligand_tumor_normal.csv")
    plex = pd.read_csv(result_dir / "tgfb2_plex_matched_contrasts.csv")
    associations = pd.read_csv(result_dir / "tgfb2_tumor_associations.csv")
    ordered = comparisons.set_index("gene").loc[LIGANDS].reset_index()
    tgfb2 = ordered[ordered["gene"] == "TGFB2"].iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.75))

    ax = axes[0, 0]
    ax.plot(
        [0, 1],
        [tgfb2["tumor_median"], tgfb2["normal_median"]],
        color=GRID,
        linewidth=1.4,
    )
    ax.scatter(
        [0, 1],
        [tgfb2["tumor_median"], tgfb2["normal_median"]],
        s=58,
        color=[CORAL, TEAL],
        edgecolor="white",
        linewidth=0.65,
        zorder=3,
    )
    ax.text(
        0.50,
        0.08,
        f"Median delta = {tgfb2['median_difference']:.2f}",
        transform=ax.transAxes,
        ha="center",
        fontsize=6.5,
        color=MUTED,
    )
    ax.set_xticks(
        [0, 1],
        [
            f"GBM tumor\nn={int(tgfb2['n_tumor'])}",
            f"Normal brain\nn={int(tgfb2['n_normal'])}",
        ],
    )
    ax.set_ylabel("TGFB2 protein log2 ratio")
    ax.set_title("Tissue-level TGFB2 signal", pad=6)
    open_axes(ax, grid_axis="y")
    panel_label(ax, "a")

    ax = axes[0, 1]
    yvals = np.arange(len(ordered))[::-1]
    for y, row in zip(yvals, ordered.itertuples()):
        color = CORAL if row.gene == "TGFB2" else GRAY_DARK
        ax.plot(
            [row.median_difference_ci_low, row.median_difference_ci_high],
            [y, y],
            color=color,
            linewidth=1.2,
        )
        ax.scatter(
            row.median_difference,
            y,
            s=34,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
    ax.axvline(0, color=GRAY_DARK, linewidth=0.65, linestyle="--")
    ax.set_yticks(yvals, ordered["gene"])
    ax.set_xlabel("Tumor minus normal median log2 ratio")
    ax.set_title("Fixed ligand panel, 95% bootstrap CI", pad=6)
    open_axes(ax)
    panel_label(ax, "b")

    ax = axes[1, 0]
    plex = plex.sort_values("tmt_plex")
    yvals = np.arange(len(plex))[::-1]
    ax.scatter(
        plex["median_difference"],
        yvals,
        s=28,
        color=TEAL,
        edgecolor="white",
        linewidth=0.45,
    )
    ax.axvline(0, color=GRAY_DARK, linewidth=0.65, linestyle="--")
    ax.set_yticks(yvals, [f"Plex {item}" for item in plex["tmt_plex"]])
    ax.set_xlabel("Tumor minus normal TGFB2 median")
    ax.set_title("Matched TMT plex contrasts", pad=6)
    open_axes(ax, grid_axis="x")
    panel_label(ax, "c")

    ax = axes[1, 1]
    labels = {
        "astrocytic_protein_score": "Astrocytic",
        "myeloid_protein_score": "Myeloid",
        "tgfbr_protein_score": "TGFBR",
    }
    ordered_assoc = associations.assign(
        label=associations["score"].map(labels)
    ).sort_values("spearman_rho")
    yvals = np.arange(len(ordered_assoc))
    for y, row in zip(yvals, ordered_assoc.itertuples()):
        color = CORAL if row.spearman_q < 0.05 else GRAY_DARK
        ax.plot([0, row.spearman_rho], [y, y], color=GRID, linewidth=1.3)
        ax.scatter(
            row.spearman_rho,
            y,
            s=36,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.text(
            row.spearman_rho + 0.012,
            y,
            f"q={row.spearman_q:.2g}",
            va="center",
            fontsize=6.2,
            color=MUTED,
        )
    ax.axvline(0, color=GRAY_DARK, linewidth=0.65)
    ax.set_yticks(yvals, ordered_assoc["label"])
    ax.set_xlim(-0.16, 0.24)
    ax.set_xlabel("Spearman rho with TGFB2 protein")
    ax.set_title("Tumor protein-score associations", pad=6)
    open_axes(ax)
    panel_label(ax, "d")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.12, hspace=0.48, wspace=0.43)
    save_figure(fig, FIGURES, "Figure6")


def supplementary_figure_s6() -> None:
    result_dir = EXTERNAL_RESULTS / "greenwald"
    patient_ligands = pd.read_csv(result_dir / "patient_ligand_metrics.csv")
    patient_spatial = pd.read_csv(result_dir / "patient_spatial_metrics.csv")
    representative = pd.read_csv(result_dir / "representative_plot.csv")
    tgfb2 = patient_ligands[patient_ligands["ligand"] == "TGFB2"].copy()
    fig = plt.figure(figsize=(7.1, 5.20))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[0.82, 1.10], hspace=0.42)
    top = GridSpecFromSubplotSpec(
        1,
        4,
        subplot_spec=outer[0],
        width_ratios=[0.85, 1.45, 0.85, 0.85],
        wspace=0.52,
    )
    bottom = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.32)

    ax = fig.add_subplot(top[0, 0])
    add_strip(
        ax,
        tgfb2["ac_minus_other_mean"],
        title="TGFB2 in AC spots",
        ylabel="AC minus other",
    )
    panel_label(ax, "a")

    ax = fig.add_subplot(top[0, 1])
    plot_data = patient_ligands.pivot(
        index="patient", columns="ligand", values="ac_minus_other_mean"
    ).reindex(columns=LIGANDS)
    for index, ligand in enumerate(LIGANDS):
        values = plot_data[ligand].dropna().to_numpy(float)
        ax.scatter(
            np.full(len(values), index) + RNG.normal(0, 0.05, len(values)),
            values,
            s=13,
            color=CORAL if ligand == "TGFB2" else GRAY_DARK,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.3,
        )
        if len(values):
            median_bar(
                ax,
                index,
                float(np.median(values)),
                color=CORAL if ligand == "TGFB2" else INK,
                width=0.28,
                linewidth=1.5,
            )
    ax.axhline(0, color=GRAY_DARK, linewidth=0.65, linestyle="--")
    ax.set_xticks(range(len(LIGANDS)), LIGANDS, rotation=38, ha="right")
    ax.set_ylabel("AC minus other malignant")
    ax.set_title("Fixed ligand panel", pad=6)
    open_axes(ax)
    panel_label(ax, "b")

    ax = fig.add_subplot(top[0, 2])
    add_strip(
        ax,
        tgfb2["ac_minus_reactive_mean"],
        title="Malignant vs\nreactive",
        ylabel="TGFB2 delta",
        color=BLUE,
    )
    panel_label(ax, "c")
    ax = fig.add_subplot(top[0, 3])
    add_strip(
        ax,
        patient_spatial["tgfb2_neighbor_receiver_delta"],
        title="Neighbor\nreceiver",
        ylabel="Score delta",
        color=PURPLE,
    )
    panel_label(ax, "d")

    representative["ac_label"] = (
        representative["label"].astype(str).eq("AC").astype(float)
    )
    specs = [
        ("ac_label", "Author-assigned AC spots", TEAL_SEQ),
        ("tgfb2", "TGFB2 expression", CORAL_SEQ),
        ("receiver", "Myeloid x TGFBR score", PURPLE_SEQ),
    ]
    for index, (field, title, cmap) in enumerate(specs):
        ax = fig.add_subplot(bottom[0, index])
        add_spatial(
            fig,
            ax,
            representative,
            x="pixel_col",
            y="pixel_row",
            value=field,
            title=title,
            cmap=cmap,
            point_size=3.6,
        )
        panel_label(ax, chr(ord("e") + index))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.07)
    save_figure(fig, FIGURES, "SupplementaryFigureS6")


def supplementary_figure_s7() -> None:
    result_dir = EXTERNAL_RESULTS / "gse253080"
    patient_ligands = pd.read_csv(result_dir / "patient_ligand_metrics.csv")
    patient_spatial = pd.read_csv(result_dir / "patient_spatial_metrics.csv")
    representative = pd.read_csv(result_dir / "representative_plot.csv")
    tgfb2 = patient_ligands[patient_ligands["ligand"] == "TGFB2"].copy()
    fig = plt.figure(figsize=(7.1, 4.90))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[0.78, 1.08], hspace=0.42)
    top = GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=outer[0],
        width_ratios=[0.80, 1.45, 0.80],
        wspace=0.50,
    )
    bottom = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.32)

    ax = fig.add_subplot(top[0, 0])
    add_strip(
        ax,
        tgfb2["rho_ac_score_adjusted"],
        title="TGFB2 vs\nAC score",
        ylabel="Adjusted rho",
    )
    panel_label(ax, "a")

    ax = fig.add_subplot(top[0, 1])
    plot_data = patient_ligands.pivot(
        index="patient", columns="ligand", values="rho_ac_score_adjusted"
    ).reindex(columns=LIGANDS)
    for index, ligand in enumerate(LIGANDS):
        values = plot_data[ligand].dropna().to_numpy(float)
        ax.scatter(
            np.full(len(values), index) + RNG.normal(0, 0.045, len(values)),
            values,
            s=17,
            color=CORAL if ligand == "TGFB2" else GRAY_DARK,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.35,
        )
        if len(values):
            median_bar(
                ax,
                index,
                float(np.median(values)),
                color=CORAL if ligand == "TGFB2" else INK,
                width=0.30,
                linewidth=1.5,
            )
    ax.axhline(0, color=GRAY_DARK, linewidth=0.65, linestyle="--")
    ax.set_xticks(range(len(LIGANDS)), LIGANDS, rotation=38, ha="right")
    ax.set_ylabel("Adjusted rho")
    ax.set_title("Fixed ligand panel", pad=6)
    open_axes(ax)
    panel_label(ax, "b")

    ax = fig.add_subplot(top[0, 2])
    add_strip(
        ax,
        patient_spatial["tgfb2_neighbor_receiver_delta"],
        title="Descriptive\nproximity",
        ylabel="Receiver-score delta",
        color=PURPLE,
    )
    panel_label(ax, "c")

    specs = [
        ("ac_score", "AC-like score", TEAL_SEQ),
        ("tgfb2", "TGFB2 expression", CORAL_SEQ),
        ("receiver_score", "Myeloid x TGFBR score", PURPLE_SEQ),
    ]
    for index, (field, title, cmap) in enumerate(specs):
        ax = fig.add_subplot(bottom[0, index])
        add_spatial(
            fig,
            ax,
            representative,
            x="pixel_col",
            y="pixel_row",
            value=field,
            title=title,
            cmap=cmap,
            point_size=3.5,
        )
        panel_label(ax, chr(ord("d") + index))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.07)
    save_figure(fig, FIGURES, "SupplementaryFigureS7")


def main() -> None:
    required = [
        DATA / "cohort_manifest.csv",
        DATA / "patient_state_expression.csv",
        DATA / "patient_ac_fraction.csv",
        DATA / "ac_vs_other_ligand_contrasts.csv",
        DATA / "meta_ligand_results.csv",
        DATA / "regional_communication_tests.csv",
        DATA / "regional_ac_fraction_tests.csv",
        DATA / "uniform_annotation_patient_contrasts.csv",
        DATA / "uniform_annotation_cohort_results.csv",
        DATA / "uniform_annotation_summary.json",
        DATA / "parameter_sensitivity_grid.csv",
        DATA / "parameter_sensitivity_cohort_results.csv",
        DATA / "exact_meta_calibration.csv",
        DATA / "exact_meta_null_tail.csv",
        EXTERNAL_RESULTS / "gbmspace" / "representative_plot.csv",
        EXTERNAL_RESULTS / "greenwald" / "representative_plot.csv",
        EXTERNAL_RESULTS / "gse253080" / "representative_plot.csv",
        EXTERNAL_RESULTS / "cptac_gbm" / "ligand_tumor_normal.csv",
        EXTERNAL_RESULTS / "cptac_gbm" / "tgfb2_plex_matched_contrasts.csv",
        EXTERNAL_RESULTS / "cptac_gbm" / "tgfb2_tumor_associations.csv",
    ]
    require_files(required)
    configure_plottie_style()
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    supplementary_figure_s1()
    supplementary_figure_s2()
    supplementary_figure_s3()
    supplementary_figure_s4()
    supplementary_figure_s5()
    supplementary_figure_s6()
    supplementary_figure_s7()
    print(f"Saved 13 Plottie-inspired figures to {FIGURES}")


if __name__ == "__main__":
    main()
