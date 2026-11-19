#!/usr/bin/env python3
# Purpose: redraw the six original submission figures and supplements S1–S7 from frozen results in the original patient,
# ligand and section order.
# Inputs in original_analysis/data/ include cohort_manifest.csv, ac_vs_other_ligand_contrasts.csv and
# meta_ligand_results.csv.
# patient_ac_fraction.csv, regional_ac_fraction_tests.csv and regional_communication_tests.csv supply cohorts, patient
# pairs, effects, P and q.
# patient_state_expression.csv and uniform_annotation_{patient_contrasts,cohort_results}.csv supply expression and
# contrasts.
# uniform_annotation_summary.json completes the uniform-annotation data on patient values, cohort effects and lineage
# adjustment.
# parameter_sensitivity_{grid,cohort_results}.csv and exact_meta_{calibration,null_tail}.csv provide sensitivity and
# exact calibration results.
# In external_validation/results/, gbmspace, greenwald and gse253080 supply patient_{ligand,spatial}_metrics.csv
# and representative_plot.csv with patient metrics and one representative section per dataset.
# gbmspace and cptac_gbm include analysis_summary.json. CPTAC also supplies ligand_tumor_normal.csv,
# tgfb2_plex_matched_contrasts.csv,
# tgfb2_tumor_associations.csv and tgfb2_sample_values.csv for sample protein values, intervals and correlations.
# Outputs in original_analysis/figures/ are Figure1–Figure6 and SupplementaryFigureS1–SupplementaryFigureS7.
# Each figure has .pdf, .png and _layout.json files. --tiff also writes Fig1.tif–Fig6.tif in RGB at 600 dpi with LZW
# compression.
# Run from original_analysis: MPLBACKEND=Agg python code/13_plot_submission_figures.py [--main-only] [--output DIR]
# [--tiff]
# Python retains the original Matplotlib panels and fonts. PDF checks and raster conversion use pypdf, pdftocairo and
# Pillow.
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from importlib import import_module
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

# The package contains both analyses. Locate it from this file so the command works from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
style = import_module("00a_figure_style")


def main(output: Path | None = None, *, main_only=False, tiff=False) -> None:
    """Draw the requested figures from the same frozen files. Stop if input or output checks fail. Returns None."""
    package_dir = Path(__file__).resolve().parents[1]
    data_dir = package_dir / "data"
    style.require_cohort_inputs(data_dir, supplementary=not main_only)
    external_results = package_dir / "external_validation" / "results"
    spatial_sources = ["gbmspace"]
    if not main_only:
        spatial_sources.extend(["greenwald", "gse253080"])
        style.require_inputs(
            data_dir,
            [
                "parameter_sensitivity_grid.csv",
                "parameter_sensitivity_cohort_results.csv",
                "exact_meta_calibration.csv",
                "exact_meta_null_tail.csv",
            ],
        )

    # The main figures need GBMSpace and CPTAC. The full set also needs the Greenwald and GSE253080 section caches.
    for source in spatial_sources:
        style.require_inputs(
            external_results / source,
            ["patient_ligand_metrics.csv", "patient_spatial_metrics.csv", "representative_plot.csv"],
        )
    style.require_inputs(external_results / "gbmspace", ["analysis_summary.json"])
    style.require_inputs(
        external_results / "cptac_gbm",
        [
            "analysis_summary.json",
            "ligand_tumor_normal.csv",
            "tgfb2_plex_matched_contrasts.csv",
            "tgfb2_tumor_associations.csv",
            "tgfb2_sample_values.csv",
        ],
    )

    # Both modes default to figures/ and share fonts, colors, PDF checks and PNG conversion.
    if output is None:
        output = package_dir / "figures"
    style.configure_plottie_style()
    save_submission = partial(style.save_figure, directory=output, layout_report=True)

    # Figure 1a scales point area by included patients. Read sampled regions directly from the cohort manifest.
    manifest_table = pd.read_csv(data_dir / "cohort_manifest.csv")
    manifest_table = manifest_table.set_index("cohort", verify_integrity=True).loc[style.cohort_order].reset_index()
    fig = plt.figure(figsize=(7.1, 4.35))
    grid_spec = GridSpec(
        2, 2, figure=fig, height_ratios=[1.0, 0.70], width_ratios=[1.32, 0.68], hspace=0.48, wspace=0.38
    )

    # Use each cohort's color in both the regional coverage and total cell-count panels.
    ax = fig.add_subplot(grid_spec[0, 0])
    regions = ["Tumor/core", "Edge", "Peritumoral"]
    y_positions = np.arange(len(manifest_table))[::-1]
    for y, row in zip(y_positions, manifest_table.itertuples()):
        present_set = set(str(row.regions).split("; "))
        present = [
            bool(present_set & {"Tumor", "Core"}),
            "Edge" in present_set,
            bool(present_set & {"Peritumoral", "Peripheral"}),
        ]
        active = np.flatnonzero(present)
        if len(active) > 1:
            ax.plot(active, np.repeat(y, len(active)), color=style.grid, linewidth=1.4)
        for x, available in enumerate(present):
            ax.scatter(
                x,
                y,
                s         = 28 + row.patients * 4 if available else 20,
                facecolor = style.cohort_colors[row.cohort] if available else "white",
                edgecolor = style.cohort_colors[row.cohort] if available else style.gray,
                linewidth = 0.7,
                zorder    = 2,
            )

        # N counts all included patients. Figure 2 gets the number with paired TGFB2 values separately from the contrast
        # table.
        ax.text(2.75, y, f"N={row.patients}", va="center", fontsize=6.5, color=style.muted)
    ax.set_xticks(range(len(regions)), regions)

    def cohort_display(cohort: str, *, short: bool = False) -> str:
        """Return the joint or single cohort's display name. Keep the input cohort code GSE274987."""
        label = "GSE274986/7" if cohort == "GSE274987" else cohort
        if short:
            label = label.removeprefix("GSE")
        return label

    ax.set_yticks(y_positions, [cohort_display(item) for item in manifest_table["cohort"]])
    ax.set_xlim(-0.42, 3.15)
    ax.set_ylim(-0.6, len(manifest_table) - 0.4)
    ax.set_title("Included cohorts and sampled regions", pad=6)
    style.clean_axes(ax)
    style.panel_label(ax, "a")

    # Figure 1b uses raw cell counts for x and labels its log-scale ticks in thousands.
    ax = fig.add_subplot(grid_spec[0, 1])
    for y, row in zip(y_positions, manifest_table.itertuples()):
        ax.scatter(
            row.included_cells,
            y,
            s         = 34,
            color     = style.cohort_colors[row.cohort],
            edgecolor = "white",
            linewidth = 0.5,
            zorder    = 3,
        )
        ax.text(
            row.included_cells * 1.08,
            y,
            f"{row.included_cells / 1000:.1f}k",
            va       = "center",
            fontsize = 6.2,
            color    = style.muted,
        )
    ax.set_xscale("log")
    ax.set_xlim(1.5e4, 1.55e5)
    tick_positions = [2e4, 5e4, 1e5]
    ax.set_xticks(tick_positions, [f"{value / 1000:.0f}k" for value in tick_positions])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_yticks(y_positions, [cohort_display(item, short=True) for item in manifest_table["cohort"]])
    ax.set_xlabel("Cell count (log scale)")
    ax.set_title("Included cells / nuclei", pad=6)
    style.open_axes(ax, grid_axis="x")
    style.panel_label(ax, "b")

    ax = fig.add_subplot(grid_spec[1, :])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # The last node in Figure 1c uses regional tests with observed q values. Label the case with no evaluable tests
    # separately.
    regional_table = pd.read_csv(data_dir / "regional_communication_tests.csv")
    regional_q = regional_table["q_value"].dropna()
    if regional_q.empty:
        regional_label = "No estimable q value"
    elif regional_q.lt(0.05).any():
        regional_label = "Regional evidence at q < 0.05"
    else:
        regional_label = "Statistically unresolved"
    lane_specs = [
        (
            0.70,
            ["AC-like vs other states", "Patient contrasts", "Cohort tests", "Aggregate estimate"],
            style.coral_light,
            style.coral,
        ),
        (
            0.25,
            ["Regional co-availability", "Paired patient effects", "Signed-rank test", regional_label],
            style.teal_light,
            style.teal,
        ),
    ]
    xs = [0.14, 0.38, 0.62, 0.86]
    for y, labels, pale, accent in lane_specs:
        for index, (x, label) in enumerate(zip(xs, labels)):
            at_end = index in (0, len(labels) - 1)
            patch = FancyBboxPatch(
                (x - 0.09, y - 0.115),
                0.18,
                0.23,
                boxstyle  = "round,pad=0.012,rounding_size=0.025",
                facecolor = pale if at_end else style.pale,
                edgecolor = accent if at_end else style.gray,
                linewidth = 0.7,
            )
            ax.add_patch(patch)
            ax.text(x, y, label, ha="center", va="center", fontsize=6.6, color=style.ink)
            if index < len(xs) - 1:
                ax.annotate(
                    "",
                    xy = (xs[index + 1] - 0.105, y),
                    xytext = (x + 0.105, y),
                    arrowprops = dict(arrowstyle="-|>", color=style.gray_dark, linewidth=0.75),
                )
    style.panel_label(ax, "c")
    fig.text(
        0.11,
        0.45,
        "N: included cohort patients; eligible TGFB2 patients are shown in Fig 2a",
        fontsize = 6.2,
        color    = style.muted,
    )
    fig.subplots_adjust(left=0.11, right=0.95, top=0.94, bottom=0.07)
    save_submission(fig, stem="Figure1", panels="abc", tiff=tiff)

    # Figure 2a expands each cohort–ligand record into patient pairs. The thick line joins the group medians.
    contrast_table = pd.read_csv(data_dir / "ac_vs_other_ligand_contrasts.csv")
    contrast_index_table = contrast_table.set_index(["cohort", "gene"], verify_integrity=True)
    inferential = ["GSE211376", "GSE273274", "GSE274987", "GSE286419"]
    fig = plt.figure(figsize=(7.1, 5.7))
    grid_spec = GridSpec(
        2, 2, figure=fig, height_ratios=[0.92, 1.12], width_ratios=[1.35, 0.78], hspace=0.70, wspace=0.42
    )
    top = GridSpecFromSubplotSpec(1, len(inferential), subplot_spec=grid_spec[0, :], wspace=0.32)
    for index, cohort in enumerate(inferential):
        ax = fig.add_subplot(top[0, index])
        contrast_row = contrast_index_table.loc[(cohort, "TGFB2"), :]
        paired_table = style.paired_expression_table(contrast_row)
        for item in paired_table.itertuples():
            ax.plot([0, 1], [item.other_mean, item.ac_mean], color=style.gray, linewidth=0.65, alpha=0.8, zorder=1)
            ax.scatter(
                [0, 1],
                [item.other_mean, item.ac_mean],
                s         = 16,
                color     = [style.state_colors["Other malignant"], style.state_colors["AC-like"]],
                edgecolor = "white",
                linewidth = 0.4,
                zorder    = 2,
            )

        # Join both observations from each patient. The thick black line connects the two group medians.
        ax.plot(
            [0, 1],
            [paired_table["other_mean"].median(), paired_table["ac_mean"].median()],
            color     = style.ink,
            linewidth = 2.0,
            zorder    = 3,
        )
        ax.set_xticks([0, 1], ["Other", "AC-like"])
        ax.set_title(f"{cohort_display(cohort, short=True)}  n={len(paired_table)}", fontsize=7.4)

        # Use the same y limits across cohorts and label each top-row panel with its number of paired patients.
        ax.set_xlim(-0.24, 1.24)
        ax.set_ylim(-0.025, 1.08)
        style.open_axes(ax, grid_axis="y")
        if index == 0:
            ax.set_ylabel("Mean log-normalized TGFB2")
            style.panel_label(ax, "a")
        else:
            ax.set_ylabel("")

    # Figure 2b scales area by eligible patients and color by difference. A dark border marks a cohort test with q <
    # 0.05.
    ax = fig.add_subplot(grid_spec[1, 0])
    gene_order = style.contrast_ligands
    cohort_order = style.cohort_order
    gene_positions = {gene: i for i, gene in enumerate(gene_order)}
    cohort_positions = {cohort: i for i, cohort in enumerate(cohort_order[::-1])}
    max_abs = float(np.nanmax(np.abs(contrast_table["median_difference"])))
    norm = mpl.colors.Normalize(-max_abs, max_abs)
    for row in contrast_table.itertuples():
        x, y = gene_positions[row.gene], cohort_positions[row.cohort]
        significant = pd.notna(row.q_value) and row.q_value < 0.05
        ax.scatter(
            x,
            y,
            s         = 24 + 6 * row.n_patients,
            color     = style.diverging(norm(row.median_difference)),
            edgecolor = style.ink if significant else style.gray,
            linewidth = 0.75 if significant else 0.4,
        )

    # Columns represent the six ligands and rows represent cohorts. The points and color bar share their center and
    # limits.
    ax.set_xticks(range(len(gene_order)), gene_order, rotation=32, ha="right")
    ax.set_yticks(range(len(cohort_order)), [cohort_display(item) for item in cohort_order[::-1]])
    ax.set_title("AC-like minus other malignant states", pad=6)
    ax.set_xlabel("Ligand")
    style.clean_axes(ax)
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=style.diverging), ax=ax, fraction=0.035, pad=0.025)
    colorbar.outline.set_visible(False)
    colorbar.set_label("Median expression difference", fontsize=6.5)
    style.panel_label(ax, "b")
    handles = []
    observed_counts = np.unique(contrast_table["n_patients"].dropna())
    example_counts = np.quantile(observed_counts, [0, 0.5, 1], method="nearest")
    for n in np.unique(example_counts):

        # The area legend uses the lowest, middle and highest observed patient counts. Each point uses its row's count.
        handles.append(ax.scatter([], [], s=24 + 6 * n, facecolor=style.pale, edgecolor=style.gray, label=f"n={n}"))
    handles.append(ax.scatter([], [], s=42, facecolor=style.pale, edgecolor=style.ink, label="q < 0.05"))
    ax.legend(
        handles        = handles,
        loc            = "lower left",
        bbox_to_anchor = (0, 1.16),
        ncol           = 4,
        frameon        = False,
        fontsize       = 5.6,
        handletextpad  = 0.3,
        columnspacing  = 0.5,
        borderaxespad  = 0,
    )

    ax = fig.add_subplot(grid_spec[1, 1])

    # Figure 2c uses coral for BH q < 0.05 on the largest leave-one-cohort-out P. Label each point with the full-model
    # q.
    meta_table = pd.read_csv(data_dir / "meta_ligand_results.csv")
    ordered_meta_table = meta_table.set_index("gene").loc[gene_order].reset_index()
    y_positions = np.arange(len(ordered_meta_table))[::-1]
    for y, row in zip(y_positions, ordered_meta_table.itertuples()):
        robust = row.loo_q_value < 0.05
        color = style.coral if robust else style.gray_dark
        ax.plot([0, row.meta_z], [y, y], color=style.grid, linewidth=1.4)
        ax.scatter(row.meta_z, y, s=42, color=color, edgecolor="white", linewidth=0.55, zorder=3)
        ax.text(4.05, y, f"q={row.q_value:.2g}", va="center", fontsize=6.1, color=color)

    # All ligands share the Stouffer z scale. Read the full-model q labels directly from the frozen results.
    ax.set_xlim(-0.1, 5.5)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.axvline(0, color=style.gray_dark, linewidth=0.65)
    ax.set_yticks(y_positions, ordered_meta_table["gene"])
    ax.set_xlabel("Signed Stouffer z")
    ax.set_title("Cross-cohort evidence", pad=6)
    style.open_axes(ax)
    style.panel_label(ax, "c")
    ax.text(0, 1.16, "Coral: leave-one-cohort-out q < 0.05", transform=ax.transAxes, fontsize=5.7, color=style.coral)
    fig.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.11)
    save_submission(fig, stem="Figure2", panels="abc", tiff=tiff)

    # Figure 3a uses malignant cells in the patient and region as the denominator. Join patients observed in both
    # regions.
    fraction_table = pd.read_csv(data_dir / "patient_ac_fraction.csv")
    fig = plt.figure(figsize=(7.1, 4.90))
    grid_spec = GridSpec(
        2, 3, figure=fig, height_ratios=[0.92, 1.05], width_ratios=[1.0, 1.0, 0.88], hspace=0.53, wspace=0.40
    )
    for index, cohort in enumerate(["GSE273274", "GSE276177", "GSE286419"]):
        ax = fig.add_subplot(grid_spec[0, index])
        cohort_table = fraction_table[fraction_table["cohort"] == cohort]
        region_table = cohort_table.pivot_table(
            index="patient", columns="region", values="fraction_of_malignant", aggfunc="mean"
        )
        region_table = region_table.reindex(columns=["Peritumoral", "Tumor"]).dropna()
        for _, row in region_table.iterrows():
            ax.plot([0, 1], row[["Peritumoral", "Tumor"]], color=style.gray, linewidth=0.7)
            ax.scatter(
                [0, 1],
                row[["Peritumoral", "Tumor"]],
                s         = 17,
                color     = [style.region_colors["Peritumoral"], style.region_colors["Tumor"]],
                edgecolor = "white",
                linewidth = 0.4,
                zorder    = 2,
            )

        # The thick line joins the patient medians in the two regions.
        if len(region_table):
            ax.plot(
                [0, 1],
                [region_table["Peritumoral"].median(), region_table["Tumor"].median()],
                color = style.ink,
                linewidth = 2.0,
            )
        ax.set_xticks([0, 1], ["Peritumoral", "Tumor"], rotation=22, ha="right")
        ax.set_ylim(-0.04, 1.04)
        ax.set_title(f"{cohort_display(cohort, short=True)}  n={len(region_table)}")
        style.open_axes(ax, grid_axis="y")
        if index == 0:
            ax.set_ylabel("AC-like fraction of malignant cells")
            style.panel_label(ax, "a")

    # Figure 3b shows the three-region trajectories by patient, preserving missing observations.
    ax = fig.add_subplot(grid_spec[1, :2])
    trajectory_ax = ax
    cohort_table = fraction_table[fraction_table["cohort"] == "GSE274987"]
    region_table = cohort_table.pivot_table(
        index="patient", columns="region", values="fraction_of_malignant", aggfunc="mean"
    )
    region_order = ["Core", "Edge", "Peripheral"]
    for _, row in region_table.iterrows():
        values = row.reindex(region_order).to_numpy(float)
        valid = np.isfinite(values)
        ax.plot(np.arange(len(region_order))[valid], values[valid], color=style.gray, linewidth=0.7)
        for index, value in enumerate(values):
            if np.isfinite(value):
                ax.scatter(
                    index,
                    value,
                    s         = 18,
                    color     = style.region_colors[region_order[index]],
                    edgecolor = "white",
                    linewidth = 0.4,
                    zorder    = 2,
                )

    # Calculate each region's median and number observed separately. Complete two-region pairs have a different
    # denominator.
    medians = [region_table.get(region, pd.Series(dtype=float)).median() for region in region_order]
    ax.plot(range(len(region_order)), medians, color=style.ink, linewidth=2.0, marker="D", markersize=3.5)
    ax.set_xticks(
        range(len(region_order)), [f"{region}\nn={region_table[region].notna().sum()}" for region in region_order]
    )
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel("AC-like fraction of malignant cells")
    ax.set_title("274986/7 regional trajectories", pad=6)
    style.open_axes(ax, grid_axis="y")
    style.panel_label(ax, "b")

    # Figure 3c uses the frozen comparison direction. Set the jitter seed once and draw subsequent panels in the
    # original order.
    ax = fig.add_subplot(grid_spec[1, 2])
    labels = {
        "GSE273274": "273274 T-P",
        "GSE276177": "276177 T-P",
        "GSE286419": "286419 T-P",
        "GSE274987_core_peripheral": "274986/7 C-P",
        "GSE274987_core_edge": "274986/7 C-E",
    }
    test_table = pd.read_csv(data_dir / "regional_ac_fraction_tests.csv")
    comparison_table = test_table.set_index("comparison").loc[list(labels)].reset_index()
    y_positions = np.arange(len(comparison_table))[::-1]
    rng = np.random.default_rng(20260717)
    for y, row in zip(y_positions, comparison_table.itertuples()):
        differences = np.asarray(json.loads(row.patient_differences), dtype=float)
        ax.scatter(
            differences,
            np.repeat(y, len(differences)) + rng.normal(0, 0.045, len(differences)),
            s         = 16,
            color     = style.cohort_colors[row.cohort],
            alpha     = 0.78,
            edgecolor = "white",
            linewidth = 0.35,
        )

        # Plot raw patient differences on x and show the frozen median as a diamond. Only the y positions use jitter.
        ax.scatter(
            row.median_difference, y, s=38, marker="D", color=style.ink, edgecolor="white", linewidth=0.5, zorder=3
        )

    # Use the patient count from the corresponding comparison row in each label.
    ax.axvline(0, color=style.gray_dark, linewidth=0.65, linestyle="--")
    ax.set_yticks(
        y_positions, [f"{labels[row.comparison]}  n={int(row.n_patients)}" for row in comparison_table.itertuples()]
    )
    ax.set_xlabel("First region minus second")
    ax.set_title("Paired effects", pad=6)
    style.open_axes(ax)
    style.panel_label(ax, "c")
    fig.subplots_adjust(left=0.11, right=0.96, top=0.95, bottom=0.14)
    trajectory_ax.set_position([0.11, 0.18, 0.48, 0.34])
    ax.set_position([0.79, 0.18, 0.17, 0.34])
    ax.set_xticks([0, 0.25, 0.50])
    fig.text(0.11, 0.025, "T: tumor; C: core; E: edge; P: peritumoral / peripheral", fontsize=6.2, color=style.muted)
    save_submission(fig, stem="Figure3", panels="abc", tiff=tiff)

    # Figure 4a obtains patient counts from the ligand–receptor rows in each comparison. Show a range if they differ.
    comparisons = list(labels)
    comparison_labels = list(labels.values())
    pairs = ["TGFB2-TGFBR1", "TGFB2-TGFBR2", "TGFB1-TGFBR1", "ANXA1-FPR1", "SPP1-CD44", "CSF1-CSF1R", "CCL2-CCR2"]
    regional_index_table = regional_table.set_index(["comparison", "pair"], verify_integrity=True)
    fig = plt.figure(figsize=(7.1, 5.65))
    grid_spec = GridSpec(
        2, 2, figure=fig, height_ratios=[0.88, 1.12], width_ratios=[1.18, 0.82], hspace=0.60, wspace=0.52
    )
    ax = fig.add_subplot(grid_spec[0, :])
    max_abs = float(np.nanmax(np.abs(regional_table["median_difference"])))
    norm = mpl.colors.Normalize(-max_abs, max_abs)
    for i, comparison in enumerate(comparisons):
        y = len(comparisons) - 1 - i
        cohort_table = regional_index_table.loc[comparison].loc[pairs]
        for x, pair in enumerate(pairs):
            row = cohort_table.loc[pair]
            ax.scatter(
                x,
                y,
                s         = 22 + 130 * abs(row.median_difference) / max_abs if max_abs else 22,
                color     = style.diverging(norm(row.median_difference)),
                edgecolor = style.gray,
                linewidth = 0.35,
            )
        counts = cohort_table["n_patients"]
        pair_count = str(int(counts.min()))
        if counts.min() != counts.max():
            pair_count = f"{counts.min():.0f}–{counts.max():.0f}"
        ax.text(6.8, y, pair_count, fontsize=6.1, ha="center", va="center", color=style.muted)
        ax.text(
            7.65,
            y,
            f"{cohort_table.minimum_attainable_p.min():.3g}",
            fontsize = 6.1,
            ha       = "center",
            va       = "center",
            color    = style.muted,
        )

    # For this row, use the lowest minimum attainable P across ligand–receptor pairs. The observed P is a separate
    # statistic.
    ax.text(6.8, 4.6, "Pairs", ha="center", fontsize=6.1, color=style.muted)
    ax.text(7.65, 4.6, "Min. P", ha="center", fontsize=6.1, color=style.muted)
    ax.set_xticks(range(len(pairs)), [x.replace("-", "\n") for x in pairs])
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_yticks(range(len(comparisons)), comparison_labels[::-1])
    ax.set_xlim(-0.5, 8.2)
    ax.set_ylim(-0.55, 4.9)
    ax.set_title("Regional ligand-receptor co-availability", pad=6)
    style.clean_axes(ax)
    style.panel_label(ax, "a")

    # Keep the color bar in its original axes position. Its units are the median regional difference.
    colorbar_axis = fig.add_axes([0.925, 0.647, 0.012, 0.235])
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=style.diverging), cax=colorbar_axis)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=5.6)
    colorbar.set_label("Median difference", fontsize=6, labelpad=2)
    ax = fig.add_subplot(grid_spec[1, 0])

    # Figure 4b uses the two specified TGFB2 receptors. Show each patient's frozen difference with evenly spaced
    # vertical offsets.
    for i, comparison in enumerate(comparisons):
        for offset, pair, point_color in [(0.15, "TGFB2-TGFBR1", style.coral), (-0.15, "TGFB2-TGFBR2", style.teal)]:
            row = regional_index_table.loc[(comparison, pair), :]
            values = np.asarray(json.loads(row.patient_differences), float)
            y = len(comparisons) - 1 - i + offset
            ax.scatter(
                values,
                y + np.linspace(-0.035, 0.035, len(values)),
                s         = 13,
                color     = point_color,
                alpha     = 0.7,
                edgecolor = "white",
                linewidth = 0.3,
            )
            ax.scatter(
                row.median_difference,
                y,
                s         = 28,
                marker    = "D",
                color     = point_color,
                edgecolor = style.ink,
                linewidth = 0.55,
                zorder    = 4,
            )
    ax.axvline(0, color=style.gray_dark, lw=0.65, ls="--")
    ax.set_yticks(range(len(comparisons)), comparison_labels[::-1])
    ax.set_ylim(-0.5, 4.55)
    ax.set_xlabel("Patient regional difference")
    ax.set_title("TGFB2 patient-level effects", pad=6)
    handles = []
    for color, receptor in [(style.coral, "TGFBR1"), (style.teal, "TGFBR2")]:
        handles.append(
            Line2D([], [], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", label=receptor)
        )
    ax.legend(
        handles=handles, loc="lower left", bbox_to_anchor=(0, 1.15), ncol=2, frameon=False, fontsize=6, borderaxespad=0
    )
    style.open_axes(ax)
    style.panel_label(ax, "b")

    # Figure 4c plots the minimum regional q against the pooled intrinsic q for each ligand. Offsets only position the
    # labels.
    ax = fig.add_subplot(grid_spec[1, 1])
    best = regional_table.assign(gene=regional_table.pair.str.split("-").str[0]).groupby("gene").q_value.min()
    for row in meta_table.itertuples():
        x = -np.log10(max(row.q_value, 1e-12))
        y = -np.log10(max(best[row.gene], 1e-12))
        color = style.coral if row.gene == "TGFB2" else style.gray_dark
        ax.scatter(x, y, s=34, color=color, edgecolor="white", linewidth=0.5)
        offsets = {"TGFB2": (-3, 6), "TGFB1": (4, -7), "CCL2": (4, 5), "ANXA1": (4, 5), "CSF1": (4, 5), "SPP1": (4, 5)}
        ax.annotate(
            row.gene,
            (x, y),
            xytext     = offsets[row.gene],
            textcoords = "offset points",
            fontsize   = 6,
            color      = color,
            ha         = "right" if row.gene == "TGFB2" else "left",
        )

    # Both reference lines mark q=0.05. The negative logarithm is for display.
    cut = -np.log10(0.05)
    ax.axvline(cut, color=style.gray_dark, lw=0.65, ls=":")
    ax.axhline(cut, color=style.gray_dark, lw=0.65, ls=":")
    ax.set_xlim(0, 3.45)
    ax.set_ylim(0, 3.15)
    ax.set_xlabel("Intrinsic evidence, −log10(q)")
    ax.set_ylabel("Best regional evidence, −log10(q)")
    ax.set_title("Evidence across analyses", pad=6)
    style.open_axes(ax)
    style.panel_label(ax, "c")
    fig.subplots_adjust(left=0.14, right=0.90, top=0.92, bottom=0.10)
    fig.text(
        0.14,
        0.022,
        "T: tumor; C: core; E: edge; P: peritumoral / peripheral. Diamonds: medians.",
        fontsize = 6,
        color    = style.muted,
    )
    save_submission(fig, stem="Figure4", panels="abc", tiff=tiff)

    # Figure 5a joins raw and adjusted correlations for the same GBMSpace patient. Read statistics from the GBMSpace
    # summary.
    result_dir = external_results / "gbmspace"
    ligand_table = pd.read_csv(result_dir / "patient_ligand_metrics.csv")
    spatial_table = pd.read_csv(result_dir / "patient_spatial_metrics.csv")
    section_table = pd.read_csv(result_dir / "representative_plot.csv")
    summary = json.loads((result_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    tgfb2_table = ligand_table[ligand_table.ligand == "TGFB2"]
    tgfb2_summary = summary["tgfb2"]
    fig = plt.figure(figsize=(7.1, 6.05))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[0.95, 1.1], hspace=0.48)
    top = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0], width_ratios=[1.42, 1.55, 1, 1], wspace=0.80)

    # Draw the raw column and then the adjusted column using the same random stream.
    ax = fig.add_subplot(top[0, 0])
    raw_values = tgfb2_table.rho_ac_fraction.to_numpy(float)
    adjusted_values = tgfb2_table.rho_ac_fraction_adjusted.to_numpy(float)
    for raw_value, adjusted_value in zip(raw_values, adjusted_values):
        if np.isfinite(raw_value) and np.isfinite(adjusted_value):
            ax.plot([0, 1], [raw_value, adjusted_value], color=style.grid, lw=0.75)
    for x, values, point_color in [(0, raw_values, style.teal), (1, adjusted_values, style.blue)]:
        ax.scatter(
            x + rng.normal(0, 0.025, len(values)), values, s=17, color=point_color, edgecolor="white", linewidth=0.35
        )

    # Use the frozen estimates for the horizontal lines and significance labels before and after adjustment.
    style.median_bar(ax, 0, tgfb2_summary["rho_ac_fraction_median"], color=style.coral)
    style.median_bar(ax, 1, tgfb2_summary["rho_ac_fraction_adjusted_median"], color=style.coral)
    ax.axhline(0, color=style.gray_dark, lw=0.65, ls="--")
    ax.set_xticks([0, 1], ["Raw", "Adjusted"])
    ax.tick_params(axis="x", labelsize=6.1)
    ax.set_ylim(-0.16, 0.225)
    ax.set_ylabel("Patient median rho")
    ax.set_title("TGFB2–AC\nassociation", fontsize=7.6, pad=6)
    association_note = (
        f"n={int(tgfb2_summary['n_patients'])}\nq={tgfb2_summary['rho_ac_fraction_signflip_q']:.2g} / "
        f"{tgfb2_summary['rho_ac_fraction_adjusted_signflip_q']:.3f}"
    )
    ax.text(0.02, 0.98, association_note, transform=ax.transAxes, va="top", fontsize=5.8, color=style.muted)
    style.open_axes(ax)
    style.panel_label(ax, "a")

    # Figure 5b has its own prespecified ligand order. Each patient–ligand combination occupies one cell.
    spatial_ligands = ["TGFB2", "TGFB1", "ANXA1", "SPP1", "CSF1", "CCL2"]
    ax = fig.add_subplot(top[0, 1])
    adjusted_ligand_table = ligand_table.pivot(index="patient", columns="ligand", values="rho_ac_fraction_adjusted")
    adjusted_ligand_table = adjusted_ligand_table.reindex(columns=spatial_ligands)
    for j, ligand in enumerate(spatial_ligands):
        values = adjusted_ligand_table[ligand].dropna().to_numpy(float)
        point_color = style.coral if ligand == "TGFB2" else style.gray_dark
        ax.scatter(
            j + rng.normal(0, 0.045, len(values)),
            values,
            s         = 12,
            color     = point_color,
            alpha     = 0.7,
            edgecolor = "white",
            linewidth = 0.25,
        )
        style.median_bar(
            ax,
            j,
            float(np.median(values)),
            color = point_color if ligand == "TGFB2" else style.ink,
            width = 0.35,
            linewidth = 1.5,
        )
    ax.axhline(0, color=style.gray_dark, lw=0.65, ls="--")
    ax.set_xticks(range(len(spatial_ligands)), spatial_ligands, rotation=55, ha="right")
    ax.tick_params(axis="x", labelsize=5.8)
    ax.set_ylabel("Adjusted patient median rho")
    ax.set_title("Fixed ligand\npanel", fontsize=7.6, pad=6)
    style.open_axes(ax)
    style.panel_label(ax, "b")

    # Figure 5c and later spatial panels reuse the strip plot. Compute medians from finite values for independent
    # patients.
    ax = fig.add_subplot(top[0, 2])

    def add_strip(ax, values, *, title, ylabel, color=style.teal) -> None:
        """Plot patient values and their median on the original axes, using this run's jitter stream. Returns None."""
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        jitter = rng.normal(0, 0.035, len(values))
        ax.scatter(jitter, values, s=22, color=color, alpha=0.82, edgecolor="white", linewidth=0.45, zorder=3)
        if len(values):
            style.median_bar(ax, 0, float(np.median(values)), color=style.coral)
        ax.axhline(0, color=style.gray_dark, linewidth=0.65, linestyle="--", zorder=1)
        ax.set_xlim(-0.27, 0.27)
        ax.set_xticks([])
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=5)
        ax.spines["bottom"].set_visible(False)
        style.open_axes(ax)

    add_strip(ax, tgfb2_table.mean_top_minus_bottom_ac_delta, title="AC-rich\ncontrast", ylabel="TGFB2 difference")
    ax.set_title("AC-rich\ncontrast", fontsize=7.6, pad=6)
    ax.yaxis.label.set_size(6.5)
    ax.tick_params(labelsize=6)
    style.panel_label(ax, "c")
    ax = fig.add_subplot(top[0, 3])
    add_strip(
        ax,
        spatial_table.tgfb2_neighbor_receiver_delta,
        title  = "Neighbor\nreceiver",
        ylabel = "Receiver-score difference",
        color  = style.purple,
    )
    ax.set_title("Neighbor\nreceiver", fontsize=7.6, pad=6)
    ax.yaxis.label.set_size(6.5)
    ax.tick_params(labelsize=6)
    ax.set_ylim(-0.005, 0.026)
    proximity = summary["proximity"]["tgfb2_neighbor_receiver_delta"]
    ax.text(
        0.04,
        0.98,
        f"n={proximity['n']}\nP={proximity['signflip_p']:.3f}",
        transform = ax.transAxes,
        va        = "top",
        fontsize  = 5.8,
        color     = style.muted,
    )
    style.panel_label(ax, "d")

    # Figure 5e–g share the same section and orientation. The 1st and 99th percentiles set color limits only. Retain
    # every point.
    bottom = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.48)
    specs = [
        ("ac_fraction", "AC-like fraction", style.teal_seq),
        ("tgfb2", "TGFB2 expression", style.coral_seq),
        ("receiver", "Myeloid × TGFBR", style.purple_seq),
    ]
    colorbar_axes = []
    for i, (field, title, cmap) in enumerate(specs):
        inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=bottom[0, i], width_ratios=[1, 0.05], wspace=0.08)
        ax = fig.add_subplot(inner[0, 0])
        values = pd.to_numeric(section_table[field], errors="coerce").to_numpy(float)
        valid = np.isfinite(values)
        if not valid.any():
            raise ValueError(f"The representative section has no plottable values for {field}")
        lo, hi = np.quantile(values[valid], [0.01, 0.99])
        scatter = ax.scatter(
            section_table.x, -section_table.y, c=values, s=3, cmap=cmap, vmin=lo, vmax=hi, linewidths=0, rasterized=True
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        style.clean_axes(ax)
        ax.set_title(title, fontsize=7.6, pad=6)
        style.panel_label(ax, chr(ord("e") + i))
        colorbar_axis = fig.add_subplot(inner[0, 1])
        colorbar = fig.colorbar(scatter, cax=colorbar_axis)
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(labelsize=5.5, width=0.45, length=1.8)
        colorbar_axes.append((ax, colorbar_axis))

    # Align color bar heights with the final spatial axes while retaining point coordinates and y-axis orientation.
    fig.subplots_adjust(left=0.075, right=0.95, top=0.885, bottom=0.075)
    fig.canvas.draw()
    for spatial_ax, colorbar_axis in colorbar_axes:
        pos = colorbar_axis.get_position()
        image_pos = spatial_ax.get_position()
        colorbar_axis.set_position([pos.x0, image_pos.y0, pos.width, image_pos.height])
    section_codes = section_table["sample"].drop_duplicates()
    if len(section_codes) != 1 or section_codes.isna().any():
        raise ValueError("The GBMSpace representative plot must contain one identified section")
    section_label = section_codes.item()
    fig.text(
        0.075,
        0.018,
        f"Representative section: {section_label}; identical coordinates and orientation in e–g.",
        fontsize = 6.1,
        color    = style.muted,
    )
    save_submission(fig, stem="Figure5", panels="abcdefg", tiff=tiff)

    # Figure 6a uses independent tumor and normal samples. Plot original protein values and check group counts against
    # the frozen table.
    result_dir = external_results / "cptac_gbm"
    protein_table = pd.read_csv(result_dir / "ligand_tumor_normal.csv")
    plex_table = pd.read_csv(result_dir / "tgfb2_plex_matched_contrasts.csv")
    association_table = pd.read_csv(result_dir / "tgfb2_tumor_associations.csv")
    sample_table = pd.read_csv(result_dir / "tgfb2_sample_values.csv")
    summary = json.loads((result_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    tgfb2_row = protein_table.set_index("gene").loc["TGFB2"]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.25))
    ax = axes[0, 0]
    for x, group, point_color in [(0, "tumor", style.coral), (1, "normal", style.teal)]:
        values = sample_table.loc[sample_table.group == group, "tgfb2_log2_ratio"].to_numpy(float)
        if np.isfinite(values).sum() != tgfb2_row[f"n_{group}"]:
            raise ValueError(f"The measured CPTAC {group} sample count differs from the statistics table")
        ax.scatter(
            x + rng.uniform(-0.11, 0.11, len(values)),
            values,
            s         = 12 if group == "tumor" else 20,
            color     = point_color,
            alpha     = 0.7,
            edgecolor = "white",
            linewidth = 0.3,
        )
        style.median_bar(ax, x, tgfb2_row[group + "_median"], color=style.ink, width=0.38)
    ax.set_xticks([0, 1], [f"GBM tumor\nn={int(tgfb2_row.n_tumor)}", f"Normal brain\nn={int(tgfb2_row.n_normal)}"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("Relative TGFB2 protein (log2 ratio)")
    ax.set_title("Tissue-level TGFB2", pad=6)
    ax.text(
        0.02,
        0.97,
        f"Median delta = {tgfb2_row.median_difference:.2f}",
        transform = ax.transAxes,
        va        = "top",
        fontsize  = 6.1,
        color     = style.muted,
    )

    # Leave room at the top of the axes for the annotation while retaining the observed point values.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.18 * (ymax - ymin))
    style.open_axes(ax, grid_axis="y")
    style.panel_label(ax, "a")

    # Each Figure 6b row shows one ligand's tumor-minus-normal median difference and its 95% interval.
    ax = axes[0, 1]
    ordered_protein_table = protein_table.set_index("gene").loc[spatial_ligands].reset_index()
    for y, row in zip(range(len(spatial_ligands) - 1, -1, -1), ordered_protein_table.itertuples()):
        point_color = style.coral if row.gene == "TGFB2" else style.gray_dark
        ax.plot([row.median_difference_ci_low, row.median_difference_ci_high], [y, y], color=point_color, lw=1.2)
        ax.scatter(row.median_difference, y, s=32, color=point_color, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(0, color=style.gray_dark, lw=0.65, ls="--")
    ax.set_yticks(range(len(spatial_ligands) - 1, -1, -1), spatial_ligands)
    ax.set_xlabel("Tumor minus normal median log2 ratio", fontsize=6.8)
    ax.set_title("Ligand contrasts, 95% CI", pad=6)
    style.open_axes(ax)
    style.panel_label(ax, "b")

    # Figure 6c plots the difference between group medians within each TMT plex. Read P from the original exact plex
    # test.
    ax = axes[1, 0]
    plex_table = plex_table.sort_values("tmt_plex")
    y = np.arange(len(plex_table))[::-1]
    ax.scatter(plex_table.median_difference, y, s=25, color=style.teal, edgecolor="white", linewidth=0.4)
    ax.set_yticks(y, [f"Plex {v}" for v in plex_table.tmt_plex])
    ax.axvline(0, color=style.gray_dark, lw=0.65, ls="--")
    ax.set_xlabel("Tumor minus normal TGFB2 median", fontsize=6.8)
    ax.set_title("Within-plex contrasts", pad=6)
    ax.text(
        0.04,
        0.98,
        f"Exact P={summary['tgfb2_plex_matched']['exact_signflip_p']:.3g}",
        transform = ax.transAxes,
        va        = "top",
        fontsize  = 6.1,
        color     = style.muted,
    )
    ax.set_ylim(-0.6, len(plex_table) + 0.2)
    style.open_axes(ax, grid_axis="x")
    style.panel_label(ax, "c")

    # Figure 6d orders scores by tumor-sample correlation. Point color uses the original Spearman q.
    ax = axes[1, 1]
    names = {
        "astrocytic_protein_score": "Astrocytic",
        "myeloid_protein_score": "Myeloid",
        "tgfbr_protein_score": "TGFBR",
    }
    ordered_score_table = association_table.assign(label=association_table.score.map(names)).sort_values("spearman_rho")
    for y, row in enumerate(ordered_score_table.itertuples()):
        point_color = style.coral if row.spearman_q < 0.05 else style.gray_dark
        ax.plot([0, row.spearman_rho], [y, y], color=style.grid, lw=1.3)
        ax.scatter(row.spearman_rho, y, s=32, color=point_color, edgecolor="white", linewidth=0.5)
        ax.text(0.235, y, f"q={row.spearman_q:.2g}", va="center", fontsize=6.2, color=style.muted)
    ax.axvline(0, color=style.gray_dark, lw=0.65)
    ax.set_yticks(range(len(ordered_score_table)), ordered_score_table.label)
    ax.set_xlim(-0.16, 0.36)
    ax.set_xticks([-0.1, 0, 0.1, 0.2])
    ax.set_xlabel("Spearman rho with TGFB2 protein", fontsize=6.8)
    ax.set_title("Tumor protein-score associations", pad=6)
    style.open_axes(ax)
    style.panel_label(ax, "d")
    fig.subplots_adjust(left=0.13, right=0.955, top=0.94, bottom=0.11, hspace=0.55, wspace=0.52)
    save_submission(fig, stem="Figure6", panels="abcd", tiff=tiff)

    # Return here for --main-only. The full run reads the supplementary inputs and draws their random values.
    if main_only:
        return

    # Each S1 row is a patient–region combination. Compare cell states within rows and retain points below the
    # eligibility threshold.
    expression_table = pd.read_csv(data_dir / "patient_state_expression.csv")
    state_order = ["AC-like", "MES-like", "OPC-like", "NPC-like", "Myeloid"]
    expression_table = expression_table[expression_table["state"].isin(state_order)].copy()
    expression_table["cohort_order"] = expression_table["cohort"].map(
        {cohort: index for index, cohort in enumerate(style.cohort_order)}
    )
    sample_table = expression_table[["cohort", "cohort_order", "patient", "region"]].drop_duplicates()
    sample_table = sample_table.sort_values(["cohort_order", "region", "patient"]).reset_index(drop=True)
    row_lookup = {(row.cohort, row.patient, row.region): index for index, row in sample_table.iterrows()}

    # Scale canvas height to the number of patient–region rows using the original minimum and maximum page heights.
    height = min(10.3, max(7.4, 1.5 + 0.13 * len(sample_table)))
    fig, ax = plt.subplots(figsize=(7.1, height))
    for row in expression_table.itertuples():
        y = len(sample_table) - 1 - row_lookup[(row.cohort, row.patient, row.region)]
        x = state_order.index(row.state)
        eligible = row.n_cells >= 20
        ax.scatter(
            x,
            y,
            s         = 8 + 14 * np.log10(max(row.n_cells, 1)),
            facecolor = style.cohort_colors[row.cohort] if eligible else "white",
            edgecolor = style.cohort_colors[row.cohort],
            linewidth = 0.65,
        )
    labels = []
    short_regions = {"Tumor": "T", "Core": "C", "Edge": "E", "Peritumoral": "P", "Peripheral": "P"}
    for row in sample_table.itertuples():
        patient_id = str(row.patient).replace(f"{row.cohort}_", "")
        region = short_regions.get(row.region, row.region)
        labels.append(f"{cohort_display(row.cohort, short=True)}  {patient_id} {region}")
    ax.set_xticks(range(len(state_order)), ["AC", "MES", "OPC", "NPC", "Myeloid"])
    ax.set_yticks(range(len(sample_table)), labels[::-1])
    ax.tick_params(axis="y", labelsize=5.5)
    ax.set_xlim(-0.55, len(state_order) - 0.45)
    ax.set_ylim(-0.8, len(sample_table) - 0.2)

    # Use cumulative counts of the sorted cohort rows for separators. The legend explains filled and open points.
    boundaries = sample_table.groupby("cohort_order", sort=True).size().cumsum().to_numpy()[:-1]
    for boundary in boundaries:
        ax.axhline(len(sample_table) - boundary - 0.5, color=style.grid, linewidth=0.8)
    ax.set_title("Patient-region state counts and predefined eligibility", pad=7)
    style.clean_axes(ax)
    handles = []
    for face, label in [(style.teal, "Eligible, n>=20"), ("white", "Below threshold")]:
        handle = Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=face, markeredgecolor=style.teal, label=label
        )
        handles.append(handle)
    ax.legend(handles=handles, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.035))
    style.panel_label(ax, "a")
    fig.subplots_adjust(left=0.24, right=0.98, top=0.97, bottom=0.045)
    save_submission(fig, stem="SupplementaryFigureS1", panels="a")

    # S2 uses the public-cohort statistics with short cohort names and abbreviated malignant states in the submission
    # layout.
    z_score_table = style.state_expression_heatmap_table(expression_table)
    z_score_table.index = [
        f"{cohort_display(cohort, short=True)} | {state.removesuffix('-like')}" for cohort, state in z_score_table.index
    ]
    fig, ax = plt.subplots(figsize=(7.1, 5.05))
    heatmap = sns.heatmap(
        z_score_table,
        cmap       = style.diverging,
        center     = 0,
        linewidths = 0.55,
        linecolor  = "white",
        cbar_kws   = {"label": "Column z score", "shrink": 0.78},
        ax         = ax,
    )
    ax.set_xlabel("Ligand")
    ax.set_ylabel("")
    ax.set_title("Median patient-level expression by malignant state", pad=7)
    ax.tick_params(axis="y", labelsize=6.2)
    for mesh in heatmap.collections:
        if mesh.colorbar is not None:
            mesh.colorbar.outline.set_visible(False)
    style.panel_label(ax, "a")
    fig.subplots_adjust(left=0.20, right=0.96, top=0.94, bottom=0.10)
    save_submission(fig, stem="SupplementaryFigureS2", panels="a")

    # S3a compares cohort effects under uniform and original labels. Keep the uniform estimate when the original
    # estimate is missing.
    patient_table = pd.read_csv(data_dir / "uniform_annotation_patient_contrasts.csv")
    uniform_table = pd.read_csv(data_dir / "uniform_annotation_cohort_results.csv")
    primary_table = contrast_table.loc[contrast_table["gene"].eq("TGFB2"), ["cohort", "median_difference"]]
    primary_table = primary_table.rename(columns={"median_difference": "primary_difference"})
    annotation_table_total = uniform_table.merge(primary_table, on="cohort", how="left", validate="one_to_one")
    summary = json.loads((data_dir / "uniform_annotation_summary.json").read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(7.1, 3.15))
    grid_spec = GridSpec(1, 3, figure=fig, width_ratios=[0.96, 1.02, 1.12], wspace=0.48)

    ax = fig.add_subplot(grid_spec[0, 0])
    sensitivity_order = ["GSE162631", *style.cohort_order]
    ordered_annotation_table = annotation_table_total.set_index("cohort").loc[sensitivity_order].reset_index()
    y_positions = np.arange(len(ordered_annotation_table))[::-1]
    for y, row in zip(y_positions, ordered_annotation_table.itertuples()):
        if pd.notna(row.primary_difference):
            ax.plot([row.primary_difference, row.median_difference], [y, y], color=style.grid, linewidth=1.1)
            ax.scatter(
                row.primary_difference,
                y,
                s         = 28,
                facecolor = "white",
                edgecolor = style.cohort_colors[row.cohort],
                linewidth = 0.8,
            )

        # Filled points show uniform annotations. Connect them to open points for cohorts with original estimates.
        ax.scatter(
            row.median_difference, y, s=32, color=style.cohort_colors[row.cohort], edgecolor="white", linewidth=0.45
        )
    ax.axvline(0, color=style.gray_dark, linewidth=0.65, linestyle="--")
    ax.set_yticks(y_positions, [cohort_display(item, short=True) for item in ordered_annotation_table["cohort"]])
    ax.set_xlabel("Median TGFB2 difference")
    ax.set_title("Annotation sensitivity", pad=6)
    style.open_axes(ax)
    style.panel_label(ax, "a")

    # S3b shows each patient's difference and draws horizontal jitter from the same random stream.
    ax = fig.add_subplot(grid_spec[0, 1])
    for index, cohort in enumerate(sensitivity_order):
        values = patient_table.loc[patient_table["cohort"] == cohort, "tgfb2_difference"].to_numpy(float)
        ax.scatter(
            np.repeat(index, len(values)) + rng.normal(0, 0.045, len(values)),
            values,
            s         = 16,
            color     = style.cohort_colors[cohort],
            alpha     = 0.8,
            edgecolor = "white",
            linewidth = 0.35,
        )

        # The diamond is the median patient difference in this cohort.
        if len(values):
            ax.scatter(index, np.median(values), s=36, marker="D", color=style.ink, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color=style.gray_dark, linewidth=0.65, linestyle="--")
    ax.set_xticks(
        range(len(sensitivity_order)),
        [cohort_display(item, short=True) for item in sensitivity_order],
        rotation = 42,
        ha = "right",
    )
    ax.set_ylabel("Uniform AC-like minus other")
    ax.set_title("Patient-level effects", pad=6)
    style.open_axes(ax)
    style.panel_label(ax, "b")

    # Each S3c point is one patient. The x and y coordinates are the AC score difference and TGFB2 expression
    # difference.
    ax = fig.add_subplot(grid_spec[0, 2])
    for cohort, cohort_table in patient_table.groupby("cohort", observed=True):
        ax.scatter(
            cohort_table["ac_score_difference"],
            cohort_table["tgfb2_difference"],
            s         = 18,
            color     = style.cohort_colors[cohort],
            alpha     = 0.78,
            edgecolor = "white",
            linewidth = 0.35,
        )

    # The line describes the scatter. Read the Spearman correlation and HC3 P from the upstream model summary.
    x = patient_table["ac_score_difference"].to_numpy(float)
    y = patient_table["tgfb2_difference"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted_x = np.linspace(x.min(), x.max(), 100)
    ax.plot(fitted_x, intercept + slope * fitted_x, color=style.ink, linewidth=1.0)
    lineage = summary["lineage_adjustment"]
    ax.text(
        0.03,
        0.97,
        f"rho={lineage['spearman_rho']:.2f}\nHC3 P={lineage['hc3_p_value']:.2g}",
        transform = ax.transAxes,
        va        = "top",
        fontsize  = 6.2,
        color     = style.muted,
    )
    ax.axhline(0, color=style.gray, linewidth=0.6, linestyle="--")
    ax.set_xlabel("AC marker-score difference")
    ax.set_ylabel("TGFB2 difference")
    ax.set_title("Lineage-program sensitivity", pad=6)
    style.open_axes(ax)
    style.panel_label(ax, "c")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.19)
    save_submission(fig, stem="SupplementaryFigureS3", panels="abc")

    # S4a and S4b show both scoring methods' full grids in the original order of four cell-count and four score
    # thresholds.
    sensitivity_table = pd.read_csv(data_dir / "parameter_sensitivity_grid.csv")
    cohort_table = pd.read_csv(data_dir / "parameter_sensitivity_cohort_results.csv")
    score_thresholds = [0.0, 0.025, 0.05, 0.10]
    cell_thresholds = [10, 20, 30, 50]
    modes = ["Raw marker mean", "Within-cohort gene z-score"]
    fig = plt.figure(figsize=(7.1, 4.85))
    grid_spec = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.78], hspace=0.50, wspace=0.42)
    for index, mode in enumerate(modes):
        ax = fig.add_subplot(grid_spec[0, index])
        mode_table = sensitivity_table[sensitivity_table["scoring_mode"] == mode]
        effect_table = mode_table.pivot(index="minimum_cells", columns="score_threshold", values="meta_z").reindex(
            index=cell_thresholds, columns=score_thresholds
        )
        labels_table = mode_table.copy()
        grid_labels = []
        for row in mode_table.itertuples():
            grid_labels.append(f"{row.meta_z:.2f}\n{int(row.positive_cohorts)}/{int(row.n_cohorts)} +")
        labels_table["label"] = grid_labels
        annotation_table = labels_table.pivot(index="minimum_cells", columns="score_threshold", values="label").reindex(
            index=cell_thresholds, columns=score_thresholds
        )

        # Each cell gives the observed z and the number of positive cohorts over evaluable cohorts. Both panels use the
        # submission color scale.
        heatmap = sns.heatmap(
            effect_table,
            ax         = ax,
            annot      = annotation_table,
            fmt        = "",
            cmap       = style.teal_seq,
            vmin       = 3.0,
            vmax       = 4.2,
            linewidths = 0.7,
            linecolor  = "white",
            cbar_kws   = {"label": "Signed Stouffer z", "shrink": 0.75},
            annot_kws  = {"fontsize": 6.2},
        )
        ax.set_xlabel("Score threshold")
        ax.set_ylabel("Minimum cells per group")
        ax.set_xticklabels([f"{value:g}" for value in effect_table.columns])
        ax.set_yticklabels([str(value) for value in effect_table.index], rotation=0)
        ax.set_title(mode, pad=6)
        for mesh in heatmap.collections:
            if mesh.colorbar is not None:
                mesh.colorbar.outline.set_visible(False)
        style.panel_label(ax, chr(ord("a") + index))

    ax = fig.add_subplot(grid_spec[1, :])

    # S4c compares the two methods in the original cohorts at a score threshold of 0.05 and at least 20 cells per group.
    baseline_table = cohort_table[
        np.isclose(cohort_table["score_threshold"], 0.05) & cohort_table["minimum_cells"].eq(20)
    ]
    order = ["GSE162631", "GSE211376", "GSE273274", "GSE274987", "GSE286419"]
    scoring_table = baseline_table.pivot(index="cohort", columns="scoring_mode", values="median_difference").reindex(
        order
    )
    y_positions = np.arange(len(order))[::-1]

    # The line joins the two scoring methods' median patient differences within the same cohort.
    raw_values = scoring_table["Raw marker mean"].to_numpy(float)
    standardized = scoring_table["Within-cohort gene z-score"].to_numpy(float)
    for y, left, right in zip(y_positions, raw_values, standardized):
        ax.plot([left, right], [y, y], color=style.grid, linewidth=1.2)
    ax.scatter(raw_values, y_positions, color=style.coral, s=30, label="Raw marker mean")
    ax.scatter(standardized, y_positions, color=style.teal, s=30, label="Gene z-score")
    ax.axvline(0, color=style.gray_dark, linewidth=0.65, linestyle="--")
    ax.set_yticks(y_positions, [cohort_display(item, short=True) for item in order])
    ax.set_xlabel("Cohort median TGFB2 difference")
    ax.set_title("Baseline parameters", pad=6)
    ax.legend(loc="lower right", ncol=2)
    style.open_axes(ax)
    style.panel_label(ax, "c")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.94, bottom=0.11)
    save_submission(fig, stem="SupplementaryFigureS4", panels="abc")

    # S5a and S5b compare the normal approximation and exact sign-flip q for the same ligand to assess calibration.
    calibration_table = pd.read_csv(data_dir / "exact_meta_calibration.csv")
    tail_table = pd.read_csv(data_dir / "exact_meta_null_tail.csv")
    ordered_calibration_table = calibration_table.set_index("gene", verify_integrity=True).loc[gene_order]
    colors = {"TGFB2": style.coral, "CSF1": style.gold, "ANXA1": style.blue}
    fig = plt.figure(figsize=(7.1, 4.55))
    grid_spec = GridSpec(2, 2, figure=fig, height_ratios=[0.95, 0.88], hspace=0.48, wspace=0.44)
    calibration_panels = [
        ("normal_approximation_q_value", "exact_signflip_q_value", "Full meta-analysis"),
        ("normal_loo_q_value", "exact_loo_q_value", "Leave-one-cohort-out"),
    ]
    for index, (normal_column, exact_column, title) in enumerate(calibration_panels):
        ax = fig.add_subplot(grid_spec[0, index])
        for gene, row in ordered_calibration_table.iterrows():
            normal = -np.log10(max(float(row[normal_column]), 1e-15))
            exact = -np.log10(max(float(row[exact_column]), 1e-15))
            color = colors.get(gene, style.gray_dark)
            ax.plot([0, 1], [normal, exact], color=color, linewidth=0.9, alpha=0.8)
            ax.scatter([0, 1], [normal, exact], color=color, s=24, edgecolor="white", linewidth=0.4, zorder=2)
            if gene in colors:
                ax.text(1.04, exact, gene, va="center", fontsize=6.1, color=color)
        ax.axhline(-np.log10(0.05), color=style.gray_dark, linestyle="--", linewidth=0.65)
        ax.set_xlim(-0.18, 1.45)
        ax.set_xticks([0, 1], ["Normal\napprox.", "Exact\nsign flip"])
        ax.set_ylabel("Evidence, -log10(q)")
        ax.set_title(title, pad=6)
        style.open_axes(ax)
        style.panel_label(ax, "ab"[index])

    # S5c uses the frozen empirical tail curves. All threshold rows for one ligand must have the same observed z and
    # exact P.
    ax = fig.add_subplot(grid_spec[1, :])
    for gene in ["TGFB2", "CSF1"]:
        gene_tail_table = tail_table[tail_table["gene"] == gene]
        color = colors[gene]
        ax.plot(
            gene_tail_table["absolute_z_threshold"],
            gene_tail_table["exact_tail_probability"],
            color = color,
            linewidth = 1.5,
            label = gene,
        )
        observed_z = gene_tail_table["observed_absolute_z"].drop_duplicates().item()
        observed_p = gene_tail_table["observed_exact_p_value"].drop_duplicates().item()
        ax.scatter(observed_z, observed_p, color=color, s=30, edgecolor="white", linewidth=0.45, zorder=3)

        # The dashed line marks the observed statistic. Plot exact tail probabilities on the original logarithmic axis.
        ax.vlines(observed_z, 1e-6, observed_p, color=color, linestyle=":", linewidth=0.75)
    ax.axhline(0.05, color=style.gray_dark, linestyle="--", linewidth=0.65)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.1)
    ax.set_xlabel("Absolute signed Stouffer z")
    ax.set_ylabel("Exact sign-flip tail probability")
    ax.set_title("Exhaustive patient-level null distribution", pad=6)
    ax.legend()
    style.open_axes(ax)
    style.panel_label(ax, "c")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.94, bottom=0.12)
    save_submission(fig, stem="SupplementaryFigureS5", panels="abc")

    # S6 and S7 each use patient ligand and proximity tables plus one representative section. Metrics and panel counts
    # depend on the dataset.
    for source in ["greenwald", "gse253080"]:
        result_dir = external_results / source
        ligand_table = pd.read_csv(result_dir / "patient_ligand_metrics.csv")
        spatial_table = pd.read_csv(result_dir / "patient_spatial_metrics.csv")
        section_table = pd.read_csv(result_dir / "representative_plot.csv")
        tgfb2_table = ligand_table[ligand_table["ligand"] == "TGFB2"]

        # S6 uses the authors' AC labels. S7 uses the frozen continuous AC score. Retain each figure's canvas, point
        # size and annotations.
        if source == "greenwald":
            fig = plt.figure(figsize=(7.1, 5.20))
            outer = GridSpec(2, 1, figure=fig, height_ratios=[0.82, 1.10], hspace=0.42)
            top = GridSpecFromSubplotSpec(
                1, 4, subplot_spec=outer[0], width_ratios=[0.85, 1.45, 0.85, 0.85], wspace=0.52
            )
            field = "ac_minus_other_mean"
            ax = fig.add_subplot(top[0, 0])
            add_strip(ax, tgfb2_table[field], title="TGFB2 in AC spots", ylabel="AC minus other")
            ylabel = "AC minus other malignant"
            spread, dot_size, opacity, edge_width, bar_width = 0.05, 13, 0.72, 0.30, 0.28
        else:
            fig = plt.figure(figsize=(7.1, 4.90))
            outer = GridSpec(2, 1, figure=fig, height_ratios=[0.78, 1.08], hspace=0.42)
            top = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], width_ratios=[0.80, 1.45, 0.80], wspace=0.50)
            field = "rho_ac_score_adjusted"
            ax = fig.add_subplot(top[0, 0])
            add_strip(ax, tgfb2_table[field], title="TGFB2 vs\nAC score", ylabel="Adjusted rho")
            ylabel = "Adjusted rho"
            spread, dot_size, opacity, edge_width, bar_width = 0.045, 17, 0.75, 0.35, 0.30
        style.panel_label(ax, "a")

        # Pivot by patient and ligand for each dataset. Use the same fixed ligand order and the original sequence of
        # random draws.
        ax = fig.add_subplot(top[0, 1])
        ligand_wide_table = ligand_table.pivot(index="patient", columns="ligand", values=field)
        for index, ligand in enumerate(spatial_ligands):
            values = ligand_wide_table[ligand].dropna().to_numpy(float)
            positions = np.full(len(values), index) + rng.normal(0, spread, len(values))
            color = style.coral if ligand == "TGFB2" else style.gray_dark
            ax.scatter(
                positions, values, s=dot_size, color=color, alpha=opacity, edgecolor="white", linewidth=edge_width
            )
            if len(values):
                bar_color = style.coral if ligand == "TGFB2" else style.ink
                style.median_bar(ax, index, float(np.median(values)), color=bar_color, width=bar_width, linewidth=1.5)
        ax.axhline(0, color=style.gray_dark, linewidth=0.65, linestyle="--")
        ax.set_xticks(range(len(spatial_ligands)), spatial_ligands, rotation=38, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title("Fixed ligand panel", pad=6)
        style.open_axes(ax)
        style.panel_label(ax, "b")

        # Greenwald includes contrasts with other malignant and reactive astrocyte spots. GSE253080 shows its available
        # proximity metric.
        if source == "greenwald":
            ax = fig.add_subplot(top[0, 2])
            add_strip(
                ax,
                tgfb2_table["ac_minus_reactive_mean"],
                title  = "Malignant vs\nreactive",
                ylabel = "TGFB2 delta",
                color  = style.blue,
            )
            style.panel_label(ax, "c")
            ax = fig.add_subplot(top[0, 3])
            add_strip(
                ax,
                spatial_table["tgfb2_neighbor_receiver_delta"],
                title  = "Neighbor\nreceiver",
                ylabel = "Score delta",
                color  = style.purple,
            )
            style.panel_label(ax, "d")

            # Code the authors' literal AC label. Use the same pixel_col and pixel_row coordinates for all three spatial
            # measures.
            section_table["ac_label"] = section_table["label"].astype(str).eq("AC").astype(float)
            spatial_fields = [
                ("ac_label", "Author-assigned AC spots", style.teal_seq),
                ("tgfb2", "TGFB2 expression", style.coral_seq),
                ("receiver", "Myeloid x TGFBR score", style.purple_seq),
            ]
            point_size, letters, stem = 3.6, "efg", "SupplementaryFigureS6"
        else:
            ax = fig.add_subplot(top[0, 2])
            add_strip(
                ax,
                spatial_table["tgfb2_neighbor_receiver_delta"],
                title  = "Descriptive\nproximity",
                ylabel = "Receiver-score delta",
                color  = style.purple,
            )
            style.panel_label(ax, "c")
            spatial_fields = [
                ("ac_score", "AC-like score", style.teal_seq),
                ("tgfb2", "TGFB2 expression", style.coral_seq),
                ("receiver_score", "Myeloid x TGFBR score", style.purple_seq),
            ]
            point_size, letters, stem = 3.5, "def", "SupplementaryFigureS7"

        # Image row numbers increase downward, so negate y. The original 1st and 99th percentiles set color limits while
        # retaining all data.
        bottom = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.32)
        for index, (field, title, cmap) in enumerate(spatial_fields):
            ax = fig.add_subplot(bottom[0, index])
            values = pd.to_numeric(section_table[field], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            if not len(finite):
                raise ValueError(f"The {source} representative section has no plottable values for {field}")
            low, high = np.quantile(finite, [0.01, 0.99])
            if np.isclose(low, high):
                low, high = float(finite.min()), float(finite.max())
            scatter = ax.scatter(
                section_table["pixel_col"],
                -section_table["pixel_row"],
                c          = values,
                s          = point_size,
                cmap       = cmap,
                vmin       = low,
                vmax       = high,
                linewidths = 0,
                rasterized = True,
            )

            # Use equal spatial scaling and show each panel's units on its color bar. Only the point layer is
            # rasterized. Text stays vector.
            ax.set_aspect("equal")
            ax.set_title(title, pad=5)
            ax.set_xticks([])
            ax.set_yticks([])
            style.clean_axes(ax)
            colorbar = fig.colorbar(scatter, ax=ax, fraction=0.038, pad=0.018)
            colorbar.outline.set_visible(False)
            colorbar.ax.tick_params(labelsize=5.6, width=0.45, length=1.8)
            style.panel_label(ax, letters[index])

        # Leave room on the right for the final color bar labels. Preserve page dimensions and the three panels' spatial
        # coordinates.
        fig.subplots_adjust(left=0.09, right=0.95, top=0.92, bottom=0.07)
        save_submission(fig, stem=stem, panels="abcd" + letters if source == "greenwald" else "abc" + letters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot submission figures from frozen results")
    parser.add_argument("--main-only", action="store_true", help="Draw only Figure1 through Figure6")
    parser.add_argument("--output", type=Path, help="Output directory (default: this submission package's figures/)")
    parser.add_argument(
        "--tiff", action="store_true", help="Export Fig1.tif through Fig6.tif from the main PDFs (600 dpi, RGB, LZW)"
    )
    args = parser.parse_args()
    main(args.output, main_only=args.main_only, tiff=args.tiff)
