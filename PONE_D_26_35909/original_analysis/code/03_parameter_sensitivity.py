#!/usr/bin/env python3
# Purpose: Assess patient TGFB2 results across raw marker means and within-cohort gene z-scores, score-margin thresholds
# and minimum cells per group.
# Inputs: The five public inputs used by 02_uniform_annotation_sensitivity.py. scoring_mode, score_threshold and
# minimum_cells identify each parameter combination.
# Outputs: data/parameter_sensitivity_grid.csv, data/parameter_sensitivity_cohort_results.csv,
# data/parameter_sensitivity_summary.json, figures/SupplementaryFigureS4.pdf and figures/SupplementaryFigureS4.png.
# Usage: Run python 03_parameter_sensitivity.py from code/. Call make_figure() to plot the two existing parameter
# tables.
# Use the original Python numerical libraries. Standalone releases write to 01_cohort_reanalysis/. Convert the PDF to
# PNG.

from __future__ import annotations

from importlib import import_module
from itertools import product

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

uniform = import_module("02_uniform_annotation_sensitivity")
paths = import_module("00a_project_paths")


def main() -> None:
    """Calculate the grid, verify the eligible cohort and patient counts per combination, and plot S4. Returns None."""
    uniform.primary.stop_if_missing()
    grid_table, cohort_table_total = calculate_grid()
    key_columns = ["scoring_mode", "score_threshold", "minimum_cells"]
    expected = set(product(scoring_modes, score_thresholds, cell_thresholds))
    if set(grid_table[key_columns].itertuples(index=False, name=None)) != expected:
        raise ValueError("The sensitivity summary is missing a prespecified combination or contains an extra one")

    # Keep descriptive rows for cohorts with fewer than three patients. Only cohorts with a two-sided P value contribute
    # to the summary counts.
    valid_table = cohort_table_total.loc[cohort_table_total["p_value"].notna() & cohort_table_total["n_patients"].ge(3)]
    counts_table = valid_table.groupby(key_columns)["n_patients"].agg(["size", "sum"])
    grid_indexed_table = grid_table.set_index(key_columns)
    counts_table = counts_table.reindex(grid_indexed_table.index, fill_value=0)
    if not counts_table["size"].equals(grid_indexed_table["n_cohorts"]):
        raise ValueError("Eligible cohort counts in the parameter summary differ from the cohort results")
    if not counts_table["sum"].equals(grid_indexed_table["n_patients"]):
        raise ValueError("Patient counts in the parameter summary differ from the cohort results")

    output = paths.analysis_output_root() / "data"
    paths.write_table(grid_table, output / "parameter_sensitivity_grid.csv", key_columns)
    paths.write_table(cohort_table_total, output / "parameter_sensitivity_cohort_results.csv", [*key_columns, "cohort"])
    summary = {
        "parameter_combinations": len(grid_table),
        "all_meta_p_below_0_05": bool((grid_table["meta_p_value"] < 0.05).all()),
        "minimum_positive_cohorts": int(grid_table["positive_cohorts"].min()),
        "minimum_meta_z": float(grid_table["meta_z"].min()),
        "maximum_meta_p_value": float(grid_table["meta_p_value"].max()),
        "maximum_leave_one_out_p_value": float(grid_table["loo_max_p_value"].max()),
    }
    paths.write_json(summary, output / "parameter_sensitivity_summary.json")
    make_figure()
    print(f"{len(grid_table)} parameter combinations and {len(cohort_table_total)} cohort result rows.")


# Read expression once per scoring mode. Apply each margin threshold after identifying the highest-scoring state in each
# cell.
scoring_modes = {"Raw marker mean": False, "Within-cohort gene z-score": True}


def calculate_grid() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the published grid. Pass each parameter explicitly and preserve other modules' defaults."""
    grid_rows = []
    cohort_parts = []
    for mode, standardize in scoring_modes.items():
        cell_table_total = uniform.load_uniform_cells(standardize=standardize, score_threshold=0.0)
        for score_threshold in score_thresholds:
            eligible_table = cell_table_total.loc[~(cell_table_total["confidence"] < score_threshold)]
            for cell_threshold in cell_thresholds:
                patient_table = uniform.patient_contrasts(eligible_table, minimum_cells=cell_threshold)
                cohort_table = uniform.cohort_results(patient_table)
                meta = uniform.meta_result(cohort_table)

                # Save one pooled estimate per combination and retain every cohort row, including those without a test
                # result.
                parameters = {"scoring_mode": mode, "score_threshold": score_threshold, "minimum_cells": cell_threshold}
                grid_rows.append(
                    {
                        **parameters,
                        "n_cohorts": meta["n_cohorts"],
                        "n_patients": meta["n_patients"],
                        "positive_cohorts": meta["positive_cohorts"],
                        "meta_z": meta["meta_z"],
                        "meta_p_value": meta["p_value"],
                        "loo_max_p_value": meta["loo_max_p_value"],
                    }
                )
                cohort_table.insert(0, "minimum_cells", cell_threshold)
                cohort_table.insert(0, "score_threshold", score_threshold)
                cohort_table.insert(0, "scoring_mode", mode)
                cohort_parts.append(cohort_table)

    grid_table = pd.DataFrame(grid_rows)
    cohort_table_total = pd.concat(cohort_parts, ignore_index=True)
    result = grid_table, cohort_table_total
    return result


# Keep the published order on both threshold axes. The baseline combination uses the defaults from stage 02.
score_thresholds = [0.0, 0.025, 0.05, 0.10]
cell_thresholds = [10, 20, 30, 50]


def make_figure() -> None:
    """Read both CSVs from the same source and plot the two S4 heatmaps and matched cohort points. Returns None."""
    directory = paths.result_input_directory(
        "parameter_sensitivity_grid.csv", "parameter_sensitivity_cohort_results.csv"
    )
    grid_table = pd.read_csv(directory / "parameter_sensitivity_grid.csv")
    cohort_table = pd.read_csv(directory / "parameter_sensitivity_cohort_results.csv")

    # Limit seaborn's white style and the original font to this figure. Restore the caller's Matplotlib settings
    # afterward.
    with mpl.rc_context():
        sns.set_theme(style="white")
        mpl.rcParams.update(
            {
                "font.family": "Arial",
                "font.size": 8,
                "axes.labelsize": 8,
                "axes.titlesize": 9,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "pdf.fonttype": 42,
                "figure.facecolor": "white",
            }
        )
        figure = plt.figure(figsize=(7.3, 6.4))
        try:
            layout = figure.add_gridspec(2, 2, height_ratios=[1.0, 0.82], hspace=0.52, wspace=0.48)
            for index, mode in enumerate(scoring_modes):
                axis = figure.add_subplot(layout[0, index])
                draw_heatmap(axis, grid_table.loc[grid_table["scoring_mode"] == mode], mode)
                axis.text(-0.13, 1.08, "ab"[index], transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")

            axis = figure.add_subplot(layout[1, :])
            draw_baseline(axis, cohort_table)
            figure.suptitle(
                "Uniform-marker TGFB2 results across analysis parameters",
                x=0.07,
                ha="left",
                fontsize=11,
                fontweight="bold",
            )
            paths.save_figure(figure, paths.analysis_output_root() / "figures" / "SupplementaryFigureS4.pdf")
        finally:
            plt.close(figure)


def draw_heatmap(axis: plt.Axes, grid_table: pd.DataFrame, title: str) -> None:
    """Use the pooled z and both cohort counts from one row per cell. Stop for an incomplete grid. Returns None."""
    indexed_table = grid_table.set_index(["minimum_cells", "score_threshold"], verify_integrity=True)
    if set(indexed_table.index) != set(product(cell_thresholds, score_thresholds)):
        raise ValueError(f"{title}: heatmap cells do not match the prespecified grid")
    values_table = indexed_table["meta_z"].unstack().loc[cell_thresholds, score_thresholds]
    labels = []
    for row in indexed_table.itertuples():
        labels.append(f"z={row.meta_z:.2f}\n{row.positive_cohorts}/{row.n_cohorts} +")
    indexed_table["label"] = labels
    labels_table = indexed_table["label"].unstack().loc[cell_thresholds, score_thresholds]
    sns.heatmap(
        values_table,
        ax=axis,
        annot=labels_table,
        fmt="",
        cmap="viridis",
        vmin=3.0,
        vmax=4.2,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Signed Stouffer z", "shrink": 0.78},
        annot_kws={"fontsize": 6.7},
    )
    axis.set_xlabel("Top-versus-second score threshold")
    axis.set_ylabel("Minimum cells per group")
    axis.set_xticklabels([f"{value:g}" for value in values_table.columns])
    axis.set_yticklabels([str(value) for value in values_table.index], rotation=0)
    axis.set_title(title, loc="left", pad=8)


def draw_baseline(axis: plt.Axes, cohort_table: pd.DataFrame) -> None:
    """Plot the five S4c cohorts. Keep GSE276177, which has fewer than three patients, in the table. Returns None."""
    baseline = np.isclose(cohort_table["score_threshold"], uniform.minimum_score_difference)
    baseline &= cohort_table["minimum_cells"].eq(uniform.primary.minimum_cells)
    baseline_table = cohort_table.loc[baseline]
    order = ["GSE162631", "GSE211376", "GSE273274", "GSE274987", "GSE286419"]
    wide_table = baseline_table.pivot(index="cohort", columns="scoring_mode", values="median_difference")
    wide_table = wide_table.loc[order, list(scoring_modes)]
    if wide_table.isna().any(axis=None):
        raise ValueError("An S4c cohort is missing the baseline contrast for a scoring mode")

    positions = np.arange(len(order))[::-1]
    raw_means = wide_table["Raw marker mean"].to_numpy(float)
    standardized = wide_table["Within-cohort gene z-score"].to_numpy(float)
    for y, left, right in zip(positions, raw_means, standardized):
        axis.plot([left, right], [y, y], color="#B7B7B7", lw=1.0, zorder=1)
    axis.scatter(
        raw_means, positions, color="#C84B31", s=38, edgecolor="white", linewidth=0.7, label="Raw marker mean", zorder=2
    )
    axis.scatter(
        standardized, positions, color="#2B7A78", s=38, edgecolor="white", linewidth=0.7, label="Gene z-score", zorder=2
    )
    axis.axvline(0, color="#777777", lw=0.8, ls="--")
    axis.set_yticks(positions, order)
    axis.set_xlabel("Cohort median TGFB2 difference (AC-like - other states)")
    axis.set_title("Baseline parameters: cohort direction under two scoring scales", loc="left", pad=8)
    axis.legend(frameon=False, loc="lower right")
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(-0.13, 1.08, "c", transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")


if __name__ == "__main__":
    main()
