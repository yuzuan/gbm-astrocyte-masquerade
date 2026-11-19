#!/usr/bin/env python3
# Purpose: Calibrate six ligands and their leave-one-cohort-out results by enumerating all patient-level sign flips in
# the four prespecified cohorts.
# Inputs from the same data/ directory: ac_vs_other_ligand_contrasts.csv (patient_values, n_patients) and
# meta_ligand_results.csv (gene, p_value, q_value, loo_max_p_value, loo_q_value).
# Outputs: data/exact_meta_calibration.csv, data/exact_meta_null_tail.csv, data/exact_meta_calibration_summary.json,
# figures/SupplementaryFigureS5.pdf and figures/SupplementaryFigureS5.png.
# Usage: Run python 05_exact_meta_calibration.py from code/ using the published or rebuilt tables. Call make_figure() to
# plot existing results.
# Standalone releases write to 01_cohort_reanalysis/. Retain SciPy's original exact Wilcoxon method and floating-point
# tail tolerance.

from __future__ import annotations

import json
from importlib import import_module
from itertools import product

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon

primary = import_module("01_patient_expression")
paths = import_module("00a_project_paths")


def main() -> None:
    """Write exhaustive calibration for each ligand, two tail probability curves and S5. Returns None."""
    result_table, tail_table, summary = calculate()
    output = paths.analysis_output_root() / "data"
    paths.write_table(result_table, output / "exact_meta_calibration.csv", ["gene"])
    paths.write_table(tail_table, output / "exact_meta_null_tail.csv", ["gene", "absolute_z_threshold"])
    paths.write_json(summary, output / "exact_meta_calibration_summary.json")
    make_figure()
    print(
        f"Calibrated {len(result_table)} ligands with "
        f"{summary['full_signflip_patterns_per_ligand']} sign patterns per ligand."
    )


def calculate() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Enumerate in published order. Derive denominators, leave-one-out results and both curves from nested patient
    differences.
    """
    directory = paths.result_input_directory("ac_vs_other_ligand_contrasts.csv", "meta_ligand_results.csv")
    contrast_table = pd.read_csv(directory / "ac_vs_other_ligand_contrasts.csv")
    normal_table = pd.read_csv(directory / "meta_ligand_results.csv")
    normal_table = normal_table.set_index("gene", verify_integrity=True)
    rows = []
    tail_rows = []
    for gene in gene_order:
        patient_values = parse_patient_values(contrast_table, gene)
        distributions = [signflip_z_distribution(values) for values in patient_values]
        included = list(range(len(patient_values)))
        observed_z, exact_p, distribution = combine_distribution(patient_values, distributions, included)
        n_patients = sum(len(values) for values in patient_values)
        if len(distribution) != 2**n_patients:
            raise ValueError(f"{gene}: the sign pattern count does not match the total patient count")

        # Omit an entire cohort, preserving the order and patient weights of the remainder. Read the unrounded normal
        # approximations from the same input tables.
        omitted_results = []
        for index, omitted in enumerate(cohort_order):
            remaining = [j for j in included if j != index]
            omitted_z, omitted_p, _ = combine_distribution(patient_values, distributions, remaining)
            omitted_results.append({"omitted": omitted, "meta_z": omitted_z, "exact_p_value": omitted_p})
        rows.append(
            {
                "gene": gene,
                "n_cohorts": len(included),
                "n_patients": n_patients,
                "observed_meta_z": observed_z,
                "normal_approximation_p_value": float(normal_table.at[gene, "p_value"]),
                "normal_approximation_q_value": float(normal_table.at[gene, "q_value"]),
                "exact_signflip_p_value": exact_p,
                "normal_loo_max_p_value": float(normal_table.at[gene, "loo_max_p_value"]),
                "normal_loo_q_value": float(normal_table.at[gene, "loo_q_value"]),
                "exact_loo_max_p_value": max(row["exact_p_value"] for row in omitted_results),
                "exact_loo_results": json.dumps(omitted_results, ensure_ascii=False),
                "full_signflip_patterns": len(distribution),
            }
        )

        # For these two ligands, the original figure samples 121 equally spaced points from zero to the maximum |z|.
        # Each sign pattern has equal weight.
        if gene in tail_genes:
            absolute = np.abs(distribution)
            for threshold in np.linspace(0, absolute.max(), 121):
                tail_rows.append(
                    {
                        "gene": gene,
                        "absolute_z_threshold": threshold,
                        "exact_tail_probability": float(np.mean(absolute >= threshold - 1e-12)),
                        "observed_absolute_z": abs(observed_z),
                        "observed_exact_p_value": exact_p,
                    }
                )

    result_table = pd.DataFrame(rows)
    result_table["exact_signflip_q_value"] = primary.bh_adjust(result_table["exact_signflip_p_value"])
    result_table["exact_loo_q_value"] = primary.bh_adjust(result_table["exact_loo_max_p_value"])
    result_table = result_table.sort_values("exact_signflip_q_value").reset_index(drop=True)
    patterns = result_table["full_signflip_patterns"].unique()
    if len(patterns) != 1:
        raise ValueError("Ligands have different exhaustive pattern counts and cannot share a denominator")

    # List only ligands meeting both exact q-value criteria. Look up TGFB2 by gene, independently of its adjusted rank.
    tgfb2_table = result_table.loc[result_table["gene"] == "TGFB2"]
    passing = result_table["exact_signflip_q_value"].lt(0.05) & result_table["exact_loo_q_value"].lt(0.05)
    summary = {
        "enumeration_is_exhaustive": True,
        "full_signflip_patterns_per_ligand": int(patterns.item()),
        "tgfb2_exact_p_value": float(tgfb2_table["exact_signflip_p_value"].item()),
        "tgfb2_exact_loo_q_value": float(tgfb2_table["exact_loo_q_value"].item()),
        "genes_passing_exact_primary_and_loo_q": result_table.loc[passing, "gene"].tolist(),
        "interpretation": "Sensitivity calibration of the signed Stouffer statistic; "
        "primary claims remain based on the primary normal approximation.",
    }
    tail_table = pd.DataFrame(tail_rows)
    result = result_table, tail_table, summary
    return result


# GSE276177 is descriptive in the published analysis. Cohort and ligand order determine floating-point summation order.
cohort_order = ["GSE211376", "GSE273274", "GSE274987", "GSE286419"]
gene_order = ["TGFB2", "CSF1", "ANXA1", "CCL2", "TGFB1", "SPP1"]
tail_genes = ["TGFB2", "CSF1"]


def parse_patient_values(contrast_table: pd.DataFrame, gene: str) -> list[np.ndarray]:
    """Read one row per cohort, verify the nested patient counts and differences, and retain file order in the arrays.
    """
    arrays = []
    for cohort in cohort_order:
        group_table = contrast_table.loc[contrast_table["gene"].eq(gene) & contrast_table["cohort"].eq(cohort)]
        records = json.loads(group_table["patient_values"].item())
        patients = [row["patient"] for row in records]
        if len(records) != group_table["n_patients"].item() or len(patients) != len(set(patients)):
            raise ValueError(f"{gene}/{cohort}: invalid patient IDs or number of patient differences")
        differences = np.asarray([row["difference"] for row in records], dtype=float)
        if not len(differences) or not np.isfinite(differences).all():
            raise ValueError(f"{gene}/{cohort}: non-finite differences prevent exhaustive enumeration")
        arrays.append(differences)
    return arrays


def signflip_z_distribution(values: np.ndarray) -> np.ndarray:
    """Enumerate all -1/+1 sign vectors and preserve repeated statistic values with their frequencies. No random
    sampling.
    """
    scores = []
    for signs in product([-1.0, 1.0], repeat=len(values)):
        scores.append(signed_wilcoxon_z(values * np.asarray(signs)))
    distribution = np.asarray(scores, dtype=float)
    return distribution


def signed_wilcoxon_z(values: np.ndarray) -> float:
    """Exclude near-zero differences. Convert the exact two-sided signed-rank P value and the median direction to z."""
    differences = np.asarray(values, dtype=float)
    nonzero = differences[~np.isclose(differences, 0.0)]
    z_value = 0.0
    if len(nonzero):
        p_value = float(wilcoxon(nonzero, alternative="two-sided", method="exact").pvalue)
        z_value = float(norm.isf(np.clip(p_value, 1e-15, 1.0) / 2) * np.sign(np.median(nonzero)))
    return z_value


def combine_distribution(
    patient_values: list[np.ndarray], cohort_distributions: list[np.ndarray], included: list[int]
) -> tuple[float, float, np.ndarray]:
    """Weight cohorts by the square root of all paired patients. Return observed z, two-sided tail probability and the
    joint null with multiplicities.
    """
    weights = np.asarray([np.sqrt(len(patient_values[index])) for index in included], dtype=float)
    observed_scores = np.asarray([signed_wilcoxon_z(patient_values[index]) for index in included])
    denominator = np.sqrt(np.dot(weights, weights))
    observed = float(np.dot(weights, observed_scores) / denominator)

    # Extend the Cartesian product one cohort at a time. Use the published 1e-12 tolerance for floating-point ties at
    # the tail.
    distribution = np.asarray([0.0])
    for weight, index in zip(weights, included):
        distribution = (distribution[:, None] + weight * cohort_distributions[index][None, :]).reshape(-1)
    distribution /= denominator
    exact_p = float(np.mean(np.abs(distribution) >= abs(observed) - 1e-12))
    result = observed, exact_p, distribution
    return result


def make_figure() -> None:
    """Read both saved CSVs and draw full calibration, leave-one-out calibration and the tail panel. Returns None."""
    directory = paths.result_input_directory("exact_meta_calibration.csv", "exact_meta_null_tail.csv")
    result_table = pd.read_csv(directory / "exact_meta_calibration.csv")
    tail_table = pd.read_csv(directory / "exact_meta_null_tail.csv")
    style = {
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
    with mpl.rc_context(style):
        figure = plt.figure(figsize=(7.3, 5.6))
        try:
            layout = figure.add_gridspec(2, 2, height_ratios=[1.0, 0.9], hspace=0.5, wspace=0.5)
            full = figure.add_subplot(layout[0, 0])
            calibration_slope(
                full,
                result_table,
                "normal_approximation_q_value",
                "exact_signflip_q_value",
                "Full meta-analysis calibration",
            )
            omitted = figure.add_subplot(layout[0, 1])
            calibration_slope(
                omitted, result_table, "normal_loo_q_value", "exact_loo_q_value", "Leave-one-cohort-out calibration"
            )

            tail_axis = figure.add_subplot(layout[1, :])
            draw_tail(tail_axis, tail_table, result_table)
            for axis, letter in zip((full, omitted, tail_axis), "abc"):
                axis.text(-0.14, 1.08, letter, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")
            figure.suptitle(
                "Calibration of the meta-analysis against exact patient-level sign flips",
                x=0.07,
                ha="left",
                fontsize=11,
                fontweight="bold",
            )
            paths.save_figure(figure, paths.analysis_output_root() / "figures" / "SupplementaryFigureS5.pdf")
        finally:
            plt.close(figure)


def calibration_slope(axis: plt.Axes, result_table: pd.DataFrame, normal_col: str, exact_col: str, title: str) -> None:
    """Validate both probabilities and plot each ligand's normal and exact q-values. Returns None."""
    ordered_table = result_table.set_index("gene", verify_integrity=True).loc[gene_order]
    evidence_table = ordered_table[[normal_col, exact_col]]
    valid_table = evidence_table.ge(0) & evidence_table.le(1)
    if not valid_table.all(axis=None):
        raise ValueError(f"{title}: missing or invalid q-values")
    for gene, row in ordered_table.iterrows():
        values = [-np.log10(max(float(row[column]), 1e-15)) for column in (normal_col, exact_col)]
        color = gene_colors.get(gene, "#8A8A8A")
        axis.plot([0, 1], values, color=color, lw=1.1, alpha=0.8)
        axis.scatter([0, 1], values, color=color, s=32, edgecolor="white", linewidth=0.6, zorder=2)
        if gene in gene_colors:
            axis.text(1.04, values[1], gene, va="center", fontsize=6.7, color=color)
    axis.axhline(-np.log10(0.05), color="#777777", ls="--", lw=0.8)
    axis.set_xlim(-0.18, 1.48)
    axis.set_xticks([0, 1], ["Normal\napproximation", "Exact\nsign flip"])
    axis.set_ylabel("Evidence, -log10(q)")
    axis.set_title(title, loc="left", pad=8)
    axis.spines[["top", "right"]].set_visible(False)


gene_colors = {"TGFB2": "#B43C52", "CSF1": "#C49A2C", "ANXA1": "#4C78A8"}


def draw_tail(axis: plt.Axes, tail_table: pd.DataFrame, result_table: pd.DataFrame) -> None:
    """Verify non-increasing tails and match observed |z| and exact P to the ligand table before plotting. Returns None.
    """
    indexed_table = result_table.set_index("gene")
    for gene in tail_genes:
        gene_table = tail_table.loc[tail_table["gene"] == gene]
        threshold = gene_table["absolute_z_threshold"]
        probability = gene_table["exact_tail_probability"]
        if gene_table.empty or not threshold.is_monotonic_increasing or not probability.is_monotonic_decreasing:
            raise ValueError(f"{gene}: missing or unsorted tail probability curve")
        if not probability.between(0, 1).all() or probability.iloc[0] != 1:
            raise ValueError(f"{gene}: invalid tail probability range or probability at threshold zero")
        z_value = abs(indexed_table.at[gene, "observed_meta_z"])
        p_value = indexed_table.at[gene, "exact_signflip_p_value"]
        matches_z = np.allclose(gene_table["observed_absolute_z"], z_value)
        matches_p = np.allclose(gene_table["observed_exact_p_value"], p_value)
        if not matches_z or not matches_p:
            raise ValueError(f"{gene}: observed point differs from the exact calibration result")

        axis.plot(threshold, probability, color=gene_colors[gene], lw=1.7, label=gene)
        axis.scatter(z_value, p_value, color=gene_colors[gene], s=42, edgecolor="white", linewidth=0.7, zorder=3)
        axis.vlines(z_value, 1e-6, p_value, color=gene_colors[gene], ls=":", lw=0.9)
    axis.axhline(0.05, color="#777777", ls="--", lw=0.8)
    axis.set_yscale("log")
    axis.set_ylim(1e-6, 1.1)
    axis.set_xlabel("Absolute signed Stouffer z")
    axis.set_ylabel("Exact sign-flip tail probability")
    axis.set_title("Exhaustive patient-level sign-flip null distribution", loc="left", pad=8)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)


if __name__ == "__main__":
    main()
