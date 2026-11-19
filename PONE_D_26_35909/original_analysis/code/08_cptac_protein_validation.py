#!/usr/bin/env python3
# Purpose: Compare ligands in tumor and normal brain in the clean CPTAC-GBM protein matrix. Test within-plex
# contrasts and associations among tumors.
# Input: --input-dir contains proteome_mssm_per_gene_clean.v4.0.tsv.gz (gene, refseq_id and sample columns)
# and proteome_tmt_design.v4.0.tsv.gz (case_id, tumor_normal and tmt_plex). Use the original NumPy/SciPy random tests.
# Output: external_validation/08_cptac_protein_validation/ contains 01_protein_coverage.csv,
# 02_tumor_normal_ligand_comparison.csv, 03_tumor_protein_scores.csv and 04_tgfb2_protein_associations.csv,
# 05_tgfb2_plex_matched_contrasts.csv, 06_tgfb2_sample_values.csv, 07_analysis_summary.json,
# 08_figure6_cptac_protein_validation.pdf and 08_figure6_cptac_protein_validation.png.
# Regenerated tables retain the case identifiers from the input.
# Usage: python 08_cptac_protein_validation.py --input-dir PROTEOME_DIRECTORY [--output-dir OUTPUT_DIRECTORY]
# Optional --matrix-url and --design-url accept direct HTTPS links. Use the source-specific raw/ cache on a local miss.

import argparse
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, permutation_test, spearmanr

spatial = import_module("00a_spatial_analysis")
files = import_module("00a_project_paths")


def main():
    """Match the clean matrix to the TMT design and write protein comparisons and figures.
    Stop on unmatched cases or absence of a paired plex. No return value.
    """
    external_dir = Path(__file__).resolve().parents[1] / "external_validation"
    parser = argparse.ArgumentParser(description="CPTAC-GBM protein validation")
    parser.add_argument("--input-dir", type=Path, default=external_dir / "raw/cptac_gbm/Proteome")
    parser.add_argument("--output-dir", type=Path, default=external_dir / Path(__file__).stem)
    parser.add_argument("--matrix-url", help="Direct HTTPS URL for the clean protein matrix")
    parser.add_argument("--design-url", help="Direct HTTPS URL for the original TMT design table")
    args = parser.parse_args()

    # Transpose gene rows to sample rows after removing refseq_id. Duplicate genes or cases would count observations
    # twice.
    matrix_path = spatial.cptac_input(
        args.input_dir / "proteome_mssm_per_gene_clean.v4.0.tsv.gz", args.matrix_url, args.output_dir / "raw",
        ["gene", "refseq_id"]
    )
    protein_table = pd.read_csv(matrix_path, sep="\t")
    expression_table = protein_table.set_index("gene", verify_integrity=True).drop(columns="refseq_id").transpose()
    expression_table.index.name = "case_id"
    design_path = spatial.cptac_input(
        args.input_dir / "proteome_tmt_design.v4.0.tsv.gz", args.design_url, args.output_dir / "raw",
        ["case_id", "tumor_normal", "tmt_plex"]
    )
    design_table = pd.read_csv(design_path, sep="\t", dtype={"case_id": str})
    design_table = design_table.set_index("case_id", verify_integrity=True)

    # Use samples in the clean matrix without imputation. Record design cases excluded by protein QC in the summary.
    if not expression_table.index.isin(design_table.index).all():
        raise ValueError("The clean protein matrix contains cases absent from the TMT design")
    excluded = design_table.index.difference(expression_table.index).tolist()
    design_table = design_table.loc[expression_table.index]
    if not design_table["tumor_normal"].isin(["tumor", "normal"]).all():
        raise ValueError("Unrecognized tissue group in tumor_normal")
    is_tumor = design_table["tumor_normal"] == "tumor"
    is_normal = design_table["tumor_normal"] == "normal"

    # Report coverage of all six ligands, receptors, SMADs and score markers. Retain a row for missing genes.
    marker_sets = {
        "astrocytic_protein_score": ["GFAP", "AQP4", "S100B", "SLC1A3", "ALDH1L1", "GLUL", "HOPX"],
        "myeloid_protein_score": ["AIF1", "CD163", "CD14", "CSF1R", "TYROBP", "FCER1G", "LST1"],
        "tgfbr_protein_score": spatial.receptors,
    }
    coverage_genes = set(spatial.ligands + spatial.receptors + ["SMAD2", "SMAD3", "SMAD4"]).union(*marker_sets.values())
    coverage_rows = []
    for gene in sorted(coverage_genes):
        present = gene in expression_table
        values = expression_table[gene] if present else pd.Series(np.nan, index=expression_table.index)
        coverage_rows.append({
            "gene": gene, "present": present, "n_quantified": int(values.notna().sum()),
            "fraction_quantified": values.notna().mean(), "n_tumor_quantified": int(values[is_tumor].notna().sum()),
            "n_normal_quantified": int(values[is_normal].notna().sum()),
        })
    coverage_table = pd.DataFrame(coverage_rows)

    # Keep ligand and RNG order: permute first, then bootstrap tumors and normal samples. Draw plot jitter last.
    rng = np.random.default_rng(20260716)
    comparison_rows = []

    # SciPy passes an axis for batched permutations. Return the tumor-minus-normal median difference for each batch.
    def median_difference(tumor, normal, axis):
        difference = np.median(tumor, axis=axis) - np.median(normal, axis=axis)
        return difference

    for gene in spatial.ligands:
        values = expression_table[gene] if gene in expression_table else pd.Series(np.nan, index=expression_table.index)
        tumor = values[is_tumor].dropna().to_numpy()
        normal = values[is_normal].dropna().to_numpy()
        record = {"gene": gene, "n_tumor": len(tumor), "n_normal": len(normal)}
        comparison_rows.append(record)
        if min(len(tumor), len(normal)) < 3:
            continue
        test = mannwhitneyu(tumor, normal, alternative="two-sided", method="asymptotic")
        permutation = permutation_test(
            (tumor, normal), median_difference, permutation_type="independent", alternative="two-sided",
            n_resamples=100_000, vectorized=True, batch=5_000, rng=rng
        )

        # Resample each group with replacement 20,000 times. Use the 2.5th and 97.5th percentiles of the median
        # differences.
        tumor_draws = rng.choice(tumor, size=(20_000, len(tumor)), replace=True)
        normal_draws = rng.choice(normal, size=(20_000, len(normal)), replace=True)
        differences = np.median(tumor_draws, axis=1) - np.median(normal_draws, axis=1)
        ci_low, ci_high = np.quantile(differences, [0.025, 0.975])
        record.update({
            "tumor_median": np.median(tumor), "normal_median": np.median(normal),
            "median_difference": np.median(tumor) - np.median(normal), "median_difference_ci_low": ci_low,
            "median_difference_ci_high": ci_high, "tumor_mean": np.mean(tumor), "normal_mean": np.mean(normal),
            "mannwhitney_u": test.statistic, "mannwhitney_p": test.pvalue, "median_permutation_p": permutation.pvalue,
            "rank_biserial": 2 * test.statistic / (len(tumor) * len(normal)) - 1,
        })

    # Retain sample sizes and missing statistics for undersampled ligands. Apply BH across ligands for each test
    # separately.
    comparison_columns = [
        "gene", "n_tumor", "n_normal", "tumor_median", "normal_median", "median_difference", "median_difference_ci_low",
        "median_difference_ci_high", "tumor_mean", "normal_mean", "mannwhitney_u", "mannwhitney_p",
        "median_permutation_p", "rank_biserial",
    ]
    comparison_table = pd.DataFrame(comparison_rows, columns=comparison_columns)
    for test_name in ["mannwhitney", "median_permutation"]:
        comparison_table[f"{test_name}_q"] = spatial.bh_adjust(comparison_table[f"{test_name}_p"])

    # Keep markers quantified in at least 80% of tumors. Remove zero-variance columns and scale by population SD.
    tumor_table = expression_table.loc[is_tumor]
    score_table = pd.DataFrame(index=tumor_table.index)
    association_rows = []
    for name, markers in marker_sets.items():
        available = [gene for gene in markers if gene in tumor_table and tumor_table[gene].notna().mean() >= 0.80]
        marker_table = tumor_table[available]
        means = marker_table.mean(axis=0, skipna=True)
        deviations = marker_table.std(axis=0, skipna=True, ddof=0)
        used = deviations[deviations > 0].index
        standardized_table = (marker_table[used] - means[used]) / deviations[used]

        # Require 60% of selected markers per case and ten valid pairs for correlation. Leave a score missing if no
        # marker qualifies.
        minimum = max(1, int(np.ceil(0.60 * len(used))))
        score = standardized_table.mean(axis=1).where(standardized_table.notna().sum(axis=1) >= minimum)
        score_table[name] = score
        valid = tumor_table["TGFB2"].notna() & score.notna()
        record = {
            "score": name, "markers": ";".join(used), "n": int(valid.sum()),
            "spearman_rho": np.nan, "spearman_p": np.nan,
        }
        if valid.sum() >= 10:
            test = spearmanr(tumor_table.loc[valid, "TGFB2"], score[valid])
            record.update(spearman_rho=test.statistic, spearman_p=test.pvalue)
        association_rows.append(record)
    association_table = pd.DataFrame(association_rows)
    association_table["spearman_q"] = spatial.bh_adjust(association_table["spearman_p"])

    # Use the within-plex tumor-minus-normal median difference. Each plex must contain both groups and is one sign-flip
    # unit.
    plex_rows = []
    for plex, group_table in design_table.groupby("tmt_plex", sort=True):
        tumor = expression_table.loc[group_table.index[group_table["tumor_normal"] == "tumor"], "TGFB2"].dropna()
        normal = expression_table.loc[group_table.index[group_table["tumor_normal"] == "normal"], "TGFB2"].dropna()
        if tumor.empty or normal.empty:
            continue
        plex_rows.append({
            "tmt_plex": int(plex), "n_tumor": len(tumor), "n_normal": len(normal), "tumor_median": tumor.median(),
            "normal_median": normal.median(), "median_difference": tumor.median() - normal.median(),
        })
    if not plex_rows:
        raise ValueError("No TMT plex quantified TGFB2 in both tumor and normal brain")
    plex_table = pd.DataFrame(plex_rows)
    plex_summary = {
        "n_plexes": len(plex_table), "median_difference": float(plex_table["median_difference"].median()),
        "n_positive": int((plex_table["median_difference"] > 0).sum()),
        "exact_signflip_p": spatial.signflip_p(plex_table["median_difference"]),
    }

    # Each Fig. 6a point comes directly from the clean matrix. Matching released P/N codes requires a separate
    # crosswalk.
    sample_table = pd.DataFrame({"group": design_table["tumor_normal"], "tgfb2_log2_ratio": expression_table["TGFB2"]})
    sample_table = sample_table.dropna(subset=["tgfb2_log2_ratio"]).reset_index()
    tgfb2_result = comparison_table.set_index("gene", drop=False).loc["TGFB2"]
    for name in ["tumor", "normal"]:
        if (sample_table["group"] == name).sum() != tgfb2_result[f"n_{name}"]:
            raise ValueError(f"The {name} TGFB2 point count differs from the group comparison")
    group_counts = coverage_table["n_tumor_quantified"] + coverage_table["n_normal_quantified"]
    if not coverage_table["n_quantified"].eq(group_counts).all():
        raise ValueError("Total protein quantifications differ from the sum of the two groups")

    files.write_table(coverage_table, args.output_dir / "01_protein_coverage.csv", ["gene"])
    files.write_table(comparison_table, args.output_dir / "02_tumor_normal_ligand_comparison.csv", ["gene"])
    files.write_table(score_table.reset_index(), args.output_dir / "03_tumor_protein_scores.csv", ["case_id"])
    files.write_table(association_table, args.output_dir / "04_tgfb2_protein_associations.csv", ["score"])
    files.write_table(plex_table, args.output_dir / "05_tgfb2_plex_matched_contrasts.csv", ["tmt_plex"])
    files.write_table(sample_table, args.output_dir / "06_tgfb2_sample_values.csv", ["case_id"])

    # Retain QC exclusions and published summary fields. Store inestimable statistics as JSON null.
    json_association_table = association_table.astype(object).where(association_table.notna(), None)
    summary = {
        "dataset": "CPTAC-GBM PDC000204", "matrix": matrix_path.name, "imputation": "none; clean matrix used",
        "n_tumor": int(is_tumor.sum()), "n_normal": int(is_normal.sum()),
        "design_samples_excluded_after_proteome_qc": excluded,
        "tgfb2": tgfb2_result.astype(object).where(tgfb2_result.notna(), None).to_dict(),
        "tgfb2_plex_matched": plex_summary, "associations": json_association_table.to_dict(orient="records"),
    }
    files.write_json(summary, args.output_dir / "07_analysis_summary.json")

    # Get tissue groups, point counts and positive plex counts from the samples. Jitter points in their original group
    # order.
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axis = axes[0, 0]
    groups = []
    for name in ["tumor", "normal"]:
        groups.append(sample_table.loc[sample_table["group"] == name, "tgfb2_log2_ratio"].to_numpy())
    colors = ["#B33A3A", "#2A7181"]
    violin = axis.violinplot(groups, positions=[1, 2], showextrema=False)
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.35)
    for i, (values, color) in enumerate(zip(groups, colors), start=1):
        axis.scatter(i + rng.normal(0, 0.045, len(values)), values, s=18, alpha=0.75, color=color, linewidth=0)
        axis.hlines(np.median(values), i - 0.20, i + 0.20, color="black")
    axis.set_xticks([1, 2], [f"GBM tumor\nn={len(groups[0])}", f"normal brain\nn={len(groups[1])}"])
    axis.set_ylabel("TGFB2 protein log2 ratio")
    axis.set_title("a  TGFB2 protein in CPTAC-GBM", loc="left", fontweight="bold")
    axis.text(
        0.04, 0.96, f"{plex_summary['n_positive']}/{len(plex_table)} matched plexes positive; "
        f"p={plex_summary['exact_signflip_p']:.4g}", transform=axis.transAxes, va="top"
    )

    # Label each tumor association with its Spearman rho and BH q value. The line is a visual guide.
    score_panels = [(axes[0, 1], "astrocytic_protein_score", "b"), (axes[1, 0], "myeloid_protein_score", "c")]
    for axis, name, letter in score_panels:
        values = tumor_table["TGFB2"]
        score = score_table[name]
        valid = values.notna() & score.notna()
        axis.scatter(values[valid], score[valid], s=24, alpha=0.72, color="#287C78")
        if valid.sum() > 2 and values[valid].nunique() > 1:
            coefficients = np.polyfit(values[valid], score[valid], 1)
            grid = np.linspace(values[valid].min(), values[valid].max(), 100)
            axis.plot(grid, np.polyval(coefficients, grid), color="#B33A3A", linewidth=1.5)
        stats = association_table.set_index("score").loc[name]
        label = f"rho={stats['spearman_rho']:.2f}, q={stats['spearman_q']:.3g}"
        axis.text(0.04, 0.95, label, transform=axis.transAxes, va="top")
        axis.set(xlabel="TGFB2 protein log2 ratio", ylabel=name.replace("_", " "))
        axis.set_title(f"{letter}  Tumor protein association", loc="left", fontweight="bold")

    # Show each median difference with its bootstrap interval. Keep the axis label without a point or interval for
    # missing ligands.
    axis = axes[1, 1]
    positions = np.arange(len(comparison_table))
    axis.hlines(
        positions, comparison_table["median_difference_ci_low"], comparison_table["median_difference_ci_high"],
        color="#8D99AE"
    )
    for column in ["median_difference_ci_low", "median_difference_ci_high"]:
        axis.plot(comparison_table[column], positions, marker="|", color="#8D99AE", linestyle="none", markersize=6)
    axis.scatter(comparison_table["median_difference"], positions, color="#30343B", s=24)
    is_tgfb2 = comparison_table["gene"] == "TGFB2"
    axis.scatter(comparison_table.loc[is_tgfb2, "median_difference"], positions[is_tgfb2], color="#B33A3A", s=52)
    axis.axvline(0, color="#777777", linestyle="--", linewidth=1)
    axis.set_yticks(positions, comparison_table["gene"])
    axis.set_xlabel("Tumor minus normal median log2 ratio (95% bootstrap CI)")
    axis.set_title("d  Fixed ligand panel", loc="left", fontweight="bold")

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Protein-level validation in CPTAC-GBM", x=0.07, ha="left", fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    files.save_figure(figure, args.output_dir / "08_figure6_cptac_protein_validation.pdf")
    plt.close(figure)
    print(f"{int(is_tumor.sum())} tumors, {int(is_normal.sum())} normal controls, {len(plex_table)} plex contrasts")


if __name__ == "__main__":
    main()
