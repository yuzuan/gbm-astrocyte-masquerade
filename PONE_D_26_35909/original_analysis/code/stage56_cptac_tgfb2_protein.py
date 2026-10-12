#!/usr/bin/env python3
"""Direct protein-level validation of TGFB2 in CPTAC-GBM."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, permutation_test, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "raw" / "cptac_gbm" / "Proteome"
RESULT_DIR = ROOT / "results" / "cptac_gbm"
FIGURE_DIR = ROOT / "figures"

MATRIX_FILE = DATA_DIR / "proteome_mssm_per_gene_clean.v4.0.tsv.gz"
DESIGN_FILE = DATA_DIR / "proteome_tmt_design.v4.0.tsv.gz"

LIGANDS = ["TGFB2", "TGFB1", "ANXA1", "SPP1", "CSF1", "CCL2"]
SCORE_DEFINITIONS = {
    "astrocytic_protein_score": [
        "GFAP",
        "AQP4",
        "S100B",
        "SLC1A3",
        "ALDH1L1",
        "GLUL",
        "HOPX",
    ],
    "myeloid_protein_score": [
        "AIF1",
        "CD163",
        "CD14",
        "CSF1R",
        "TYROBP",
        "FCER1G",
        "LST1",
    ],
    "tgfbr_protein_score": ["TGFBR1", "TGFBR2"],
}
RNG = np.random.default_rng(20260716)


def exact_signflip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    observed = abs(values.mean())
    signs = 1 - 2 * (
        (np.arange(2 ** len(values))[:, None] >> np.arange(len(values))) & 1
    )
    null = np.abs((signs * values).mean(axis=1))
    return float(np.mean(null >= observed - 1e-15))


def bh_adjust(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    valid = values.notna()
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if not valid.any():
        return result
    p = values.loc[valid].to_numpy()
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result.loc[valid] = restored
    return result


def bootstrap_median_difference(
    tumor: np.ndarray, normal: np.ndarray, n_resamples: int = 20_000
) -> tuple[float, float]:
    tumor_draws = RNG.choice(tumor, size=(n_resamples, len(tumor)), replace=True)
    normal_draws = RNG.choice(normal, size=(n_resamples, len(normal)), replace=True)
    differences = np.median(tumor_draws, axis=1) - np.median(normal_draws, axis=1)
    return tuple(np.quantile(differences, [0.025, 0.975]).astype(float))


def median_difference(
    tumor: np.ndarray, normal: np.ndarray, axis: int = 0
) -> np.ndarray:
    return np.median(tumor, axis=axis) - np.median(normal, axis=axis)


def group_comparison(
    gene: str, values: pd.Series, design: pd.DataFrame
) -> dict[str, float | int | str]:
    tumor = values[design.loc[values.index, "tumor_normal"] == "tumor"].dropna()
    normal = values[design.loc[values.index, "tumor_normal"] == "normal"].dropna()
    if len(tumor) < 3 or len(normal) < 3:
        return {
            "gene": gene,
            "n_tumor": len(tumor),
            "n_normal": len(normal),
        }
    test = mannwhitneyu(tumor, normal, alternative="two-sided", method="asymptotic")
    permutation = permutation_test(
        (tumor.to_numpy(), normal.to_numpy()),
        median_difference,
        permutation_type="independent",
        alternative="two-sided",
        n_resamples=100_000,
        vectorized=True,
        batch=5_000,
        rng=RNG,
    )
    ci_low, ci_high = bootstrap_median_difference(
        tumor.to_numpy(), normal.to_numpy()
    )
    return {
        "gene": gene,
        "n_tumor": len(tumor),
        "n_normal": len(normal),
        "tumor_median": float(tumor.median()),
        "normal_median": float(normal.median()),
        "median_difference": float(tumor.median() - normal.median()),
        "median_difference_ci_low": ci_low,
        "median_difference_ci_high": ci_high,
        "tumor_mean": float(tumor.mean()),
        "normal_mean": float(normal.mean()),
        "mannwhitney_u": float(test.statistic),
        "mannwhitney_p": float(test.pvalue),
        "median_permutation_p": float(permutation.pvalue),
        "rank_biserial": float(
            2 * test.statistic / (len(tumor) * len(normal)) - 1
        ),
    }


def zscore_columns(frame: pd.DataFrame) -> pd.DataFrame:
    means = frame.mean(axis=0, skipna=True)
    standard_deviations = frame.std(axis=0, skipna=True, ddof=0)
    valid = standard_deviations > 0
    return (frame.loc[:, valid] - means.loc[valid]) / standard_deviations.loc[valid]


def score_from_markers(
    tumor_expression: pd.DataFrame, markers: list[str]
) -> tuple[pd.Series, list[str]]:
    available = [
        marker
        for marker in markers
        if marker in tumor_expression.columns
        and tumor_expression[marker].notna().mean() >= 0.80
    ]
    if not available:
        return pd.Series(np.nan, index=tumor_expression.index), []
    standardized = zscore_columns(tumor_expression[available])
    minimum = max(1, int(np.ceil(0.60 * standardized.shape[1])))
    return standardized.mean(axis=1, skipna=True).where(
        standardized.notna().sum(axis=1) >= minimum
    ), list(standardized.columns)


def association(
    name: str, tgfb2: pd.Series, score: pd.Series, markers: list[str]
) -> dict[str, float | int | str]:
    valid = tgfb2.notna() & score.notna()
    if valid.sum() < 10:
        return {
            "score": name,
            "markers": ";".join(markers),
            "n": int(valid.sum()),
        }
    test = spearmanr(tgfb2.loc[valid], score.loc[valid])
    return {
        "score": name,
        "markers": ";".join(markers),
        "n": int(valid.sum()),
        "spearman_rho": float(test.statistic),
        "spearman_p": float(test.pvalue),
    }


def plex_matched_contrasts(
    values: pd.Series, design: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for plex, plex_design in design.groupby("tmt_plex", sort=True):
        tumor_ids = plex_design.index[plex_design["tumor_normal"] == "tumor"]
        normal_ids = plex_design.index[plex_design["tumor_normal"] == "normal"]
        tumor = values.reindex(tumor_ids).dropna()
        normal = values.reindex(normal_ids).dropna()
        if tumor.empty or normal.empty:
            continue
        rows.append(
            {
                "tmt_plex": int(plex),
                "n_tumor": int(len(tumor)),
                "n_normal": int(len(normal)),
                "tumor_median": float(tumor.median()),
                "normal_median": float(normal.median()),
                "median_difference": float(tumor.median() - normal.median()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_csv(MATRIX_FILE, sep="\t", index_col=0)
    matrix = matrix.drop(columns=["refseq_id"]).transpose()
    matrix.index.name = "case_id"
    design = pd.read_csv(DESIGN_FILE, sep="\t").set_index("case_id")

    missing_design = matrix.index.difference(design.index)
    if len(missing_design):
        raise ValueError(f"Matrix samples missing from design: {list(missing_design)}")
    excluded_after_qc = design.index.difference(matrix.index).tolist()
    design = design.loc[matrix.index]

    coverage_genes = sorted(
        set(LIGANDS)
        | {"TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "SMAD4"}
        | {gene for markers in SCORE_DEFINITIONS.values() for gene in markers}
    )
    coverage = []
    for gene in coverage_genes:
        present = gene in matrix.columns
        values = matrix[gene] if present else pd.Series(np.nan, index=matrix.index)
        coverage.append(
            {
                "gene": gene,
                "present": present,
                "n_quantified": int(values.notna().sum()),
                "fraction_quantified": float(values.notna().mean()),
                "n_tumor_quantified": int(
                    values[design["tumor_normal"] == "tumor"].notna().sum()
                ),
                "n_normal_quantified": int(
                    values[design["tumor_normal"] == "normal"].notna().sum()
                ),
            }
        )
    coverage = pd.DataFrame(coverage)

    comparison_rows = []
    for gene in LIGANDS:
        if gene in matrix.columns:
            comparison_rows.append(group_comparison(gene, matrix[gene], design))
        else:
            comparison_rows.append({"gene": gene, "n_tumor": 0, "n_normal": 0})
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["mannwhitney_q"] = bh_adjust(comparisons["mannwhitney_p"])
    comparisons["median_permutation_q"] = bh_adjust(
        comparisons["median_permutation_p"]
    )

    tumor_ids = design.index[design["tumor_normal"] == "tumor"]
    tumor_expression = matrix.loc[tumor_ids]
    scores = {}
    score_markers = {}
    for name, markers in SCORE_DEFINITIONS.items():
        scores[name], score_markers[name] = score_from_markers(
            tumor_expression, markers
        )
    score_frame = pd.DataFrame(scores)

    associations = pd.DataFrame(
        [
            association(
                name,
                tumor_expression["TGFB2"],
                score_frame[name],
                score_markers[name],
            )
            for name in SCORE_DEFINITIONS
        ]
    )
    associations["spearman_q"] = bh_adjust(associations["spearman_p"])
    tgfb2_plex = plex_matched_contrasts(matrix["TGFB2"], design)
    if tgfb2_plex.empty:
        raise ValueError("No TMT plex contained both tumor and normal TGFB2")

    coverage.to_csv(RESULT_DIR / "protein_coverage.csv", index=False)
    comparisons.to_csv(RESULT_DIR / "ligand_tumor_normal.csv", index=False)
    score_frame.to_csv(RESULT_DIR / "tumor_protein_scores.csv")
    associations.to_csv(RESULT_DIR / "tgfb2_tumor_associations.csv", index=False)
    tgfb2_plex.to_csv(
        RESULT_DIR / "tgfb2_plex_matched_contrasts.csv", index=False
    )

    tgfb2_comparison = comparisons.loc[comparisons["gene"] == "TGFB2"].iloc[0]
    summary = {
        "dataset": "CPTAC-GBM PDC000204",
        "matrix": MATRIX_FILE.name,
        "imputation": "none; clean matrix used",
        "n_tumor": int((design["tumor_normal"] == "tumor").sum()),
        "n_normal": int((design["tumor_normal"] == "normal").sum()),
        "design_samples_excluded_after_proteome_qc": excluded_after_qc,
        "tgfb2": {
            key: (
                value.item()
                if isinstance(value, (np.integer, np.floating))
                else value
            )
            for key, value in tgfb2_comparison.to_dict().items()
        },
        "tgfb2_plex_matched": {
            "n_plexes": int(len(tgfb2_plex)),
            "median_difference": float(tgfb2_plex["median_difference"].median()),
            "n_positive": int((tgfb2_plex["median_difference"] > 0).sum()),
            "exact_signflip_p": exact_signflip_p(
                tgfb2_plex["median_difference"].to_numpy()
            ),
        },
        "associations": associations.to_dict(orient="records"),
    }
    with (RESULT_DIR / "analysis_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)

    colors = {"tumor": "#B33A3A", "normal": "#2A7181"}
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    groups = [
        matrix.loc[design["tumor_normal"] == group, "TGFB2"].dropna().to_numpy()
        for group in ("tumor", "normal")
    ]
    violin = ax.violinplot(groups, positions=[1, 2], showextrema=False)
    for body, group in zip(violin["bodies"], ("tumor", "normal")):
        body.set_facecolor(colors[group])
        body.set_edgecolor("none")
        body.set_alpha(0.35)
    for position, (group, values) in enumerate(
        zip(("tumor", "normal"), groups), start=1
    ):
        jitter = RNG.normal(0, 0.045, len(values))
        ax.scatter(
            position + jitter,
            values,
            s=18,
            alpha=0.75,
            color=colors[group],
            linewidth=0,
        )
        ax.hlines(np.median(values), position - 0.20, position + 0.20, color="black")
    ax.set_xticks([1, 2], ["GBM tumor\nn=99", "normal brain\nn=10"])
    ax.set_ylabel("TGFB2 protein log2 ratio")
    ax.set_title("a  TGFB2 protein in CPTAC-GBM", loc="left", fontweight="bold")
    ax.text(
        0.04,
        0.96,
        (
            f"{len(tgfb2_plex)}/{len(tgfb2_plex)} matched plexes positive; "
            f"p={exact_signflip_p(tgfb2_plex['median_difference'].to_numpy()):.4g}"
        ),
        transform=ax.transAxes,
        va="top",
    )

    for ax, score_name, panel in (
        (axes[0, 1], "astrocytic_protein_score", "b"),
        (axes[1, 0], "myeloid_protein_score", "c"),
    ):
        x = tumor_expression["TGFB2"]
        y = score_frame[score_name]
        valid = x.notna() & y.notna()
        ax.scatter(x[valid], y[valid], s=24, alpha=0.72, color="#287C78")
        if valid.sum() > 2:
            coefficients = np.polyfit(x[valid], y[valid], 1)
            xx = np.linspace(x[valid].min(), x[valid].max(), 100)
            ax.plot(xx, np.polyval(coefficients, xx), color="#B33A3A", linewidth=1.5)
        row = associations.loc[associations["score"] == score_name].iloc[0]
        ax.text(
            0.04,
            0.95,
            f"rho={row['spearman_rho']:.2f}, q={row['spearman_q']:.3g}",
            transform=ax.transAxes,
            va="top",
        )
        ax.set_xlabel("TGFB2 protein log2 ratio")
        ax.set_ylabel(score_name.replace("_", " "))
        ax.set_title(
            f"{panel}  Tumor protein association", loc="left", fontweight="bold"
        )

    ax = axes[1, 1]
    plot_frame = comparisons.dropna(subset=["median_difference"]).copy()
    plot_frame = plot_frame.set_index("gene").loc[LIGANDS].reset_index()
    positions = np.arange(len(plot_frame))
    ax.errorbar(
        plot_frame["median_difference"],
        positions,
        xerr=np.vstack(
            [
                plot_frame["median_difference"]
                - plot_frame["median_difference_ci_low"],
                plot_frame["median_difference_ci_high"]
                - plot_frame["median_difference"],
            ]
        ),
        fmt="o",
        color="#30343B",
        ecolor="#8D99AE",
        capsize=3,
    )
    tgfb2_position = plot_frame.index[plot_frame["gene"] == "TGFB2"][0]
    ax.scatter(
        plot_frame.loc[tgfb2_position, "median_difference"],
        tgfb2_position,
        color="#B33A3A",
        s=52,
        zorder=3,
    )
    ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_yticks(positions, plot_frame["gene"])
    ax.set_xlabel("Tumor minus normal median log2 ratio (95% bootstrap CI)")
    ax.set_title("d  Fixed ligand panel", loc="left", fontweight="bold")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Protein-level validation in CPTAC-GBM",
        x=0.07,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(
        FIGURE_DIR / "Figure6_CPTAC_protein_validation.png",
        dpi=600,
        bbox_inches="tight",
    )
    figure.savefig(
        FIGURE_DIR / "Figure6_CPTAC_protein_validation.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)

    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
