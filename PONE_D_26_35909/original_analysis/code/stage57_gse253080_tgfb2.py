#!/usr/bin/env python3
"""Three-patient spatial sensitivity analysis using raw GSE253080 counts."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import io, sparse
from scipy.stats import spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "raw" / "gse253080"
RESULT_DIR = ROOT / "results" / "gse253080"
FIGURE_DIR = ROOT / "figures"

LIGANDS = ["TGFB2", "TGFB1", "ANXA1", "SPP1", "CSF1", "CCL2"]
RECEPTORS = ["TGFBR1", "TGFBR2"]
AC_MARKERS = ["GFAP", "S100B", "HOPX", "SLC1A3", "MLC1"]
MYELOID_MARKERS = ["AIF1", "CD163", "CSF1R", "LST1", "TYROBP", "FCER1G"]
TARGET_GENES = sorted(
    set(LIGANDS + RECEPTORS + AC_MARKERS + MYELOID_MARKERS)
)
MIN_SPOTS = 100
MIN_SOURCE_SPOTS = 10
RNG = np.random.default_rng(20260716)


def read_gzip_lines(path: Path) -> list[str]:
    with gzip.open(path, "rt") as handle:
        return [line.rstrip("\n") for line in handle]


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


def exact_signflip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    observed = abs(values.mean())
    signs = 1 - 2 * ((np.arange(2 ** len(values))[:, None] >> np.arange(len(values))) & 1)
    null = np.abs((signs * values).mean(axis=1))
    return float(np.mean(null >= observed - 1e-15))


def safe_wilcoxon(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values != 0)]
    if len(values) < 2:
        return np.nan
    return float(wilcoxon(values, alternative="two-sided", method="exact").pvalue)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    standard_deviation = np.nanstd(values)
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        return np.full(len(values), np.nan)
    return (values - np.nanmean(values)) / standard_deviation


def module_score(expression: pd.DataFrame, genes: list[str]) -> np.ndarray:
    available = [
        gene
        for gene in genes
        if gene in expression and expression[gene].std(ddof=0) > 0
    ]
    if len(available) < 2:
        return np.full(len(expression), np.nan)
    standardized = np.column_stack(
        [zscore(expression[gene].to_numpy()) for gene in available]
    )
    return np.nanmean(standardized, axis=1)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < MIN_SPOTS:
        return np.nan
    if np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def rank_residual(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"target": values})
    for index in range(controls.shape[1]):
        frame[f"control_{index}"] = controls[:, index]
    valid = np.isfinite(frame.to_numpy()).all(axis=1)
    result = np.full(len(values), np.nan)
    if valid.sum() < MIN_SPOTS:
        return result
    ranked = frame.loc[valid].rank(method="average") / (valid.sum() + 1)
    y = ranked.pop("target").to_numpy()
    x = ranked.to_numpy()
    keep = np.nanstd(x, axis=0) > 1e-12
    x = x[:, keep]
    x = np.column_stack([np.ones(len(y)), x])
    beta = np.linalg.lstsq(x, y, rcond=1e-10)[0]
    result[valid] = y - np.einsum("ij,j->i", x, beta)
    return result


def neighbor_mean(
    values: np.ndarray, array_row: np.ndarray, array_col: np.ndarray
) -> np.ndarray:
    keys = pd.Series(
        values,
        index=pd.Index(
            [f"{row}:{col}" for row, col in zip(array_row, array_col)]
        ),
    )
    offsets = [(0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    neighbors = []
    for row_offset, col_offset in offsets:
        neighbor_keys = [
            f"{row + row_offset}:{col + col_offset}"
            for row, col in zip(array_row, array_col)
        ]
        neighbors.append(keys.reindex(neighbor_keys).to_numpy())
    neighbor_values = np.column_stack(neighbors)
    present = np.isfinite(neighbor_values).sum(axis=1)
    neighbor_sum = np.nansum(neighbor_values, axis=1)
    result = np.divide(
        neighbor_sum,
        present,
        out=np.full(neighbor_sum.shape, np.nan, dtype=float),
        where=present > 0,
    )
    return result


def sample_from_description(description: str) -> str:
    return str(description).split("_")[0]


def read_sample(gsm: str, sample: str) -> pd.DataFrame:
    prefix = f"{gsm}_{sample}"
    matrix = io.mmread(DATA_DIR / f"{prefix}_matrix.mtx.gz").tocsr()
    features = pd.read_csv(
        DATA_DIR / f"{prefix}_features.tsv.gz",
        sep="\t",
        header=None,
        names=["gene_id", "gene_symbol", "feature_type"],
    )
    barcodes = read_gzip_lines(DATA_DIR / f"{prefix}_barcodes.tsv.gz")
    positions = pd.read_csv(
        DATA_DIR / f"{prefix}_tissue_positions_list.csv.gz",
        header=None,
        names=[
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pixel_row",
            "pixel_col",
        ],
    ).set_index("barcode")

    if matrix.shape == (len(features), len(barcodes)):
        matrix = matrix.transpose().tocsr()
    elif matrix.shape != (len(barcodes), len(features)):
        raise ValueError(f"{sample}: unexpected matrix shape {matrix.shape}")

    positions = positions.reindex(barcodes)
    keep = positions["in_tissue"].fillna(0).to_numpy() == 1
    matrix = matrix[keep]
    positions = positions.iloc[np.flatnonzero(keep)].copy()
    positions.index.name = "barcode"

    gene_symbols = features["gene_symbol"].astype(str).str.upper().to_numpy()
    selected = {}
    for gene in TARGET_GENES:
        columns = np.flatnonzero(gene_symbols == gene)
        if len(columns):
            selected[gene] = np.asarray(matrix[:, columns].sum(axis=1)).ravel()
    library_size = np.asarray(matrix.sum(axis=1)).ravel()
    library_size = np.maximum(library_size, 1)
    expression = pd.DataFrame(selected, index=positions.index)
    expression = np.log1p(expression.div(library_size, axis=0) * 10_000)
    expression["library_size"] = library_size
    for column in positions.columns:
        expression[column] = positions[column].to_numpy()
    return expression


def analyze_sample(
    gsm: str, sample: str, patient: str, include_plot: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    expression = read_sample(gsm, sample)
    ac_score = module_score(expression, AC_MARKERS)
    myeloid_score = module_score(expression, MYELOID_MARKERS)
    receptor_score = module_score(expression, RECEPTORS)
    receiver_score = np.nanmean(
        np.column_stack([zscore(myeloid_score), zscore(receptor_score)]), axis=1
    )
    receiver_neighbor = neighbor_mean(
        receiver_score,
        expression["array_row"].to_numpy(),
        expression["array_col"].to_numpy(),
    )
    controls = np.column_stack(
        [np.log1p(expression["library_size"].to_numpy()), myeloid_score]
    )
    ac_residual = rank_residual(ac_score, controls)

    ligand_rows = []
    for ligand in LIGANDS:
        values = expression[ligand].to_numpy()
        ligand_residual = rank_residual(values, controls)
        positive = values > 0
        high = positive & (values >= np.median(values[positive])) if positive.any() else positive
        zero = values == 0
        ligand_rows.append(
            {
                "dataset": "GSE253080",
                "patient": patient,
                "sample": sample,
                "ligand": ligand,
                "n_spots": len(expression),
                "detection_fraction": float(np.mean(positive)),
                "rho_ac_score": safe_spearman(values, ac_score),
                "rho_ac_score_adjusted": safe_spearman(
                    ligand_residual, ac_residual
                ),
                "high_minus_zero_ac_score": (
                    float(np.nanmean(ac_score[high]) - np.nanmean(ac_score[zero]))
                    if high.sum() >= MIN_SOURCE_SPOTS and zero.sum() >= MIN_SOURCE_SPOTS
                    else np.nan
                ),
            }
        )

    tgfb2 = expression["TGFB2"].to_numpy()
    ac_high = ac_score >= np.nanquantile(ac_score, 0.75)
    positive_source = ac_high & (tgfb2 > 0) & np.isfinite(receiver_neighbor)
    high_source = np.zeros(len(expression), dtype=bool)
    if positive_source.sum() >= MIN_SOURCE_SPOTS:
        high_source = positive_source & (
            tgfb2 >= np.nanmedian(tgfb2[positive_source])
        )
    zero_source = ac_high & (tgfb2 == 0) & np.isfinite(receiver_neighbor)
    enough = (
        high_source.sum() >= MIN_SOURCE_SPOTS
        and zero_source.sum() >= MIN_SOURCE_SPOTS
    )
    proximity = pd.DataFrame(
        [
            {
                "dataset": "GSE253080",
                "patient": patient,
                "sample": sample,
                "n_spots": len(expression),
                "n_tgfb2_high_ac": int(high_source.sum()),
                "n_tgfb2_zero_ac": int(zero_source.sum()),
                "tgfb2_neighbor_receiver_delta": (
                    float(
                        np.nanmean(receiver_neighbor[high_source])
                        - np.nanmean(receiver_neighbor[zero_source])
                    )
                    if enough
                    else np.nan
                ),
            }
        ]
    )

    plot_data = None
    if include_plot:
        plot_data = pd.DataFrame(
            {
                "pixel_col": expression["pixel_col"],
                "pixel_row": expression["pixel_row"],
                "tgfb2": tgfb2,
                "ac_score": ac_score,
                "receiver_score": receiver_score,
            }
        )
    return pd.DataFrame(ligand_rows), proximity, plot_data


def metric_summary(values: pd.Series, prefix: str) -> dict[str, float | int]:
    values = values.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return {
        f"{prefix}_n_patients": int(len(values)),
        f"{prefix}_median": float(np.median(values)) if len(values) else np.nan,
        f"{prefix}_mean": float(np.mean(values)) if len(values) else np.nan,
        f"{prefix}_n_positive": int(np.sum(values > 0)),
        f"{prefix}_signflip_p": exact_signflip_p(values),
        f"{prefix}_wilcoxon_p": safe_wilcoxon(values),
    }


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(DATA_DIR / "pd.csv")
    metadata["sample"] = metadata["description"].map(sample_from_description)
    metadata = metadata.rename(columns={"subject id:ch1": "patient"})

    ligand_results = []
    proximity_results = []
    for index, row in metadata.iterrows():
        print(f"[{index + 1:02d}/{len(metadata):02d}] {row['sample']}", flush=True)
        ligand, proximity, _ = analyze_sample(
            row["geo_accession"], row["sample"], row["patient"]
        )
        ligand_results.append(ligand)
        proximity_results.append(proximity)

    section_ligand = pd.concat(ligand_results, ignore_index=True)
    section_proximity = pd.concat(proximity_results, ignore_index=True)
    patient_ligand = (
        section_ligand.groupby(["patient", "ligand"], as_index=False)
        .agg(
            rho_ac_score=("rho_ac_score", "median"),
            rho_ac_score_adjusted=("rho_ac_score_adjusted", "median"),
            high_minus_zero_ac_score=("high_minus_zero_ac_score", "median"),
        )
    )
    patient_proximity = (
        section_proximity.groupby("patient", as_index=False)
        .agg(
            tgfb2_neighbor_receiver_delta=(
                "tgfb2_neighbor_receiver_delta",
                "median",
            )
        )
    )

    summary_rows = []
    for ligand in LIGANDS:
        group = patient_ligand.loc[patient_ligand["ligand"] == ligand]
        summary = {"ligand": ligand}
        summary.update(metric_summary(group["rho_ac_score"], "rho_ac_score"))
        summary.update(
            metric_summary(
                group["rho_ac_score_adjusted"], "rho_ac_score_adjusted"
            )
        )
        summary.update(
            metric_summary(
                group["high_minus_zero_ac_score"], "high_minus_zero_ac_score"
            )
        )
        summary_rows.append(summary)
    ligand_summary = pd.DataFrame(summary_rows)
    for column in [
        "rho_ac_score_signflip_p",
        "rho_ac_score_adjusted_signflip_p",
        "high_minus_zero_ac_score_signflip_p",
    ]:
        ligand_summary[column.replace("_p", "_q")] = bh_adjust(
            ligand_summary[column]
        )

    proximity_summary = metric_summary(
        patient_proximity["tgfb2_neighbor_receiver_delta"],
        "tgfb2_neighbor_receiver_delta",
    )
    tgfb2_summary = ligand_summary.loc[
        ligand_summary["ligand"] == "TGFB2"
    ].iloc[0]
    summary = {
        "dataset": "GSE253080",
        "n_sections": int(len(metadata)),
        "n_patients": int(metadata["patient"].nunique()),
        "tgfb2": {
            key: (
                value.item()
                if isinstance(value, (np.integer, np.floating))
                else value
            )
            for key, value in tgfb2_summary.to_dict().items()
        },
        "proximity": proximity_summary,
        "inference_limit": "three patients; directional sensitivity analysis only",
    }

    section_ligand.to_csv(RESULT_DIR / "section_ligand_metrics.csv", index=False)
    patient_ligand.to_csv(RESULT_DIR / "patient_ligand_metrics.csv", index=False)
    ligand_summary.to_csv(RESULT_DIR / "ligand_summary.csv", index=False)
    section_proximity.to_csv(
        RESULT_DIR / "section_spatial_metrics.csv", index=False
    )
    patient_proximity.to_csv(
        RESULT_DIR / "patient_spatial_metrics.csv", index=False
    )
    with (RESULT_DIR / "analysis_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)

    tgfb2_sections = section_ligand.loc[
        (section_ligand["ligand"] == "TGFB2")
        & section_ligand["rho_ac_score_adjusted"].notna()
    ]
    cohort_median = tgfb2_sections["rho_ac_score_adjusted"].median()
    representative_sample = tgfb2_sections.iloc[
        (tgfb2_sections["rho_ac_score_adjusted"] - cohort_median)
        .abs()
        .argmin()
    ]["sample"]
    row = metadata.loc[metadata["sample"] == representative_sample].iloc[0]
    _, _, representative = analyze_sample(
        row["geo_accession"],
        row["sample"],
        row["patient"],
        include_plot=True,
    )
    representative.insert(0, "sample", representative_sample)
    representative.to_csv(RESULT_DIR / "representative_plot.csv", index=False)

    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    ax = axes[0, 0]
    values = patient_ligand.loc[
        patient_ligand["ligand"] == "TGFB2", "rho_ac_score_adjusted"
    ].to_numpy()
    ax.scatter(
        RNG.normal(1, 0.035, len(values)),
        values,
        color="#267D83",
        s=55,
    )
    ax.hlines(np.median(values), 0.82, 1.18, color="#B33A3A", linewidth=2.5)
    ax.axhline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlim(0.65, 1.35)
    ax.set_xticks([1], ["3 patient medians"])
    ax.set_ylabel("Adjusted within-section Spearman rho")
    ax.set_title("a  TGFB2 vs AC-like score", loc="left", fontweight="bold")

    ax = axes[0, 1]
    for index, ligand in enumerate(LIGANDS):
        ligand_values = patient_ligand.loc[
            patient_ligand["ligand"] == ligand, "rho_ac_score_adjusted"
        ].dropna()
        ax.scatter(
            RNG.normal(index, 0.045, len(ligand_values)),
            ligand_values,
            color="#6C7A89",
            alpha=0.75,
            s=30,
        )
        ax.hlines(
            ligand_values.median(),
            index - 0.20,
            index + 0.20,
            color="#B33A3A" if ligand == "TGFB2" else "#222222",
            linewidth=2,
        )
    ax.axhline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(LIGANDS)), LIGANDS, rotation=45, ha="right")
    ax.set_ylabel("Adjusted within-section Spearman rho")
    ax.set_title("b  Fixed ligand panel", loc="left", fontweight="bold")

    ax = axes[0, 2]
    proximity_values = patient_proximity[
        "tgfb2_neighbor_receiver_delta"
    ].dropna()
    ax.scatter(
        RNG.normal(1, 0.035, len(proximity_values)),
        proximity_values,
        color="#6A4C93",
        s=55,
    )
    if len(proximity_values):
        ax.hlines(
            proximity_values.median(), 0.82, 1.18, color="#B33A3A", linewidth=2.5
        )
    ax.axhline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlim(0.65, 1.35)
    ax.set_xticks([1], ["patient medians"])
    ax.set_ylabel("Neighbor receiver score difference")
    ax.set_title("c  Descriptive proximity", loc="left", fontweight="bold")

    def spatial_panel(ax, values: pd.Series, title: str, palette: str) -> None:
        scatter = ax.scatter(
            representative["pixel_col"],
            -representative["pixel_row"],
            c=values,
            s=4,
            cmap=palette,
            linewidth=0,
        )
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, loc="left", fontweight="bold")
        figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)

    spatial_panel(axes[1, 0], representative["ac_score"], "d  AC-like score", "viridis")
    spatial_panel(axes[1, 1], representative["tgfb2"], "e  TGFB2 expression", "magma")
    spatial_panel(
        axes[1, 2],
        representative["receiver_score"],
        "f  Myeloid/TGFBR score",
        "cividis",
    )
    for ax in axes[0]:
        ax.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"GSE253080 spatial sensitivity analysis; representative {representative_sample}",
        x=0.06,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(
        FIGURE_DIR / "SupplementaryFigureS7_GSE253080_spatial_validation.png",
        dpi=600,
        bbox_inches="tight",
    )
    figure.savefig(
        FIGURE_DIR / "SupplementaryFigureS7_GSE253080_spatial_validation.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
