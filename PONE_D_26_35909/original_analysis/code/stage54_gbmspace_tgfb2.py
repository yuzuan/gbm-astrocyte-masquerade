#!/usr/bin/env python3
"""Patient-level spatial validation of AC-like TGFB2 in GBM-Space."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "raw" / "gbmspace" / "extracted" / "anndata"
RESULT_DIR = ROOT / "results" / "gbmspace"
FIGURE_DIR = ROOT / "figures"

LIGANDS = ["TGFB2", "TGFB1", "ANXA1", "SPP1", "CSF1", "CCL2"]
RECEPTORS = ["TGFBR1", "TGFBR2"]
AC_MARKERS = ["GFAP", "S100B", "HOPX", "SLC1A3", "MLC1"]
TARGET_GENES = LIGANDS + RECEPTORS + AC_MARKERS

MIN_SPOTS = 100
MIN_GROUP_SPOTS = 20
MALIGNANT_FRACTION_THRESHOLD = 0.20
RNG = np.random.default_rng(20260716)


def dense(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


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
    n = len(values)
    if n <= 20:
        signs = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
        null = np.abs((signs * values).mean(axis=1))
        return float(np.mean(null >= observed - 1e-15))
    null = np.empty(100_000, dtype=float)
    for start in range(0, len(null), 5_000):
        stop = min(start + 5_000, len(null))
        signs = RNG.choice((-1.0, 1.0), size=(stop - start, n))
        null[start:stop] = np.abs((signs * values).mean(axis=1))
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def safe_wilcoxon(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values != 0]
    if len(values) < 2:
        return np.nan
    method = "exact" if len(values) <= 25 else "auto"
    return float(wilcoxon(values, alternative="two-sided", method=method).pvalue)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < MIN_SPOTS:
        return np.nan
    if np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def rank_residual(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"target": values})
    for i in range(controls.shape[1]):
        frame[f"control_{i}"] = controls[:, i]
    valid = np.isfinite(frame.to_numpy()).all(axis=1)
    result = np.full(len(values), np.nan)
    if valid.sum() < MIN_SPOTS:
        return result
    ranked = frame.loc[valid].rank(method="average") / (valid.sum() + 1.0)
    y = ranked["target"].to_numpy(dtype=float)
    controls_ranked = ranked.drop(columns="target").to_numpy(dtype=float)
    means = controls_ranked.mean(axis=0)
    standard_deviations = controls_ranked.std(axis=0)
    keep = standard_deviations > 1e-12
    controls_ranked = (
        controls_ranked[:, keep] - means[keep]
    ) / standard_deviations[keep]
    x = np.column_stack([np.ones(len(y)), controls_ranked])
    beta = np.linalg.lstsq(x, y, rcond=1e-10)[0]
    result[valid] = y - np.einsum("ij,j->i", x, beta)
    return result


def is_malignant_state(name: str) -> bool:
    prefixes = (
        "AC progenitor-like",
        "AC-gliosis-like",
        "Hypoxic ",
        "NPC-neuronal-like",
        "OPC-NPC-like",
        "OPC-like ",
        "OPC-neuronal-like",
        "Proliferative AC-OPC-like",
        "Proliferative NPC-OPC-like",
        "Proliferative nIPC-like",
        "Undefined ",
    )
    return name.startswith(prefixes)


def state_masks(names: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "ac": np.array(
            [
                name.startswith(("AC progenitor-like", "AC-gliosis-like"))
                for name in names
            ]
        ),
        "ac_broad": np.array(
            [
                name.startswith(
                    (
                        "AC progenitor-like",
                        "AC-gliosis-like",
                        "Proliferative AC-OPC-like",
                    )
                )
                for name in names
            ]
        ),
        "malignant": np.array([is_malignant_state(name) for name in names]),
        "myeloid": np.array(
            [
                ("TAMs" in name) or name in {"Monocytes", "Dendritic cells"}
                for name in names
            ]
        ),
        "normal_astro": np.array(
            [
                ("astrocytes" in name.lower())
                or name in {"Gliosis-like", "Neural-support astrocytes"}
                for name in names
            ]
        ),
    }


def feature_indices(
    names: np.ndarray, feature_types: np.ndarray, targets: list[str], feature_type: str
) -> list[int]:
    result = []
    for target in targets:
        hits = np.flatnonzero((names == target) & (feature_types == feature_type))
        if len(hits) != 1:
            raise ValueError(
                f"{target}: expected one {feature_type} feature, found {len(hits)}"
            )
        result.append(int(hits[0]))
    return result


def neighbor_mean(values: np.ndarray, coordinates: np.ndarray) -> tuple[np.ndarray, float]:
    tree = cKDTree(coordinates)
    distances, _ = tree.query(coordinates, k=2)
    nearest = distances[:, 1]
    spacing = float(np.median(nearest[np.isfinite(nearest) & (nearest > 0)]))
    radius = spacing * 1.25
    neighborhoods = tree.query_ball_point(coordinates, r=radius)
    result = np.full(len(values), np.nan)
    for i, neighbors in enumerate(neighborhoods):
        neighbors = [j for j in neighbors if j != i and np.isfinite(values[j])]
        if neighbors:
            result[i] = float(np.mean(values[neighbors]))
    return result, spacing


def group_delta(values: np.ndarray, grouping: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    subset = valid & np.isfinite(values) & np.isfinite(grouping)
    if subset.sum() < 2 * MIN_GROUP_SPOTS:
        return {"delta": np.nan, "n_high": 0, "n_low": 0}
    q1, q3 = np.quantile(grouping[subset], [0.25, 0.75])
    low = subset & (grouping <= q1)
    high = subset & (grouping >= q3)
    if low.sum() < MIN_GROUP_SPOTS or high.sum() < MIN_GROUP_SPOTS:
        return {"delta": np.nan, "n_high": int(high.sum()), "n_low": int(low.sum())}
    return {
        "delta": float(np.mean(values[high]) - np.mean(values[low])),
        "n_high": int(high.sum()),
        "n_low": int(low.sum()),
    }


def tgfb2_neighbor_delta(
    tgfb2: np.ndarray,
    ac_fraction: np.ndarray,
    neighbor_receiver: np.ndarray,
    receiver: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    ac_rich = valid & np.isfinite(tgfb2) & np.isfinite(ac_fraction)
    if ac_rich.sum() < 4 * MIN_GROUP_SPOTS:
        return {
            "neighbor_delta": np.nan,
            "same_spot_delta": np.nan,
            "n_high": 0,
            "n_low": 0,
        }
    ac_cut = np.median(ac_fraction[ac_rich])
    ac_rich &= ac_fraction >= ac_cut
    positive = ac_rich & (tgfb2 > 0)
    zero = ac_rich & (tgfb2 == 0)
    if positive.sum() >= MIN_GROUP_SPOTS and zero.sum() >= MIN_GROUP_SPOTS:
        high_cut = np.median(tgfb2[positive])
        high = positive & (tgfb2 >= high_cut) & np.isfinite(neighbor_receiver)
        low = zero & np.isfinite(neighbor_receiver)
    else:
        q1, q3 = np.quantile(tgfb2[ac_rich], [0.25, 0.75])
        low = ac_rich & (tgfb2 <= q1) & np.isfinite(neighbor_receiver)
        high = (
            ac_rich
            & (tgfb2 >= q3)
            & (tgfb2 > q1)
            & np.isfinite(neighbor_receiver)
        )
    if low.sum() < MIN_GROUP_SPOTS or high.sum() < MIN_GROUP_SPOTS:
        return {
            "neighbor_delta": np.nan,
            "same_spot_delta": np.nan,
            "n_high": int(high.sum()),
            "n_low": int(low.sum()),
        }
    return {
        "neighbor_delta": float(
            np.median(neighbor_receiver[high]) - np.median(neighbor_receiver[low])
        ),
        "same_spot_delta": float(
            np.median(receiver[high]) - np.median(receiver[low])
        ),
        "n_high": int(high.sum()),
        "n_low": int(low.sum()),
    }


def process_section(path: Path) -> tuple[list[dict], dict, pd.DataFrame]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        adata = ad.read_h5ad(path, backed="r")
    names = np.asarray(adata.var_names, dtype=str)
    feature_types = adata.var["feature_types"].astype(str).to_numpy()
    sample_name = str(adata.obs["sample_name"].iloc[0])
    patient = sample_name.split("-")[0]

    gene_indices = feature_indices(
        names, feature_types, TARGET_GENES, "Gene Expression"
    )
    all_gene_indices = np.flatnonzero(feature_types == "Gene Expression")
    state_indices = np.flatnonzero(feature_types == "Cell state abundances")
    state_names = names[state_indices]
    masks = state_masks(state_names)
    if masks["ac"].sum() == 0 or masks["malignant"].sum() == 0:
        raise ValueError(f"{sample_name}: required malignant AC states are absent")
    if masks["myeloid"].sum() == 0:
        raise ValueError(f"{sample_name}: myeloid states are absent")

    selected_gene_counts = dense(adata[:, gene_indices].X).astype(float)
    gene_totals = np.asarray(adata[:, all_gene_indices].X.sum(axis=1)).ravel()
    normalized = np.log1p(
        selected_gene_counts / np.maximum(gene_totals[:, None], 1.0) * 10_000
    )
    gene_frame = pd.DataFrame(normalized, columns=TARGET_GENES)

    states = dense(adata[:, state_indices].X).astype(float)
    total_cell = states.sum(axis=1)
    malignant = states[:, masks["malignant"]].sum(axis=1)
    ac = states[:, masks["ac"]].sum(axis=1)
    ac_broad = states[:, masks["ac_broad"]].sum(axis=1)
    myeloid = states[:, masks["myeloid"]].sum(axis=1)
    normal_astro = states[:, masks["normal_astro"]].sum(axis=1)
    malignant_fraction = malignant / np.maximum(total_cell, 1e-12)
    ac_fraction = ac / np.maximum(malignant, 1e-12)
    ac_broad_fraction = ac_broad / np.maximum(malignant, 1e-12)
    myeloid_fraction = myeloid / np.maximum(total_cell, 1e-12)
    normal_astro_fraction = normal_astro / np.maximum(total_cell, 1e-12)

    receptor_expression = gene_frame[RECEPTORS].mean(axis=1).to_numpy()
    receiver = myeloid_fraction * receptor_expression
    coordinates = np.asarray(adata.obsm["spatial"], dtype=float)
    neighbor_receiver, spacing = neighbor_mean(receiver, coordinates)

    in_tissue = adata.obs["in_tissue"].astype(float).to_numpy() > 0
    valid = (
        in_tissue
        & (gene_totals > 0)
        & (malignant > 0)
        & (malignant_fraction >= MALIGNANT_FRACTION_THRESHOLD)
    )

    controls = np.column_stack(
        [
            normal_astro_fraction,
            malignant_fraction,
            np.log1p(gene_totals),
        ]
    )
    ac_residual = rank_residual(ac_fraction, controls)
    ac_broad_residual = rank_residual(ac_broad_fraction, controls)
    ac_marker_score = gene_frame[AC_MARKERS].mean(axis=1).to_numpy()

    rows = []
    for ligand in LIGANDS:
        expression = gene_frame[ligand].to_numpy()
        expression_residual = rank_residual(expression, controls)
        top_bottom = group_delta(expression, ac_fraction, valid)
        broad_top_bottom = group_delta(expression, ac_broad_fraction, valid)
        rows.append(
            {
                "dataset": "GBM-Space",
                "patient": patient,
                "sample": sample_name,
                "ligand": ligand,
                "n_spots": int(len(expression)),
                "n_eligible_spots": int(valid.sum()),
                "detection_fraction": float(np.mean(expression[valid] > 0)),
                "rho_ac_fraction": safe_spearman(
                    expression[valid], ac_fraction[valid]
                ),
                "rho_ac_fraction_adjusted": safe_spearman(
                    expression_residual[valid], ac_residual[valid]
                ),
                "rho_ac_broad_fraction_adjusted": safe_spearman(
                    expression_residual[valid], ac_broad_residual[valid]
                ),
                "rho_ac_marker_score": safe_spearman(
                    expression[valid], ac_marker_score[valid]
                ),
                "mean_top_minus_bottom_ac_delta": top_bottom["delta"],
                "mean_top_minus_bottom_ac_broad_delta": broad_top_bottom["delta"],
                "n_top_ac": top_bottom["n_high"],
                "n_bottom_ac": top_bottom["n_low"],
            }
        )

    proximity = tgfb2_neighbor_delta(
        gene_frame["TGFB2"].to_numpy(),
        ac_fraction,
        neighbor_receiver,
        receiver,
        valid,
    )
    section_row = {
        "dataset": "GBM-Space",
        "patient": patient,
        "sample": sample_name,
        "n_spots": int(adata.n_obs),
        "n_eligible_spots": int(valid.sum()),
        "n_cell_states": int(len(state_names)),
        "n_ac_states": int(masks["ac"].sum()),
        "n_malignant_states": int(masks["malignant"].sum()),
        "n_myeloid_states": int(masks["myeloid"].sum()),
        "median_malignant_fraction": float(np.median(malignant_fraction[valid])),
        "median_ac_fraction": float(np.median(ac_fraction[valid])),
        "median_myeloid_fraction": float(np.median(myeloid_fraction[valid])),
        "neighbor_spacing": spacing,
        "tgfb2_neighbor_receiver_delta": proximity["neighbor_delta"],
        "tgfb2_same_spot_receiver_delta": proximity["same_spot_delta"],
        "n_tgfb2_high_ac_rich": proximity["n_high"],
        "n_tgfb2_low_ac_rich": proximity["n_low"],
    }

    plot_frame = pd.DataFrame(
        {
            "patient": patient,
            "sample": sample_name,
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "valid": valid,
            "tgfb2": gene_frame["TGFB2"].to_numpy(),
            "ac_fraction": ac_fraction,
            "normal_astro_fraction": normal_astro_fraction,
            "receiver": receiver,
            "neighbor_receiver": neighbor_receiver,
        }
    )
    adata.file.close()
    return rows, section_row, plot_frame


def summarize_ligands(section_ligands: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = [
        "rho_ac_fraction",
        "rho_ac_fraction_adjusted",
        "rho_ac_broad_fraction_adjusted",
        "rho_ac_marker_score",
        "mean_top_minus_bottom_ac_delta",
        "mean_top_minus_bottom_ac_broad_delta",
    ]
    patient = (
        section_ligands.groupby(["patient", "ligand"], as_index=False)[metric_columns]
        .median()
        .sort_values(["ligand", "patient"])
    )
    summary_rows = []
    for ligand, group in patient.groupby("ligand"):
        row = {"ligand": ligand, "n_patients": int(group["patient"].nunique())}
        for metric in metric_columns:
            values = group[metric].dropna().to_numpy()
            row[f"{metric}_median"] = (
                float(np.median(values)) if len(values) else np.nan
            )
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{metric}_n_positive"] = int(np.sum(values > 0))
            row[f"{metric}_signflip_p"] = exact_signflip_p(values)
            row[f"{metric}_wilcoxon_p"] = safe_wilcoxon(values)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    for metric in metric_columns:
        summary[f"{metric}_signflip_q"] = bh_adjust(
            summary[f"{metric}_signflip_p"]
        )
    return patient, summary


def summarize_proximity(sections: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    metrics = [
        "tgfb2_neighbor_receiver_delta",
        "tgfb2_same_spot_receiver_delta",
    ]
    patient = sections.groupby("patient", as_index=False)[metrics].median()
    summary = {"n_patients": int(patient["patient"].nunique())}
    for metric in metrics:
        values = patient[metric].dropna().to_numpy()
        summary[metric] = {
            "n": int(len(values)),
            "median": float(np.median(values)) if len(values) else np.nan,
            "mean": float(np.mean(values)) if len(values) else np.nan,
            "n_positive": int(np.sum(values > 0)),
            "signflip_p": exact_signflip_p(values),
            "wilcoxon_p": safe_wilcoxon(values),
        }
    return patient, summary


def choose_representative(
    plot_frames: dict[str, pd.DataFrame], section_ligands: pd.DataFrame
) -> str:
    tgfb2 = section_ligands.loc[
        section_ligands["ligand"] == "TGFB2", ["sample", "rho_ac_fraction_adjusted"]
    ].dropna()
    cohort_median = tgfb2["rho_ac_fraction_adjusted"].median()
    tgfb2["distance"] = (
        tgfb2["rho_ac_fraction_adjusted"] - cohort_median
    ).abs()
    for sample in tgfb2.sort_values("distance")["sample"]:
        if sample in plot_frames:
            return str(sample)
    return sorted(plot_frames)[0]


def add_patient_strip(
    ax: plt.Axes,
    frame: pd.DataFrame,
    y_column: str,
    title: str,
    ylabel: str,
) -> None:
    values = frame[y_column].dropna().to_numpy()
    jitter = RNG.normal(0, 0.035, len(values))
    ax.scatter(jitter, values, s=34, color="#167D8D", alpha=0.85, zorder=3)
    if len(values):
        median = float(np.median(values))
        ax.plot([-0.18, 0.18], [median, median], color="#C0392B", lw=3, zorder=4)
    ax.axhline(0, color="#777777", lw=1, ls="--")
    ax.set_xlim(-0.35, 0.35)
    ax.set_xticks([])
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right", "bottom"]].set_visible(False)


def make_figure(
    patient_ligands: pd.DataFrame,
    patient_proximity: pd.DataFrame,
    plot_frames: dict[str, pd.DataFrame],
    section_ligands: pd.DataFrame,
) -> None:
    tgfb2_patient = patient_ligands[patient_ligands["ligand"] == "TGFB2"].copy()
    ligand_order = LIGANDS
    fig = plt.figure(figsize=(13.8, 8.2))
    grid = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], wspace=0.75, hspace=0.50)

    ax_a = fig.add_subplot(grid[0, 0])
    raw = tgfb2_patient["rho_ac_fraction"].to_numpy()
    adjusted = tgfb2_patient["rho_ac_fraction_adjusted"].to_numpy()
    for raw_value, adjusted_value in zip(raw, adjusted):
        if np.isfinite(raw_value) and np.isfinite(adjusted_value):
            ax_a.plot([0, 1], [raw_value, adjusted_value], color="#B7C3C8", lw=0.8)
    ax_a.scatter(
        RNG.normal(0, 0.035, len(raw)),
        raw,
        s=30,
        color="#167D8D",
        alpha=0.85,
        zorder=3,
    )
    ax_a.scatter(
        RNG.normal(1, 0.035, len(adjusted)),
        adjusted,
        s=30,
        color="#6C7A89",
        alpha=0.85,
        zorder=3,
    )
    ax_a.hlines(np.nanmedian(raw), -0.18, 0.18, color="#C0392B", lw=2.5)
    ax_a.hlines(np.nanmedian(adjusted), 0.82, 1.18, color="#C0392B", lw=2.5)
    ax_a.axhline(0, color="#777777", lw=1, ls="--")
    ax_a.set_xlim(-0.35, 1.35)
    ax_a.set_xticks([0, 1], ["Raw", "Adjusted"])
    ax_a.set_ylabel("Patient median within-section rho")
    ax_a.set_title("a  TGFB2-AC association", loc="left", fontsize=10, fontweight="bold")
    ax_a.spines[["top", "right"]].set_visible(False)

    ax_b = fig.add_subplot(grid[0, 1])
    plot_data = patient_ligands.pivot(
        index="patient", columns="ligand", values="rho_ac_fraction_adjusted"
    ).reindex(columns=ligand_order)
    for x, ligand in enumerate(ligand_order):
        values = plot_data[ligand].dropna().to_numpy()
        ax_b.scatter(
            np.full(len(values), x) + RNG.normal(0, 0.06, len(values)),
            values,
            s=18,
            color="#6C7A89",
            alpha=0.65,
        )
        if len(values):
            ax_b.plot(
                [x - 0.22, x + 0.22],
                [np.median(values), np.median(values)],
                color="#C0392B" if ligand == "TGFB2" else "#222222",
                lw=2.5,
            )
    ax_b.axhline(0, color="#777777", lw=1, ls="--")
    ax_b.set_xticks(range(len(ligand_order)), ligand_order, rotation=45, ha="right")
    ax_b.set_ylabel("Patient median adjusted rho")
    ax_b.set_title("b  Fixed ligand panel", loc="left", fontsize=10, fontweight="bold")
    ax_b.spines[["top", "right"]].set_visible(False)

    ax_c = fig.add_subplot(grid[0, 2])
    add_patient_strip(
        ax_c,
        tgfb2_patient,
        "mean_top_minus_bottom_ac_delta",
        "c  AC-rich minus AC-poor",
        "TGFB2 log-normalized delta",
    )

    ax_d = fig.add_subplot(grid[0, 3])
    add_patient_strip(
        ax_d,
        patient_proximity,
        "tgfb2_neighbor_receiver_delta",
        "d  Myeloid/TGFBR proximity",
        "High-minus-low TGFB2 delta",
    )

    sample = choose_representative(plot_frames, section_ligands)
    representative = plot_frames[sample]
    representative = representative[representative["valid"]].copy()
    fields = [
        ("ac_fraction", "Malignant AC fraction", "viridis"),
        ("tgfb2", "TGFB2 expression", "magma"),
        ("receiver", "Myeloid x TGFBR score", "cividis"),
    ]
    for i, (column, label, cmap) in enumerate(fields):
        ax = fig.add_subplot(grid[1, i])
        scatter = ax.scatter(
            representative["x"],
            -representative["y"],
            c=representative[column],
            s=4,
            cmap=cmap,
            linewidths=0,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{chr(ord('e') + i)}  {label}", loc="left", fontsize=10, fontweight="bold"
        )
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)

    ax_note = fig.add_subplot(grid[1, 3])
    ax_note.axis("off")
    ax_note.text(
        0,
        1,
        "Representative section\n"
        f"{sample}\n\n"
        "Section selected as the one closest\n"
        "to the cohort-median adjusted rho.\n\n"
        "Inference uses 12 patient medians;\n"
        "spots and sections are not treated\n"
        "as independent patients.",
        va="top",
        fontsize=10,
        linespacing=1.45,
    )

    fig.suptitle(
        "GBM-Space patient-level spatial validation of AC-like TGFB2",
        x=0.02,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_DIR / "Figure5_GBMSpace_spatial_validation.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        FIGURE_DIR / "Figure5_GBMSpace_spatial_validation.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    files = sorted(DATA_DIR.glob("*.h5ad"))
    if len(files) != 97:
        raise FileNotFoundError(f"Expected 97 GBM-Space h5ad files, found {len(files)}")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    ligand_rows: list[dict] = []
    section_rows: list[dict] = []
    plot_frames: dict[str, pd.DataFrame] = {}
    for i, path in enumerate(files, start=1):
        print(f"[{i:02d}/{len(files)}] {path.name}", flush=True)
        rows, section, plot_frame = process_section(path)
        ligand_rows.extend(rows)
        section_rows.append(section)
        plot_frames[section["sample"]] = plot_frame

    section_ligands = pd.DataFrame(ligand_rows)
    sections = pd.DataFrame(section_rows)
    patient_ligands, ligand_summary = summarize_ligands(section_ligands)
    patient_proximity, proximity_summary = summarize_proximity(sections)

    section_ligands.to_csv(RESULT_DIR / "section_ligand_metrics.csv", index=False)
    sections.to_csv(RESULT_DIR / "section_spatial_metrics.csv", index=False)
    patient_ligands.to_csv(RESULT_DIR / "patient_ligand_metrics.csv", index=False)
    patient_proximity.to_csv(RESULT_DIR / "patient_spatial_metrics.csv", index=False)
    ligand_summary.to_csv(RESULT_DIR / "ligand_summary.csv", index=False)
    representative_sample = choose_representative(plot_frames, section_ligands)
    plot_frames[representative_sample].to_csv(
        RESULT_DIR / "representative_plot.csv", index=False
    )

    tgfb2 = ligand_summary.loc[ligand_summary["ligand"] == "TGFB2"].iloc[0]
    summary = {
        "dataset": "GBM-Space",
        "n_sections": int(sections["sample"].nunique()),
        "n_patients": int(sections["patient"].nunique()),
        "n_spots": int(sections["n_spots"].sum()),
        "n_eligible_spots": int(sections["n_eligible_spots"].sum()),
        "malignant_fraction_threshold": MALIGNANT_FRACTION_THRESHOLD,
        "tgfb2": {
            key: (None if pd.isna(value) else float(value))
            for key, value in tgfb2.items()
            if key != "ligand"
        },
        "proximity": proximity_summary,
    }
    (RESULT_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    make_figure(patient_ligands, patient_proximity, plot_frames, section_ligands)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
