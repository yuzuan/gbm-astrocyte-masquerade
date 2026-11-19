#!/usr/bin/env python3
# Purpose: Estimate within-section associations between malignant AC-like states and six ligands, then test by patient.
# Input: One H5AD per section under --input-dir, with X, var.feature_types, obs.sample_name/in_tissue and obsm.spatial.
# Use backed AnnData reads and the original NumPy/SciPy calculations. The public section_spatial_metrics.csv
# supplies the complete section count.
# Output: external_validation/06_gbmspace_spatial_analysis/ contains 01_section_ligand_metrics.csv,
# 02_patient_ligand_metrics.csv, 03_section_spatial_metrics.csv, 04_patient_spatial_metrics.csv,
# 05_ligand_summary.csv, 06_representative_section_spots.csv, 07_analysis_summary.json,
# 08_figure5_gbmspace_spatial_signaling.pdf and 08_figure5_gbmspace_spatial_signaling.png.
# Usage: python 06_gbmspace_spatial_analysis.py --input-dir H5AD_DIRECTORY [--output-dir OUTPUT_DIRECTORY]

import argparse
from contextlib import closing
from importlib import import_module
from itertools import product
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

spatial = import_module("00a_spatial_analysis")
files = import_module("00a_project_paths")


def main():
    """Read sections and write patient, section and plot data. Stop on missing input or invalid output. No return."""
    external_dir = Path(__file__).resolve().parents[1] / "external_validation"
    parser = argparse.ArgumentParser(description="Patient-level spatial validation in GBM-Space")
    parser.add_argument("--input-dir", type=Path, default=external_dir / "raw/gbmspace/extracted/anndata")
    parser.add_argument("--output-dir", type=Path, default=external_dir / Path(__file__).stem)
    args = parser.parse_args()

    # Read the expected count from the frozen table. Section identifiers use different codes in the two sources.
    baseline_table = pd.read_csv(external_dir / "results/gbmspace/section_spatial_metrics.csv", usecols=["sample"])
    paths = sorted(args.input_dir.glob("*.h5ad"))
    if not paths or len(paths) != baseline_table["sample"].nunique():
        raise ValueError(f"Found {len(paths)} H5AD files for {baseline_table['sample'].nunique()} published sections")

    # Read just the target genes and state matrix from each file. All genes contribute to each spot's library size.
    target_genes = spatial.ligands + spatial.receptors + spatial.ac_markers
    ligand_rows = []
    section_rows = []
    plot_tables = {}
    for path in paths:
        section = ad.read_h5ad(path, backed="r")
        with closing(section.file):
            sample_names = section.obs["sample_name"].drop_duplicates()
            if len(sample_names) != 1 or sample_names.isna().any():
                raise ValueError(f"{path.name} must contain exactly one section name")
            sample = str(sample_names.iloc[0])
            patient = sample.split("-")[0]
            names = np.asarray(section.var_names, dtype=str)
            types = section.var["feature_types"].astype(str).to_numpy()

            # AnnData holds genes and cell abundances. Match feature type and name to obtain exactly one column per
            # target.
            target_mask = (types == "Gene Expression") & np.isin(names, target_genes)
            feature_ids = pd.Series(np.flatnonzero(target_mask), index=names[target_mask])
            if feature_ids.index.has_duplicates or set(feature_ids.index) != set(target_genes):
                raise ValueError(f"{sample} has a missing or duplicate target gene expression feature")
            indices = feature_ids.loc[target_genes].to_numpy()
            selected = section[:, indices].X
            counts = selected.toarray() if sparse.issparse(selected) else np.asarray(selected)
            totals = np.asarray(section[:, types == "Gene Expression"].X.sum(axis=1)).ravel()

            # Read coordinates and the tissue mask in the same obs order before closing the file.
            state_indices = np.flatnonzero(types == "Cell state abundances")
            selected = section[:, state_indices].X
            states = selected.toarray() if sparse.issparse(selected) else np.asarray(selected)
            states = states.astype(float)
            coordinates = np.asarray(section.obsm["spatial"], dtype=float)
            in_tissue = section.obs["in_tissue"].astype(float).to_numpy() > 0

        # The broad AC definition includes proliferating AC-OPC states. Divide AC abundance by total malignant
        # abundance.
        state_names = pd.Index(names[state_indices])
        ac_prefixes = ("AC progenitor-like", "AC-gliosis-like")
        broad_prefixes = ac_prefixes + ("Proliferative AC-OPC-like",)
        malignant_prefixes = broad_prefixes + (
            "Hypoxic ", "NPC-neuronal-like", "OPC-NPC-like", "OPC-like ", "OPC-neuronal-like",
            "Proliferative NPC-OPC-like", "Proliferative nIPC-like", "Undefined ",
        )
        ac_mask = state_names.str.startswith(ac_prefixes)
        malignant_mask = state_names.str.startswith(malignant_prefixes)
        myeloid_mask = state_names.str.contains("TAMs", regex=False)
        myeloid_mask |= state_names.isin(["Monocytes", "Dendritic cells"])
        if not ac_mask.any() or not malignant_mask.any() or not myeloid_mask.any():
            raise ValueError(f"{sample} is missing AC, malignant or myeloid states")

        # Divide myeloid and normal astrocyte abundances by total abundance, retaining the original denominator floor.
        astro_mask = state_names.str.lower().str.contains("astrocytes", regex=False)
        astro_mask |= state_names.isin(["Gliosis-like", "Neural-support astrocytes"])
        total_abundance = np.maximum(states.sum(axis=1), 1e-12)
        malignant = states[:, malignant_mask].sum(axis=1)
        malignant_fraction = malignant / total_abundance
        ac_fraction = states[:, ac_mask].sum(axis=1) / np.maximum(malignant, 1e-12)
        broad_abundance = states[:, state_names.str.startswith(broad_prefixes)].sum(axis=1)
        broad_fraction = broad_abundance / np.maximum(malignant, 1e-12)
        myeloid_fraction = states[:, myeloid_mask].sum(axis=1) / total_abundance
        astro_fraction = states[:, astro_mask].sum(axis=1) / total_abundance

        # Normalize each spot's total gene counts to 10,000 and apply log1p. Weight mean receptor expression by myeloid
        # fraction.
        normalized = np.log1p(counts.astype(float) / np.maximum(totals[:, None], 1.0) * 10_000)
        expression_table = pd.DataFrame(normalized, columns=target_genes)
        marker_score = expression_table[spatial.ac_markers].mean(axis=1).to_numpy()
        receiver = myeloid_fraction * expression_table[spatial.receptors].mean(axis=1).to_numpy()
        tree = cKDTree(coordinates)
        distances, _ = tree.query(coordinates, k=2)
        nearest = distances[:, 1]
        spacing = float(np.median(nearest[np.isfinite(nearest) & (nearest > 0)]))

        # Use a radius of 1.25 times the median nearest-neighbor distance. Exclude the spot itself and retain NaN for no
        # neighbor.
        neighbor_receiver = np.full(len(receiver), np.nan)
        for i, neighbors in enumerate(tree.query_ball_point(coordinates, r=1.25 * spacing)):
            neighbors = np.asarray(neighbors, dtype=int)
            neighbors = neighbors[(neighbors != i) & np.isfinite(receiver[neighbors])]
            if len(neighbors):
                neighbor_receiver[i] = np.mean(receiver[neighbors])

        # Eligible spots are in tissue, have expression and malignant abundance, and meet the 20% malignant fraction
        # threshold.
        # Fit the rank regression on the whole section first.
        malignant_threshold = 0.20
        valid = in_tissue & (totals > 0) & (malignant > 0) & (malignant_fraction >= malignant_threshold)
        controls = np.column_stack([astro_fraction, malignant_fraction, np.log1p(totals)])
        ac_residual = spatial.rank_residual(ac_fraction, controls, standardize_controls=True)
        broad_residual = spatial.rank_residual(broad_fraction, controls, standardize_controls=True)

        # For each ligand, use narrow or broad AC quartiles within eligible spots. Retain boundary ties and require 20
        # per group.
        contrasts = {}
        groupings = {"ac": ac_fraction, "broad": broad_fraction}
        for ligand, grouping_name in product(spatial.ligands, groupings):
            values = expression_table[ligand].to_numpy()
            grouping = groupings[grouping_name]
            eligible = valid & np.isfinite(values) & np.isfinite(grouping)
            contrasts[ligand, grouping_name] = (np.nan, 0, 0)
            if eligible.sum() < 40:
                continue
            lower, upper = np.quantile(grouping[eligible], [0.25, 0.75])
            low = eligible & (grouping <= lower)
            high = eligible & (grouping >= upper)
            delta = np.mean(values[high]) - np.mean(values[low]) if min(high.sum(), low.sum()) >= 20 else np.nan
            contrasts[ligand, grouping_name] = (delta, int(high.sum()), int(low.sum()))

        # Each row describes one section and ligand. Correlations use eligible spots and are later summarized by
        # patient.
        for ligand in spatial.ligands:
            values = expression_table[ligand].to_numpy()
            residual = spatial.rank_residual(values, controls, standardize_controls=True)
            delta, n_high, n_low = contrasts[ligand, "ac"]
            ligand_rows.append({
                "dataset": "GBM-Space", "patient": patient, "sample": sample, "ligand": ligand,
                "n_spots": len(values), "n_eligible_spots": int(valid.sum()),
                "detection_fraction": float(np.mean(values[valid] > 0)) if valid.any() else np.nan,
                "rho_ac_fraction": spatial.spot_spearman(values[valid], ac_fraction[valid]),
                "rho_ac_fraction_adjusted": spatial.spot_spearman(residual[valid], ac_residual[valid]),
                "rho_ac_broad_fraction_adjusted": spatial.spot_spearman(residual[valid], broad_residual[valid]),
                "rho_ac_marker_score": spatial.spot_spearman(values[valid], marker_score[valid]),
                "mean_top_minus_bottom_ac_delta": delta,
                "mean_top_minus_bottom_ac_broad_delta": contrasts[ligand, "broad"][0],
                "n_top_ac": n_high, "n_bottom_ac": n_low,
            })

        # Require at least 80 candidate spots before taking the upper half by AC fraction. Prefer the upper half of
        # positive TGFB2 expression versus zero expression.
        tgfb2 = expression_table["TGFB2"].to_numpy()
        ac_rich = valid & np.isfinite(tgfb2) & np.isfinite(ac_fraction)
        high = np.zeros(len(tgfb2), dtype=bool)
        low = high.copy()
        if ac_rich.sum() >= 80:
            ac_rich &= ac_fraction >= np.median(ac_fraction[ac_rich])
            positive = ac_rich & (tgfb2 > 0)
            zero = ac_rich & (tgfb2 == 0)
            if min(positive.sum(), zero.sum()) >= 20:
                high = positive & (tgfb2 >= np.median(tgfb2[positive]))
                low = zero
            else:
                lower, upper = np.quantile(tgfb2[ac_rich], [0.25, 0.75])
                high = ac_rich & (tgfb2 >= upper) & (tgfb2 > lower)
                low = ac_rich & (tgfb2 <= lower)
            high &= np.isfinite(neighbor_receiver)
            low &= np.isfinite(neighbor_receiver)

        # Report a median difference when both groups have 20 spots. Use the same groups for same-spot and neighbor
        # scores.
        enough = min(high.sum(), low.sum()) >= 20
        neighbor_delta = np.median(neighbor_receiver[high]) - np.median(neighbor_receiver[low]) if enough else np.nan
        same_spot_delta = np.median(receiver[high]) - np.median(receiver[low]) if enough else np.nan
        section_rows.append({
            "dataset": "GBM-Space", "patient": patient, "sample": sample, "n_spots": len(valid),
            "n_eligible_spots": int(valid.sum()), "n_cell_states": len(state_names), "n_ac_states": int(ac_mask.sum()),
            "n_malignant_states": int(malignant_mask.sum()), "n_myeloid_states": int(myeloid_mask.sum()),
            "median_malignant_fraction": np.median(malignant_fraction[valid]) if valid.any() else np.nan,
            "median_ac_fraction": np.median(ac_fraction[valid]) if valid.any() else np.nan,
            "median_myeloid_fraction": np.median(myeloid_fraction[valid]) if valid.any() else np.nan,
            "neighbor_spacing": spacing,
            "tgfb2_neighbor_receiver_delta": neighbor_delta, "tgfb2_same_spot_receiver_delta": same_spot_delta,
            "n_tgfb2_high_ac_rich": int(high.sum()), "n_tgfb2_low_ac_rich": int(low.sum()),
        })
        plot_tables[sample] = pd.DataFrame({
            "patient": patient, "sample": sample, "x": coordinates[:, 0], "y": coordinates[:, 1], "valid": valid,
            "tgfb2": tgfb2, "ac_fraction": ac_fraction, "normal_astro_fraction": astro_fraction,
            "receiver": receiver, "neighbor_receiver": neighbor_receiver,
        })

    # Take the section median per patient and ligand, then adjust across the six ligands separately for each metric.
    section_ligand_table_total = pd.DataFrame(ligand_rows)
    section_table_total = pd.DataFrame(section_rows)
    metrics = [
        "rho_ac_fraction", "rho_ac_fraction_adjusted", "rho_ac_broad_fraction_adjusted", "rho_ac_marker_score",
        "mean_top_minus_bottom_ac_delta", "mean_top_minus_bottom_ac_broad_delta",
    ]
    grouped = section_ligand_table_total.groupby(["patient", "ligand"], as_index=False)
    patient_ligand_table = grouped[metrics].median().sort_values(["ligand", "patient"])
    rng = np.random.default_rng(20260716)
    summary_rows = []
    for ligand, group_table in patient_ligand_table.groupby("ligand"):
        record = {"ligand": ligand, "n_patients": group_table["patient"].nunique()}
        for metric in metrics:
            measures = spatial.patient_metrics(group_table[metric], rng)
            for name in ["median", "mean", "n_positive", "signflip_p", "wilcoxon_p"]:
                record[f"{metric}_{name}"] = measures[name]
        summary_rows.append(record)
    summary_table = pd.DataFrame(summary_rows)
    for metric in metrics:
        summary_table[f"{metric}_signflip_q"] = spatial.bh_adjust(summary_table[f"{metric}_signflip_p"])

    # Take patient medians of neighboring and same-spot scores. Keep the original tie order for section selection.
    # If no valid correlation is available, use the first section name in sorted order.
    proximity_columns = ["tgfb2_neighbor_receiver_delta", "tgfb2_same_spot_receiver_delta"]
    patient_table = section_table_total.groupby("patient", as_index=False)[proximity_columns].median()
    proximity = {"n_patients": patient_table["patient"].nunique()}
    for metric in proximity_columns:
        proximity[metric] = spatial.patient_metrics(patient_table[metric], rng)
    tgfb2_table = section_ligand_table_total.loc[section_ligand_table_total["ligand"] == "TGFB2"]
    adjusted = tgfb2_table.set_index("sample")["rho_ac_fraction_adjusted"].dropna()
    distances = (adjusted - adjusted.median()).abs()
    sample = distances.sort_values().index[0] if len(distances) else sorted(plot_tables)[0]
    representative_table = plot_tables[sample]

    # Reread each CSV. Derive the total and eligible spot counts and the patient count from the section table.
    spatial.write_spatial_tables(
        args.output_dir, section_ligand_table_total, patient_ligand_table, section_table_total, patient_table,
        summary_table, representative_table
    )
    tgfb2_result = summary_table.set_index("ligand").loc["TGFB2"]
    summary = {
        "dataset": "GBM-Space", "n_sections": len(section_table_total), "n_patients": len(patient_table),
        "n_spots": int(section_table_total["n_spots"].sum()),
        "n_eligible_spots": int(section_table_total["n_eligible_spots"].sum()),
        "malignant_fraction_threshold": malignant_threshold,
        "tgfb2": {key: float(value) if pd.notna(value) else None for key, value in tgfb2_result.items()},
        "proximity": proximity,
    }
    files.write_json(summary, args.output_dir / "07_analysis_summary.json")

    # The first two panels consume the original random stream in order. Jitter affects horizontal display positions
    # only.
    figure = plt.figure(figsize=(13.8, 8.2))
    grid = figure.add_gridspec(2, 4, height_ratios=[1, 1.15], wspace=0.75, hspace=0.50)
    axis = figure.add_subplot(grid[0, 0])
    tgfb2_patient_table = patient_ligand_table.loc[patient_ligand_table["ligand"] == "TGFB2"]
    unadjusted = tgfb2_patient_table["rho_ac_fraction"].to_numpy()
    adjusted = tgfb2_patient_table["rho_ac_fraction_adjusted"].to_numpy()
    for first, second in zip(unadjusted, adjusted):
        if np.isfinite(first) and np.isfinite(second):
            axis.plot([0, 1], [first, second], color="#B7C3C8", lw=0.8)
    for position, values, color in [(0, unadjusted, "#167D8D"), (1, adjusted, "#6C7A89")]:
        axis.scatter(rng.normal(position, 0.035, len(values)), values, s=30, color=color, alpha=0.85, zorder=3)
        axis.hlines(np.nanmedian(values), position - 0.18, position + 0.18, color="#C0392B", lw=2.5)
    axis.set(xlim=(-0.35, 1.35), ylabel="Patient median within-section rho")
    axis.set_xticks([0, 1], ["Raw", "Adjusted"])
    axis.set_title("a  TGFB2-AC association", loc="left", fontsize=10, fontweight="bold")

    # Show each ligand's patient distribution and its own median in the second panel.
    panel_axis = figure.add_subplot(grid[0, 1])
    pivot_table = patient_ligand_table.pivot(index="patient", columns="ligand", values="rho_ac_fraction_adjusted")
    for i, ligand in enumerate(spatial.ligands):
        values = pivot_table[ligand].dropna().to_numpy()
        panel_axis.scatter(i + rng.normal(0, 0.06, len(values)), values, s=18, color="#6C7A89", alpha=0.65)
        if len(values):
            color = "#C0392B" if ligand == "TGFB2" else "#222222"
            panel_axis.hlines(np.median(values), i - 0.22, i + 0.22, color=color, lw=2.5)
    panel_axis.set_xticks(range(len(spatial.ligands)), spatial.ligands, rotation=45, ha="right")
    panel_axis.set_ylabel("Patient median adjusted rho")
    panel_axis.set_title("b  Fixed ligand panel", loc="left", fontsize=10, fontweight="bold")

    # Show high-minus-low AC contrasts and differences in neighboring receptor scores for high versus low TGFB2.
    strip_panels = [
        (tgfb2_patient_table["mean_top_minus_bottom_ac_delta"], "c  AC-rich minus AC-poor", "TGFB2 log-normalized Δ"),
        (patient_table["tgfb2_neighbor_receiver_delta"], "d  Myeloid/TGFBR proximity", "High-minus-low TGFB2 delta"),
    ]
    for i, (series, title, label) in enumerate(strip_panels, start=2):
        strip = figure.add_subplot(grid[0, i])
        values = series.dropna().to_numpy()
        strip.scatter(rng.normal(0, 0.035, len(values)), values, s=34, color="#167D8D", alpha=0.85, zorder=3)
        if len(values):
            strip.hlines(np.median(values), -0.18, 0.18, color="#C0392B", lw=3, zorder=4)
        strip.set(xlim=(-0.35, 0.35), xticks=[], ylabel=label)
        strip.set_title(title, loc="left", fontsize=10, fontweight="bold")
        strip.spines["bottom"].set_visible(False)
    for panel in figure.axes:
        panel.axhline(0, color="#777777", lw=1, ls="--")
        panel.spines[["top", "right"]].set_visible(False)

    # All three maps use the same eligible spots and orientation. Color each metric on its original scale.
    map_table = representative_table.loc[representative_table["valid"]]
    maps = [
        ("ac_fraction", "e  Malignant AC fraction", "viridis"), ("tgfb2", "f  TGFB2 expression", "magma"),
        ("receiver", "g  Myeloid x TGFBR score", "cividis"),
    ]
    for i, (column, title, palette) in enumerate(maps):
        panel = figure.add_subplot(grid[1, i])
        points = panel.scatter(map_table["x"], -map_table["y"], c=map_table[column], s=4, cmap=palette, linewidths=0)
        panel.set(aspect="equal", xticks=[], yticks=[])
        panel.set_title(title, loc="left", fontsize=10, fontweight="bold")
        panel.spines[["top", "right", "bottom", "left"]].set_visible(False)
        figure.colorbar(points, ax=panel, fraction=0.046, pad=0.02)

    # Use the selected section and actual patient count in the caption. Render the 600 dpi PNG from the single PDF.
    note = figure.add_subplot(grid[1, 3])
    note.axis("off")
    selection = "Closest to the median adjusted rho." if len(distances) else "First section by name."
    note_text = f"Representative section\n{sample}\n\n{selection}\n\nPatient medians: {len(patient_table)}."
    note.text(0, 1, note_text, va="top", fontsize=10, linespacing=1.45)
    figure.suptitle("GBM-Space patient-level spatial validation of AC-like TGFB2", x=0.02, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    files.save_figure(figure, args.output_dir / "08_figure5_gbmspace_spatial_signaling.pdf")
    plt.close(figure)
    print(
        f"{len(section_table_total)} sections, {len(patient_table)} patients, "
        f"{summary['n_eligible_spots']} eligible spots"
    )


if __name__ == "__main__":
    main()
