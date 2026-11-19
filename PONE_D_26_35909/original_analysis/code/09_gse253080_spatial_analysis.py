#!/usr/bin/env python3
# Purpose: Compare AC-like scores, six ligands and neighboring myeloid receptor signal in GSE253080 Visium counts.
# Input: --input-dir contains pd.csv (geo_accession, description, subject id:ch1) and each section's compressed
# matrix, features, barcodes and tissue_positions_list files. Use SciPy sparse matrices and the original random stream.
# Output: external_validation/09_gse253080_spatial_analysis/ contains 01_section_ligand_metrics.csv,
# 02_patient_ligand_metrics.csv, 03_section_spatial_metrics.csv, 04_patient_spatial_metrics.csv,
# 05_ligand_summary.csv, 06_representative_section_spots.csv, 07_analysis_summary.json,
# 08_figure_s7_gse253080_spatial_signaling.pdf and 08_figure_s7_gse253080_spatial_signaling.png.
# Usage: python 09_gse253080_spatial_analysis.py --input-dir GSE253080_DIRECTORY [--output-dir OUTPUT_DIRECTORY]

import argparse
import gzip
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import mmread

spatial = import_module("00a_spatial_analysis")
files = import_module("00a_project_paths")


def main():
    """Read sections and summarize by patient. Stop on unmatched samples, coordinates or invalid output. No return."""
    external_dir = Path(__file__).resolve().parents[1] / "external_validation"
    parser = argparse.ArgumentParser(description="Patient-level spatial sensitivity analysis in GSE253080")
    parser.add_argument("--input-dir", type=Path, default=external_dir / "raw/gse253080")
    parser.add_argument("--output-dir", type=Path, default=external_dir / Path(__file__).stem)
    args = parser.parse_args()

    # Each row represents one section. Combine the GSM accession with the first part of the GEO description for the
    # prefix.
    metadata_table = pd.read_csv(args.input_dir / "pd.csv", dtype={"geo_accession": str, "subject id:ch1": str})
    metadata_table["sample"] = metadata_table["description"].str.split("_").str[0]
    metadata_table = metadata_table.rename(columns={"subject id:ch1": "patient"})
    identity_table = metadata_table[["geo_accession", "sample", "patient"]]
    if identity_table.isna().any(axis=None) or metadata_table["sample"].duplicated().any() or metadata_table.empty:
        raise ValueError("pd.csv has missing sample identifiers, duplicate sections or no samples")

    # The three marker sets define separate scores. The six ligands contribute only their expression measurements.
    marker_sets = {
        "ac": spatial.ac_markers, "myeloid": ["AIF1", "CD163", "CSF1R", "LST1", "TYROBP", "FCER1G"],
        "receptor": spatial.receptors,
    }
    score_genes = sorted(set().union(*marker_sets.values()))
    target_genes = sorted(set(score_genes) | set(spatial.ligands))
    ligand_rows = []
    section_rows = []
    plot_tables = {}
    for row in metadata_table.itertuples(index=False):
        prefix = f"{row.geo_accession}_{row.sample}"
        matrix = mmread(args.input_dir / f"{prefix}_matrix.mtx.gz").tocsr()
        feature_table = pd.read_csv(
            args.input_dir / f"{prefix}_features.tsv.gz", sep="\t", header=None,
            names=["gene_id", "gene_symbol", "feature_type"]
        )
        with gzip.open(args.input_dir / f"{prefix}_barcodes.tsv.gz", "rt") as stream:
            barcodes = [line.rstrip("\n") for line in stream]

        # Align the headerless position file by barcode. Apply the tissue mask to the count and position rows together.
        position_table = pd.read_csv(
            args.input_dir / f"{prefix}_tissue_positions_list.csv.gz", header=None,
            names=["barcode", "in_tissue", "array_row", "array_col", "pixel_row", "pixel_col"]
        )
        if matrix.shape == (len(feature_table), len(barcodes)):
            matrix = matrix.transpose().tocsr()
        elif matrix.shape != (len(barcodes), len(feature_table)):
            raise ValueError(f"Matrix dimensions for {row.sample} disagree with the gene and barcode counts")
        position_table = position_table.set_index("barcode", verify_integrity=True).reindex(barcodes)
        in_tissue = position_table["in_tissue"].fillna(0).to_numpy() == 1
        matrix = matrix[in_tissue]
        position_table = position_table.loc[in_tissue]

        # Sum raw counts for duplicate gene symbols, normalize to 10,000 across all features and apply log1p.
        # Use a denominator of one for an empty library, as in the original analysis.
        symbols = feature_table["gene_symbol"].astype(str).str.upper().to_numpy()
        counts = {}
        for gene in target_genes:
            columns = np.flatnonzero(symbols == gene)
            if len(columns):
                counts[gene] = np.asarray(matrix[:, columns].sum(axis=1)).ravel()
        library_sizes = np.maximum(np.asarray(matrix.sum(axis=1)).ravel(), 1)
        expression_table = pd.DataFrame(counts, index=position_table.index)
        expression_table = np.log1p(expression_table.div(library_sizes, axis=0) * 10_000)

        # Standardize each gene across spots. Average a set only if at least two markers vary. Otherwise retain missing
        # scores.
        standardized = {}
        for gene in score_genes:
            if gene in expression_table and expression_table[gene].std(ddof=0) > 0:
                values = expression_table[gene].to_numpy()
                standardized[gene] = (values - np.nanmean(values)) / np.nanstd(values)
        scores = {}
        for name, markers in marker_sets.items():
            available = [standardized[gene] for gene in markers if gene in standardized]
            scores[name] = np.full(len(position_table), np.nan)
            if len(available) >= 2:
                scores[name] = np.nanmean(np.column_stack(available), axis=1)

        # Standardize the myeloid and receptor scores again within section, then average their available components.
        components = []
        for name in ["myeloid", "receptor"]:
            values = scores[name]
            deviation = np.nanstd(values)
            component = np.full(len(values), np.nan)
            if np.isfinite(deviation) and deviation > 0:
                component = (values - np.nanmean(values)) / deviation
            components.append(component)
        receiver = np.nanmean(np.column_stack(components), axis=1)

        # Look up six Visium neighbors by row and column offsets. Exclude the spot itself. Retain NaN when none are
        # valid.
        array_rows = position_table["array_row"].to_numpy()
        array_cols = position_table["array_col"].to_numpy()
        coordinates = pd.MultiIndex.from_arrays([array_rows, array_cols])
        lookup = pd.Series(receiver, index=coordinates)
        neighbors = []
        for row_offset, col_offset in [(0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor_coordinates = pd.MultiIndex.from_arrays([array_rows + row_offset, array_cols + col_offset])
            neighbors.append(lookup.reindex(neighbor_coordinates).to_numpy())
        neighbor_matrix = np.column_stack(neighbors)
        present = np.isfinite(neighbor_matrix).sum(axis=1)
        neighbor_receiver = np.full(len(position_table), np.nan)
        np.divide(np.nansum(neighbor_matrix, axis=1), present, out=neighbor_receiver, where=present > 0)

        # Adjust ranks for log1p library size and myeloid score. This cohort uses unstandardized covariate ranks.
        controls = np.column_stack([np.log1p(library_sizes), scores["myeloid"]])
        ac_score = scores["ac"]
        ac_residual = spatial.rank_residual(ac_score, controls, standardize_controls=False)
        for ligand in spatial.ligands:
            values = expression_table[ligand].to_numpy()
            residual = spatial.rank_residual(values, controls, standardize_controls=False)
            positive = values > 0
            high = positive & (values >= np.median(values[positive])) if positive.any() else positive
            zero = values == 0
            delta = np.nan
            if min(high.sum(), zero.sum()) >= 10:
                delta = np.nanmean(ac_score[high]) - np.nanmean(ac_score[zero])
            ligand_rows.append({
                "dataset": "GSE253080", "patient": row.patient, "sample": row.sample, "ligand": ligand,
                "n_spots": len(values), "detection_fraction": np.mean(positive),
                "rho_ac_score": spatial.spot_spearman(values, ac_score),
                "rho_ac_score_adjusted": spatial.spot_spearman(residual, ac_residual),
                "high_minus_zero_ac_score": delta,
            })

        # Within the upper AC-score quartile, compare neighbor means for high positive versus zero TGFB2.
        # Use the upper half of positive TGFB2 values and require ten spots per group.
        tgfb2 = expression_table["TGFB2"].to_numpy()
        ac_high = ac_score >= np.nanquantile(ac_score, 0.75)
        positive = ac_high & (tgfb2 > 0) & np.isfinite(neighbor_receiver)
        high = np.zeros(len(tgfb2), dtype=bool)
        if positive.sum() >= 10:
            high = positive & (tgfb2 >= np.nanmedian(tgfb2[positive]))
        zero = ac_high & (tgfb2 == 0) & np.isfinite(neighbor_receiver)
        delta = np.nan
        if min(high.sum(), zero.sum()) >= 10:
            delta = np.nanmean(neighbor_receiver[high]) - np.nanmean(neighbor_receiver[zero])
        section_rows.append({
            "dataset": "GSE253080", "patient": row.patient, "sample": row.sample, "n_spots": len(tgfb2),
            "n_tgfb2_high_ac": int(high.sum()), "n_tgfb2_zero_ac": int(zero.sum()),
            "tgfb2_neighbor_receiver_delta": delta,
        })
        plot_tables[row.sample] = pd.DataFrame({
            "sample": row.sample, "pixel_col": position_table["pixel_col"], "pixel_row": position_table["pixel_row"],
            "tgfb2": tgfb2, "ac_score": ac_score, "receiver_score": receiver,
        })

    # Take the section median for each patient. Apply BH across ligands per metric and summarize proximity separately.
    section_ligand_table_total = pd.DataFrame(ligand_rows)
    section_table_total = pd.DataFrame(section_rows)
    metrics = ["rho_ac_score", "rho_ac_score_adjusted", "high_minus_zero_ac_score"]
    patient_ligand_table = section_ligand_table_total.groupby(["patient", "ligand"], as_index=False)[metrics].median()
    patient_table = section_table_total.groupby("patient", as_index=False)[["tgfb2_neighbor_receiver_delta"]].median()
    summary_rows = []
    for ligand in spatial.ligands:
        group_table = patient_ligand_table.loc[patient_ligand_table["ligand"] == ligand]
        record = {"ligand": ligand}
        for metric in metrics:
            measures = spatial.patient_metrics(group_table[metric])
            for name, value in measures.items():
                suffix = "n_patients" if name == "n" else name
                record[f"{metric}_{suffix}"] = value
        summary_rows.append(record)
    summary_table = pd.DataFrame(summary_rows)
    for metric in metrics:
        summary_table[f"{metric}_signflip_q"] = spatial.bh_adjust(summary_table[f"{metric}_signflip_p"])

    # Take the two central finite correlations in sorted order (one middle value for an odd count).
    # Choose the first section in pd.csv whose correlation equals either central value.
    tgfb2_table = section_ligand_table_total.loc[section_ligand_table_total["ligand"] == "TGFB2"]
    correlations = tgfb2_table.set_index("sample")["rho_ac_score_adjusted"]
    finite_correlations = correlations[np.isfinite(correlations)]
    if finite_correlations.empty:
        raise ValueError("No finite adjusted TGFB2 correlation is available to select a representative section")
    ordered = np.sort(finite_correlations.to_numpy())
    n_sections = len(ordered)
    middle_values = [ordered[(n_sections - 1) // 2], ordered[n_sections // 2]]
    sample = finite_correlations.index[finite_correlations.isin(middle_values)][0]
    representative_table = plot_tables[sample]
    spatial.write_spatial_tables(
        args.output_dir, section_ligand_table_total, patient_ligand_table, section_table_total, patient_table,
        summary_table, representative_table
    )

    # Keep the published patient-count field for each metric. Use the original three-patient wording only for that
    # count.
    proximal = spatial.patient_metrics(patient_table["tgfb2_neighbor_receiver_delta"])
    proximity = {}
    for name, value in proximal.items():
        suffix = "n_patients" if name == "n" else name
        proximity[f"tgfb2_neighbor_receiver_delta_{suffix}"] = value if pd.notna(value) else None
    tgfb2_result = summary_table.loc[summary_table["ligand"] == "TGFB2"].iloc[0]
    patient_count = len(patient_table)
    patient_text = "three" if patient_count == 3 else str(patient_count)
    summary = {
        "dataset": "GSE253080", "n_sections": len(section_table_total), "n_patients": patient_count,
        "tgfb2": tgfb2_result.astype(object).where(tgfb2_result.notna(), None).to_dict(), "proximity": proximity,
        "inference_limit": f"{patient_text} patients; directional sensitivity analysis only",
    }
    files.write_json(summary, args.output_dir / "07_analysis_summary.json")

    # Draw patient jitter in the original panel order using one random stream. Jitter affects display only.
    rng = np.random.default_rng(20260716)
    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    axis = axes[0, 0]
    values = patient_ligand_table.loc[patient_ligand_table["ligand"] == "TGFB2", "rho_ac_score_adjusted"].to_numpy()
    axis.scatter(rng.normal(1, 0.035, len(values)), values, color="#267D83", s=55)
    axis.hlines(np.median(values), 0.82, 1.18, color="#B33A3A", linewidth=2.5)
    axis.set(xlim=(0.65, 1.35), ylabel="Adjusted within-section Spearman rho")
    axis.set_xticks([1], [f"{np.isfinite(values).sum()} patient medians"])
    axis.set_title("a  TGFB2 vs AC-like score", loc="left", fontweight="bold")

    # Show all ligands in the fixed panel. For proximity, include only patients with an estimable difference.
    axis = axes[0, 1]
    for i, ligand in enumerate(spatial.ligands):
        values = patient_ligand_table.loc[patient_ligand_table["ligand"] == ligand, "rho_ac_score_adjusted"].dropna()
        axis.scatter(rng.normal(i, 0.045, len(values)), values, color="#6C7A89", alpha=0.75, s=30)
        color = "#B33A3A" if ligand == "TGFB2" else "#222222"
        axis.hlines(values.median(), i - 0.20, i + 0.20, color=color, linewidth=2)
    axis.set_xticks(range(len(spatial.ligands)), spatial.ligands, rotation=45, ha="right")
    axis.set_ylabel("Adjusted within-section Spearman rho")
    axis.set_title("b  Fixed ligand panel", loc="left", fontweight="bold")

    axis = axes[0, 2]
    values = patient_table["tgfb2_neighbor_receiver_delta"].dropna()
    axis.scatter(rng.normal(1, 0.035, len(values)), values, color="#6A4C93", s=55)
    if len(values):
        axis.hlines(values.median(), 0.82, 1.18, color="#B33A3A", linewidth=2.5)
    axis.set(xlim=(0.65, 1.35), ylabel="Neighbor receiver score difference")
    axis.set_xticks([1], ["patient medians"])
    axis.set_title("c  Descriptive proximity", loc="left", fontweight="bold")
    for axis in axes[0]:
        axis.axhline(0, color="#777777", linestyle="--", linewidth=1)
        axis.spines[["top", "right"]].set_visible(False)

    # All three spatial panels use the representative section's pixel coordinates, with pixel rows increasing downward.
    maps = [
        ("ac_score", "d  AC-like score", "viridis"), ("tgfb2", "e  TGFB2 expression", "magma"),
        ("receiver_score", "f  Myeloid/TGFBR score", "cividis"),
    ]
    for axis, (column, title, palette) in zip(axes[1], maps):
        points = axis.scatter(
            representative_table["pixel_col"], -representative_table["pixel_row"], c=representative_table[column],
            s=4, cmap=palette, linewidth=0
        )
        axis.set_aspect("equal")
        axis.axis("off")
        axis.set_title(title, loc="left", fontweight="bold")
        figure.colorbar(points, ax=axis, fraction=0.046, pad=0.02)
    title = f"GSE253080 spatial sensitivity analysis; representative {sample}"
    figure.suptitle(title, x=0.06, ha="left", fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    files.save_figure(figure, args.output_dir / "08_figure_s7_gse253080_spatial_signaling.pdf")
    plt.close(figure)
    print(f"{len(section_table_total)} sections, {patient_count} patients")


if __name__ == "__main__":
    main()
