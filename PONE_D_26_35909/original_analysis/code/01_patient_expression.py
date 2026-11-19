#!/usr/bin/env python3
# Purpose: Compare ligand expression in AC-like and other malignant states within patients and assess regional
# composition and ligand-receptor coavailability.
# Inputs: raw_inputs/GSE211376_raw_counts.tsv.gz and GSE211376_metadata.csv.gz (patient, predicted.high_hierarchy).
# Read scGBM_GSE162631_GSE273274.h5ad, snGBM_GSE276177_GSE286419.h5ad and regional_GSE274987_GSE241037.h5ad from
# processed_inputs/.
# Use X, dataset, patient, region and original annotation columns. Study scope notes come from
# TableS1_public_cohort_manifest.csv.
# Outputs in the working package's data/ directory: patient_state_expression.csv, patient_state_composition.csv,
# patient_ac_fraction.csv,
# ac_vs_other_ligand_contrasts.csv, meta_ligand_results.csv, patient_communication_scores.csv,
# regional_communication_tests.csv, regional_ac_fraction_tests.csv, cohort_manifest.csv,
# input_provenance.csv, excluded_datasets.csv and analysis_summary.json. Downstream stages use the published filenames.
# Usage: Run python 01_patient_expression.py from code/. Set PUBLIC_INPUT_ROOT to the original public inputs.
# Standalone releases write to 01_cohort_reanalysis/. Stage 14 also uses the patient-level functions below.

from __future__ import annotations

import gzip
import json
from importlib import import_module
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm, wilcoxon

paths = import_module("00a_project_paths")


def main() -> None:
    """Generate patient-level results and provenance tables from the original public inputs. Returns None."""
    stop_if_missing()
    expression_table_total, composition_table_total = read_primary_cohorts()
    tables = primary_tables(expression_table_total, composition_table_total)
    output = paths.analysis_output_root() / "data"

    # The minimal inputs also reproduce these eight tables. Use the same patient, region and gene keys for both
    # read-back and reference comparisons.
    for name, result_table in tables.items():
        paths.write_table(result_table, output / f"{name}.csv", primary_row_columns[name])

    # Count the cells actually included in the patient analysis. Measure the size of each source file used in this run.
    manifest_table = build_manifest(expression_table_total)
    paths.write_table(manifest_table, output / "cohort_manifest.csv", ["cohort"])
    provenance_table = input_provenance()
    paths.write_table(provenance_table, output / "input_provenance.csv", ["input"])

    # Keep the two published cohort exclusions and their original fields for the provenance table and downstream
    # figures.
    excluded_table = pd.DataFrame(
        [
            {
                "dataset": "GSE162631",
                "reason": (
                    "Processed public object did not provide a harmonized four-state malignant annotation; "
                    "AC-like counts were below the predefined 20-cell threshold in several patient-regions."
                ),
                "status": "excluded from primary label-based analysis; included in uniform-marker sensitivity analysis",
            },
            {
                "dataset": "GSE174554",
                "reason": (
                    "The prior intermediate clustering object was unavailable, so the earlier cluster-specific "
                    "results could not be regenerated from the recorded workflow."
                ),
                "status": "excluded; no legacy summary values reused",
            },
        ]
    )
    paths.write_table(excluded_table, output / "excluded_datasets.csv", ["dataset"])

    # Use unrounded q-values. For regional tests, also record whether the number of pairs permits a nominally
    # significant result.
    contrast_table = tables["ac_vs_other_ligand_contrasts"]
    meta_table = tables["meta_ligand_results"]
    regional_table = tables["regional_communication_tests"]
    significant_contrast_table = contrast_table.loc[
        contrast_table["q_value"] < 0.05, ["cohort", "gene", "median_difference", "q_value"]
    ]
    significant_meta_table = meta_table.loc[
        meta_table["q_value"] < 0.05, ["gene", "n_cohorts", "n_patients_total", "positive_cohorts", "meta_z", "q_value"]
    ]
    significant_region_table = regional_table.loc[
        regional_table["q_value"] < 0.05, ["comparison", "pair", "median_difference", "q_value"]
    ]
    summary = {
        "public_only": True,
        "minimum_cells_per_patient_region_state": minimum_cells,
        "analysis_unit": "patient-region",
        "cohorts": manifest_table.to_dict("records"),
        "significant_ac_ligand_contrasts_q_lt_0_05": significant_contrast_table.to_dict("records"),
        "meta_ligands_q_lt_0_05": significant_meta_table.to_dict("records"),
        "significant_regional_communication_q_lt_0_05": significant_region_table.to_dict("records"),
        "regional_communication_tests_capable_of_raw_p_lt_0_05": int(regional_table["can_attain_p_below_0_05"].sum()),
        "interpretation_boundary": (
            "Expression co-availability is an association index and is not evidence of physical signaling or causality."
        ),
    }
    paths.write_json(summary, output / "analysis_summary.json")
    print(f"Included {len(manifest_table)} cohorts and calculated {len(contrast_table)} cohort-level ligand contrasts.")


def stop_if_missing() -> None:
    """Report all missing public inputs before opening the first large matrix. Returns None."""
    missing = [str(path) for path in paths.primary_input_files().values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing public analysis inputs:\n" + "\n".join(missing))


def read_primary_cohorts(minimal_directory: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read five cohorts in their original order. Minimal files already contain normalized expression in X."""
    if minimal_directory is None:
        files = paths.primary_input_files()
        expression_table, composition_table = read_gse211376()
    else:
        files = {
            "scGBM": minimal_directory / "scGBM_minimal.h5ad",
            "snGBM": minimal_directory / "snGBM_minimal.h5ad",
            "GSE274987": minimal_directory / "regional_GSE274987_minimal.h5ad",
        }
        expression_table, composition_table = read_h5ad_cohort(
            minimal_directory / "GSE211376_minimal.h5ad",
            "GSE211376",
            "GSE211376",
            "patient",
            "region",
            "dataset",
            state_labels,
            "predicted.high_hierarchy",
            tam_myeloid,
        )
    expression_parts = [expression_table]
    composition_parts = [composition_table]

    # Only GSE273274 enters the primary scGBM comparison. Select the regional cohort by its legacy dataset value and
    # analyze the two platforms together.
    cohorts = [
        ("GSE273274", "scGBM", "patient_id", "tissue_type", "Minor_cluster", sc_state_labels, sc_myeloid),
        ("GSE276177", "snGBM", "patient", "tissue", "celltype", state_labels, celltype_myeloid),
        ("GSE286419", "snGBM", "patient", "tissue", "celltype", state_labels, celltype_myeloid),
        ("GSE274987", "GSE274987", "patient_id", "tissue_type", "celltype", state_labels, celltype_myeloid),
    ]
    for cohort, source, patient, region, state, labels, myeloid_mask in cohorts:
        expression_table, composition_table = read_h5ad_cohort(
            files[source], cohort, cohort, patient, region, "dataset", labels, state, myeloid_mask
        )
        expression_parts.append(expression_table)
        composition_parts.append(composition_table)

    expression_table_total = pd.concat(expression_parts, ignore_index=True)
    composition_table_total = pd.concat(composition_parts, ignore_index=True)
    result = expression_table_total, composition_table_total
    return result


# State order also sets the order used for tied uniform-marker scores. Both scGBM NPC labels map to one state.
malignant_states = ["AC-like", "MES-like", "OPC-like", "NPC-like"]
state_labels = {state: state for state in malignant_states}
sc_state_labels = {"AC": "AC-like", "MES1": "MES-like", "OPC": "OPC-like", "NPC1": "NPC-like", "NPC2": "NPC-like"}


def tam_myeloid(obs_table: pd.DataFrame) -> pd.Series:
    """Return a cell-aligned mask for the two TAM classes in the published GSE211376 annotations."""
    selected = obs_table["predicted.high_hierarchy"].astype(str).isin(["TAM-MG", "TAM-BDM"])
    return selected


def sc_myeloid(obs_table: pd.DataFrame) -> pd.Series:
    """Return a mask for myeloid cells using the broad scGBM categories."""
    selected = obs_table["Major_category"].astype(str).isin(["TAM/Macrophage", "Microglia", "Monocyte"])
    return selected


def celltype_myeloid(obs_table: pd.DataFrame) -> pd.Series:
    """Return a mask for myeloid cells using the celltype labels in the snGBM and regional objects."""
    selected = obs_table["celltype"].astype(str).isin(["Macrophage", "Microglia", "Monocyte"])
    return selected


def read_h5ad_cohort(
    path: Path,
    cohort: str,
    dataset_filter: str | None,
    patient_col: str,
    region_col: str,
    dataset_col: str,
    state_map: dict[str, str],
    state_col: str,
    myeloid_mask,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select X with the specified columns, state map and myeloid mask. Require all genes and return state expression
    and malignant-cell composition by patient.
    """
    source = ad.read_h5ad(path, backed="r")
    try:
        obs_table = source.obs
        states = obs_table[state_col].astype(str).map(state_map)
        selected = states.notna().to_numpy() | np.asarray(myeloid_mask(obs_table), dtype=bool)
        if dataset_filter is not None:
            selected &= obs_table[dataset_col].astype(str).eq(dataset_filter).to_numpy()

        # Keep the malignant state when both labels apply. Assign Myeloid only to cells with a myeloid label and no
        # recognized malignant state.
        positions = np.flatnonzero(selected)
        matrix = selected_expression(source, positions, primary_genes)
        selected_table = obs_table.iloc[positions]
        cell_table = pd.DataFrame(
            {
                "cohort": cohort,
                "dataset": selected_table[dataset_col].astype(str).to_numpy(),
                "patient": selected_table[patient_col].astype(str).to_numpy(),
                "region": selected_table[region_col].astype(str).to_numpy(),
                "state": states.iloc[positions].fillna("Myeloid").to_numpy(),
            }
        )
    finally:
        source.file.close()

    result = summarize_states(matrix, cell_table)
    return result


def selected_expression(source, positions: np.ndarray, genes: list[str]) -> np.ndarray:
    """Extract a small matrix in the requested cell and gene order from open AnnData. Require every target gene."""
    columns = source.var_names.get_indexer(genes)
    missing = [gene for gene, column in zip(genes, columns) if column < 0]
    if missing:
        raise ValueError(f"Genes missing from the expression matrix: {missing}")
    matrix = source.X[positions, :][:, columns]
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    expression = np.asarray(matrix, dtype=np.float32)
    return expression


# Six ligands and seven prespecified pairs. Keep each gene's first occurrence so the matrix columns match the published
# header.
ligands = ["TGFB2", "TGFB1", "ANXA1", "SPP1", "CSF1", "CCL2"]
ligand_receptor_pairs = [
    ("TGFB2", "TGFBR1"),
    ("TGFB2", "TGFBR2"),
    ("TGFB1", "TGFBR1"),
    ("ANXA1", "FPR1"),
    ("SPP1", "CD44"),
    ("CSF1", "CSF1R"),
    ("CCL2", "CCR2"),
]
primary_genes = list(dict.fromkeys(ligands + [receptor for _, receptor in ligand_receptor_pairs]))


def read_gse211376() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize raw counts and use the original states. Require a nonzero count for each primary target gene."""
    metadata_table, matrix, _ = read_count_expression(primary_genes)
    if metadata_table.isna().any(axis=None):
        raise ValueError("GSE211376 count columns do not align with the full metadata")
    if not np.all(np.any(matrix != 0, axis=0)):
        raise ValueError("A GSE211376 primary target gene has an all-zero count column")

    labels = {**state_labels, "TAM-MG": "Myeloid", "TAM-BDM": "Myeloid"}
    states = metadata_table["predicted.high_hierarchy"].map(labels)
    keep = states.notna().to_numpy()
    cell_table = pd.DataFrame(
        {
            "cohort": "GSE211376",
            "dataset": "GSE211376",
            "patient": metadata_table.loc[keep, "patient"].astype(str),
            "region": "Tumor",
            "state": states.loc[keep],
        }
    )
    result = summarize_states(matrix[keep], cell_table.reset_index(drop=True))
    return result


def read_count_expression(genes: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Stream GSE211376 counts. Return metadata in count-column order, log1p target expression and all-gene totals."""
    files = paths.primary_input_files()
    metadata_table = pd.read_csv(files["GSE211376_meta"], index_col=0)
    with gzip.open(files["GSE211376_counts"], "rt") as stream:
        header = stream.readline().rstrip("\n").split("\t")
        cells = header[1:] if header[0] == "" else header
        totals = np.zeros(len(cells), dtype=np.float64)
        gene_counts = {}
        for line in stream:
            symbol, *fields = line.rstrip("\n").split("\t")
            values = np.asarray(fields, dtype=np.float32)
            if len(values) != len(cells):
                raise ValueError(f"GSE211376 {symbol}: count column number differs from the header")
            totals += values
            if symbol in genes:
                gene_counts[symbol] = values

    # Use all genes in the library-size denominator and retain only the requested genes in memory. Zero-count cells
    # retain zero expression.
    missing = sorted(set(genes) - gene_counts.keys())
    if missing:
        raise ValueError(f"Missing GSE211376 target genes: {missing}")
    scale = totals / 1e4
    scale[scale == 0] = 1.0
    matrix = np.column_stack([np.log1p(gene_counts[gene] / scale) for gene in genes]).astype(np.float32)

    metadata_table = metadata_table.reindex(cells)
    if metadata_table[["predicted.high_hierarchy", "patient"]].isna().any(axis=None):
        raise ValueError("A GSE211376 count column lacks a patient ID or cell annotation")
    result = metadata_table, matrix, totals
    return result


def summarize_states(matrix: np.ndarray, cell_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize the same cells in two tables. Composition uses all annotated malignant states in each patient-region.
    """
    expression_table = aggregate_selected_expression(matrix, cell_table, primary_genes)
    malignant = expression_table["state"].isin(malignant_states)
    composition_table = expression_table.loc[malignant, [*patient_region, "state", "n_cells"]].copy()
    denominators = composition_table.groupby(patient_region, observed=True)["n_cells"].transform("sum")
    composition_table["fraction_of_malignant"] = composition_table["n_cells"] / denominators
    result = expression_table, composition_table.reset_index(drop=True)
    return result


patient_region = ["cohort", "dataset", "patient", "region"]


def aggregate_selected_expression(
    expression: np.ndarray, metadata_table: pd.DataFrame, genes: list[str]
) -> pd.DataFrame:
    """Each matrix row is a cell. Compute gene means and fractions of nonzero cells within patient, region and state."""
    if expression.shape != (len(metadata_table), len(genes)):
        raise ValueError("Cell metadata and gene list do not match the expression matrix dimensions")
    group_columns = [*patient_region, "state"]
    rows = []
    for group, positions in metadata_table.groupby(group_columns, observed=True).indices.items():
        block = expression[positions]
        row = dict(zip(group_columns, group))
        row["n_cells"] = len(positions)
        for column, gene in enumerate(genes):
            values = block[:, column]
            row[f"mean_{gene}"] = float(np.mean(values))
            row[f"detect_{gene}"] = float(np.mean(values > 0))
        rows.append(row)
    expression_table = pd.DataFrame(rows)
    return expression_table


def primary_tables(expression_table: pd.DataFrame, composition_table: pd.DataFrame) -> dict:
    """Build eight published results from the two patient-state tables and check denominators and state cell counts."""
    totals = composition_table.groupby(patient_region, observed=True)["fraction_of_malignant"].sum()
    if totals.empty or not np.allclose(totals, 1) or (expression_table["n_cells"] <= 0).any():
        raise ValueError("Malignant-state fractions must sum to one per region and each state must contain cells")

    contrast_table = ac_vs_other_contrasts(expression_table)
    score_table = communication_scores(expression_table)
    ac_table, fraction_test_table = composition_tests(composition_table)
    tables = {
        "patient_state_expression": expression_table,
        "patient_state_composition": composition_table,
        "ac_vs_other_ligand_contrasts": contrast_table,
        "meta_ligand_results": meta_ligand_results(contrast_table),
        "patient_communication_scores": score_table,
        "regional_communication_tests": regional_pair_tests(score_table),
        "patient_ac_fraction": ac_table,
        "regional_ac_fraction_tests": fraction_test_table,
    }
    return tables


# Count each patient's regions separately. Each ligand contrast row represents one cohort, region and gene.
primary_row_columns = {
    "patient_state_expression": ["cohort", "patient", "region", "state"],
    "patient_state_composition": ["cohort", "patient", "region", "state"],
    "ac_vs_other_ligand_contrasts": ["cohort", "region", "gene"],
    "meta_ligand_results": ["gene"],
    "patient_communication_scores": ["cohort", "patient", "region", "pair"],
    "regional_communication_tests": ["cohort", "comparison", "pair"],
    "patient_ac_fraction": ["cohort", "patient", "region", "state"],
    "regional_ac_fraction_tests": ["cohort", "comparison"],
}


def ac_vs_other_contrasts(expression_table: pd.DataFrame) -> pd.DataFrame:
    """Pair by patient, filter each state by cell count, then weight eligible non-AC states by their cell counts."""
    records = []
    malignant_table = expression_table.loc[expression_table["state"].isin(malignant_states)]
    for (cohort, region), region_table in malignant_table.groupby(["cohort", "region"], observed=True):
        if region != ("Core" if cohort == "GSE274987" else "Tumor"):
            continue
        eligible_table = region_table.loc[region_table["n_cells"] >= minimum_cells]
        for gene in ligands:
            values = patient_ligand_differences(eligible_table, gene)
            differences = [row["difference"] for row in values]
            records.append(
                {
                    "cohort": cohort,
                    "region": region,
                    "gene": gene,
                    "n_patients": len(differences),
                    "median_difference": float(np.median(differences)) if differences else np.nan,
                    "mean_difference": float(np.mean(differences)) if differences else np.nan,
                    "p_value": safe_wilcoxon(differences),
                    "patient_values": json.dumps(values, ensure_ascii=False),
                }
            )
    contrast_table = pd.DataFrame(records)
    contrast_table["q_value"] = contrast_table.groupby("cohort", observed=True)["p_value"].transform(bh_adjust)
    return contrast_table


# The published AC threshold applies per patient-region. In the primary contrast, each other state must meet this
# threshold separately.
minimum_cells = 20


def patient_ligand_differences(eligible_table: pd.DataFrame, gene: str) -> list[dict]:
    """Return one AC mean and one cell-count-weighted mean of the other eligible states per patient."""
    records = []
    for patient, patient_table in eligible_table.groupby("patient", observed=True):
        ac_table = patient_table.loc[patient_table["state"] == "AC-like"]
        other_table = patient_table.loc[patient_table["state"] != "AC-like"]
        if ac_table.empty or other_table.empty:
            continue
        ac_mean = float(ac_table[f"mean_{gene}"].item())
        weights = other_table["n_cells"].to_numpy(float)
        other_mean = float(np.average(other_table[f"mean_{gene}"].to_numpy(float), weights=weights))
        records.append(
            {"patient": patient, "ac_mean": ac_mean, "other_mean": other_mean, "difference": ac_mean - other_mean}
        )
    return records


def safe_wilcoxon(diff: np.ndarray) -> float:
    """Use a two-sided paired Wilcoxon test. Return missing for fewer than three finite differences or all near-zero
    differences.
    """
    values = np.asarray(diff, dtype=float)
    values = values[np.isfinite(values)]
    p_value = np.nan
    if len(values) >= 3 and not np.allclose(values, 0):
        p_value = float(wilcoxon(values, alternative="two-sided", method="auto").pvalue)
    return p_value


def bh_adjust(values: pd.Series) -> pd.Series:
    """Apply BH adjustment within the supplied family, excluding missing P values from ranking and retaining their
    positions.
    """
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().to_numpy(float)
    if not len(valid):
        return adjusted
    order = np.argsort(valid)
    ranked = valid[order] * len(valid) / np.arange(1, len(valid) + 1)
    corrected = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty_like(corrected)
    restored[order] = np.clip(corrected, 0, 1)
    adjusted.loc[values.notna()] = restored
    return adjusted


def communication_scores(expression_table: pd.DataFrame) -> pd.DataFrame:
    """Compute the geometric mean for each prespecified pair in patient-regions with enough AC and myeloid cells."""
    records = []
    eligible_table = expression_table.loc[expression_table["n_cells"] >= minimum_cells]
    for group, region_table in eligible_table.groupby(patient_region, observed=True):
        states_table = region_table.set_index("state", verify_integrity=True)
        if "AC-like" not in states_table.index or "Myeloid" not in states_table.index:
            continue
        for ligand, receptor in ligand_receptor_pairs:
            ligand_mean = float(states_table.at["AC-like", f"mean_{ligand}"])
            receptor_mean = float(states_table.at["Myeloid", f"mean_{receptor}"])
            records.append(
                {
                    **dict(zip(patient_region, group)),
                    "pair": f"{ligand}-{receptor}",
                    "ligand": ligand,
                    "receptor": receptor,
                    "n_ac": int(states_table.at["AC-like", "n_cells"]),
                    "n_myeloid": int(states_table.at["Myeloid", "n_cells"]),
                    "ligand_mean": ligand_mean,
                    "receptor_mean": receptor_mean,
                    "coavailability": float(np.sqrt(max(ligand_mean, 0) * max(receptor_mean, 0))),
                }
            )
    score_table = pd.DataFrame(records)
    return score_table


def meta_ligand_results(contrast_table: pd.DataFrame) -> pd.DataFrame:
    """Pool cohorts with patient-level P values for each gene. Leave one cohort out if at least three are estimable."""
    records = []
    for gene, gene_table in contrast_table.groupby("gene", observed=True):
        eligible_table = gene_table.dropna(subset=["p_value", "median_difference"])
        eligible_table = eligible_table.loc[eligible_table["n_patients"] >= 3]
        if eligible_table.empty:
            continue
        meta_z, p_value = combine_meta(eligible_table)

        # Recompute the denominator after each omission. Adjust the six primary P values and the six worst leave-one-out
        # P values as separate families.
        omitted_results = []
        if len(eligible_table) >= 3:
            for omitted in eligible_table["cohort"]:
                z_omitted, p_omitted = combine_meta(eligible_table.loc[eligible_table["cohort"] != omitted])
                omitted_results.append({"omitted": omitted, "meta_z": z_omitted, "p_value": p_omitted})
        effects = eligible_table[["cohort", "n_patients", "median_difference", "p_value"]].to_dict("records")
        records.append(
            {
                "gene": gene,
                "n_cohorts": len(eligible_table),
                "n_patients_total": int(eligible_table["n_patients"].sum()),
                "positive_cohorts": int((eligible_table["median_difference"] > 0).sum()),
                "negative_cohorts": int((eligible_table["median_difference"] < 0).sum()),
                "meta_z": meta_z,
                "p_value": p_value,
                "loo_max_p_value": max((row["p_value"] for row in omitted_results), default=np.nan),
                "loo_results": json.dumps(omitted_results, ensure_ascii=False),
                "cohort_effects": json.dumps(effects, ensure_ascii=False),
            }
        )
    meta_table = pd.DataFrame(records)
    meta_table["q_value"] = bh_adjust(meta_table["p_value"])
    meta_table["loo_q_value"] = bh_adjust(meta_table["loo_max_p_value"])
    meta_table = meta_table.sort_values("q_value").reset_index(drop=True)
    return meta_table


def combine_meta(cohort_table: pd.DataFrame) -> tuple[float, float]:
    """Combine two-sided P values using cohort median signs and square-root patient-count weights. Empty input returns
    missing.
    """
    if cohort_table.empty:
        result = np.nan, np.nan
        return result
    p_values = np.clip(cohort_table["p_value"].to_numpy(float), 1e-15, 1.0)
    signs = np.sign(cohort_table["median_difference"].to_numpy(float))
    z_scores = norm.isf(p_values / 2) * signs
    weights = np.sqrt(cohort_table["n_patients"].to_numpy(float))
    z_combined = float(np.sum(weights * z_scores) / np.sqrt(np.sum(weights**2)))
    result = z_combined, float(2 * norm.sf(abs(z_combined)))
    return result


def regional_pair_tests(score_table: pd.DataFrame) -> pd.DataFrame:
    """Compare patient differences for each ligand-receptor pair across two regions. Adjust P values within each
    regional comparison.
    """
    rows = []
    for label, cohort, region_a, region_b in regional_comparisons:
        cohort_table = score_table.loc[score_table["cohort"] == cohort]
        for pair, pair_table in cohort_table.groupby("pair", observed=True):
            statistics = paired_region_statistics(pair_table, "coavailability", region_a, region_b)
            rows.append(
                {
                    "comparison": label,
                    "cohort": cohort,
                    "region_a": region_a,
                    "region_b": region_b,
                    "pair": pair,
                    **statistics,
                }
            )
    test_table = pd.DataFrame(rows)
    test_table["q_value"] = test_table.groupby("comparison", observed=True)["p_value"].transform(bh_adjust)
    return test_table


# Subtract the second region from the first. Both comparisons use the same joint regional cohort.
regional_comparisons = [
    ("GSE273274", "GSE273274", "Tumor", "Peritumoral"),
    ("GSE276177", "GSE276177", "Tumor", "Peritumoral"),
    ("GSE286419", "GSE286419", "Tumor", "Peritumoral"),
    ("GSE274987_core_peripheral", "GSE274987", "Core", "Peripheral"),
    ("GSE274987_core_edge", "GSE274987", "Core", "Edge"),
]


def composition_tests(composition_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair only patient-regions with AC records. Leave absent states missing and use one BH family for all composition
    tests.
    """
    ac_table = composition_table.loc[composition_table["state"] == "AC-like"].copy()
    rows = []
    for label, cohort, region_a, region_b in regional_comparisons:
        cohort_table = ac_table.loc[ac_table["cohort"] == cohort]
        statistics = paired_region_statistics(cohort_table, "fraction_of_malignant", region_a, region_b)
        del statistics["mean_difference"]
        rows.append({"comparison": label, "cohort": cohort, "region_a": region_a, "region_b": region_b, **statistics})
    test_table = pd.DataFrame(rows)
    test_table["q_value"] = bh_adjust(test_table["p_value"])
    result = ac_table, test_table
    return result


def paired_region_statistics(region_table: pd.DataFrame, column: str, region_a: str, region_b: str) -> dict:
    """Pair regions by patient. With no pairs, report zero pairs and empty statistics. Preserve the published
    differences without IDs.
    """
    wide_table = region_table.pivot_table(index="patient", columns="region", values=column, aggfunc="mean")
    differences = np.array([], dtype=float)
    if region_a in wide_table and region_b in wide_table:
        paired_table = wide_table[[region_a, region_b]].dropna()
        differences = paired_table[region_a].to_numpy(float) - paired_table[region_b].to_numpy(float)

    # Count differences outside the near-zero tolerance for the theoretical lower bound. The Wilcoxon test uses all
    # paired differences.
    nonzero = count_nonzero_pairs(differences)
    minimum_p = minimum_attainable_two_sided_p(nonzero)
    statistics = {
        "n_patients": len(differences),
        "n_nonzero_pairs": nonzero,
        "minimum_attainable_p": minimum_p,
        "can_attain_p_below_0_05": bool(minimum_p < 0.05),
        "median_difference": float(np.median(differences)) if len(differences) else np.nan,
        "mean_difference": float(np.mean(differences)) if len(differences) else np.nan,
        "p_value": safe_wilcoxon(differences),
        "patient_differences": json.dumps(differences.tolist()),
    }
    return statistics


def count_nonzero_pairs(diff: np.ndarray) -> int:
    """Count finite patient pairs outside the original near-zero tolerance using isclose."""
    values = np.asarray(diff, dtype=float)
    finite = values[np.isfinite(values)]
    count = int(np.count_nonzero(~np.isclose(finite, 0.0)))
    return count


def minimum_attainable_two_sided_p(n_pairs: int) -> float:
    """Return the theoretical minimum two-sided signed-rank P from the nonzero pair count, or missing if there are no
    pairs.
    """
    minimum = float(2 ** (1 - n_pairs)) if n_pairs > 0 else np.nan
    return minimum


def build_manifest(expression_table: pd.DataFrame) -> pd.DataFrame:
    """Count included patients and cells per cohort. Get patient counts by source from the joint object's sequencing
    platforms.
    """
    reference_path = Path(__file__).resolve().parents[1] / "source_data" / "TableS1_public_cohort_manifest.csv"
    reference_table = pd.read_csv(reference_path)
    source_notes = reference_table.set_index("cohort", verify_integrity=True)["analysis_scope_note"]
    rows = []
    for cohort, cohort_table in expression_table.groupby("cohort", observed=True):
        patient_count = cohort_table["patient"].nunique()
        accessions = cohort
        if cohort == "GSE274987":
            accessions = "GSE274986; GSE274987"
            platform_counts = regional_platform_counts(cohort_table["patient"])
            scope = (
                f"Joint Wu et al. regional cohort: {platform_counts['Multiome']} GSE274986 Multiome patients and "
                f"{platform_counts['ParseBio']} GSE274987 ParseBio patients analyzed together; the legacy "
                "internal cohort label is GSE274987"
            )
        elif cohort == "GSE211376":
            scope = f"{patient_count} tumor patients analyzed"
        else:
            tumor = cohort_table.loc[cohort_table["region"] == "Tumor", "patient"]
            peritumoral = cohort_table.loc[cohort_table["region"] == "Peritumoral", "patient"]
            matched = len(set(tumor) & set(peritumoral))
            scope = f"{matched} matched tumor-peritumoral patients analyzed"

            # The text after the semicolon describes the source study. Count matched patients by intersecting the two
            # regions in this run.
            if cohort == "GSE276177":
                source_note = source_notes[cohort].split("; ", maxsplit=1)[1]
                scope = f"{matched} matched patients analyzed; {source_note}"
            elif cohort == "GSE286419":
                source_note = source_notes[cohort].split("; ", maxsplit=1)[1]
                scope = f"{matched} matched patients from the public confident-sample subset analyzed; {source_note}"

        counts = cohort_table.groupby(["patient", "region", "state"], observed=True)["n_cells"].first()
        rows.append(
            {
                "cohort": cohort,
                "accessions": accessions,
                "patients": int(patient_count),
                "regions": "; ".join(sorted(cohort_table["region"].unique())),
                "included_cells": int(counts.sum()),
                "analysis_unit": "patient-region",
                "analysis_scope_note": scope,
            }
        )
    manifest_table = pd.DataFrame(rows)
    return manifest_table


def regional_platform_counts(patients: pd.Series) -> pd.Series:
    """Count included regional patients by platform. Each must occur once in the patient-to-platform mapping."""
    source = ad.read_h5ad(paths.primary_input_files()["GSE274987"], backed="r")
    try:
        obs_table = source.obs
        selected = obs_table["dataset"].astype(str).eq("GSE274987") & obs_table["patient_id"].astype(str).isin(patients)
        member_table = obs_table.loc[selected, ["patient_id", "Platform"]].astype(str).drop_duplicates()
    finally:
        source.file.close()
    if member_table["patient_id"].duplicated().any() or set(member_table["patient_id"]) != set(patients):
        raise ValueError("Each patient in the joint regional cohort must map to exactly one platform")
    if set(member_table["Platform"]) != {"Multiome", "ParseBio"}:
        raise ValueError("The joint cohort platforms do not match the two published sources")
    counts = member_table["Platform"].value_counts()
    return counts


def input_provenance() -> pd.DataFrame:
    """Record actual filenames, relative locations and sizes for the five inputs, retaining the published input names.
    """
    descriptions = {
        "GSE211376_counts": ("GSE211376", "GEO raw count matrix"),
        "GSE211376_meta": ("GSE211376", "GEO cell metadata"),
        "scGBM": ("GSE162631; GSE273274", "project-generated processed AnnData"),
        "snGBM": ("GSE276177; GSE286419", "project-generated processed AnnData"),
        "GSE274987": (
            "GSE274986; GSE274987; GSE241037",
            "project-generated processed AnnData; joint GSE274986/GSE274987 "
            "cohort selected through legacy dataset value GSE274987; GSE241037 excluded",
        ),
    }
    input_root = paths.public_input_root(paths.project_root())
    rows = []
    for name, path in paths.primary_input_files().items():
        accessions, description = descriptions[name]
        rows.append(
            {
                "input": name,
                "accessions": accessions,
                "input_type": description,
                "source_file_name": path.name,
                "repository_target_path": path.relative_to(input_root).as_posix(),
                "bytes": path.stat().st_size,
            }
        )
    provenance_table = pd.DataFrame(rows)
    return provenance_table


if __name__ == "__main__":
    main()
