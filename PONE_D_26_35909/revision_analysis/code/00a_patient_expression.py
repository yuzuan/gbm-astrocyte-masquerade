# ==============================================================================
# 00a_patient_expression.py -- Shared functions for the PONE-D-26-35909 revision.
# Patient-level TGFB2 expression and random-effects analyses.
#
# Patient identifiers are unique within each cohort.
# These functions align cell annotations and expression matrices, calculate patient contrasts,
# and read or write the tables used in the supplementary analyses.
# The numbered scripts call these functions at each analysis step.
# Use your editor's Go to Definition command to trace a function from its call site.
# Table keys match the identifiers in the published supplementary files.

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


# Map published table identifiers to the output paths shared by both reproduction routes.
table_files = {
    "count_input_audit": "01_counts_and_preprocessing/01_count_input_audit.csv",
    "preprocessing_patient_contrasts": "01_counts_and_preprocessing/02_patient_expression_contrasts.csv",
    "preprocessing_eligibility": "01_counts_and_preprocessing/03_patient_eligibility.csv",
    "preprocessing_cell_membership": "01_counts_and_preprocessing/04_included_cell_counts.csv",
    "GSE162631_state_audit": "01_counts_and_preprocessing/05_gse162631_cell_states.csv",
    "GSE162631_annotation_crosswalk": "01_counts_and_preprocessing/06_gse162631_annotation_crosswalk.csv",
    "preprocessing_meta_stouffer": "01_counts_and_preprocessing/07_combined_cohort_tests.csv",
    "preprocessing_cohort_summary": "01_counts_and_preprocessing/08_per_cohort_tests.csv",
    "GSE162631_public_count_audit": "02_source_count_audit/01_tumor_sample_count_audit.csv",
    "GSE162631_source_sensitivity": "02_source_count_audit/02_source_count_patient_contrasts.csv",
    "GSE162631_source_eligibility": "02_source_count_audit/03_source_count_eligibility.csv",
    "GSE162631_leave_patient_out": "02_source_count_audit/04_leave_one_patient_out.csv",
    "meta_patient_inputs": "03_model_inputs/01_model_patient_contrasts.csv",
    "meta_cohort_effects": "04_random_effects_models/01_cohort_effects.csv",
    "meta_summary": "04_random_effects_models/02_pooled_effects.csv",
    "meta_leave_one_out": "04_random_effects_models/03_leave_one_cohort_out.csv",
}


def result_root():
    """Resolve the output root for the current run from PLOS_REVISION_DIR or the package directory."""
    default_root = Path(__file__).resolve().parents[1] / "03_result"
    directory = Path(os.environ.get("PLOS_REVISION_DIR", default_root)).expanduser()
    return directory


def read_table(name):
    """Read an analysis table by its published identifier."""
    result_table = pd.read_csv(result_root() / table_files[name])
    return result_table


def write_table(name, result_table):
    """Create the output directory, write the table and check its shape and columns on disk."""
    path = result_root() / table_files[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    result_table.to_csv(path, index=False)
    saved_table = pd.read_csv(path)
    if saved_table.shape != result_table.shape or list(saved_table) != list(result_table):
        raise ValueError(f"Output dimensions or columns differ after writing {path.name}")


def patient_from_tumor_sample(sample):
    """Read the patient identifier from an original or publicly recoded tumor sample name."""
    public = re.fullmatch(r"(P[1-9]\d*)-S1", sample)
    if public:
        subject = public.group(1)
        return subject
    source = re.fullmatch(r"(R[1-9]\d*)_T", sample)
    if source is None:
        raise ValueError(f"Unrecognized tumor sample identifier: {sample}")
    subject = "GSE162631_" + source.group(1)
    return subject


def read_cohort_cache(cache, cohort):
    """Align annotations with the count and expression matrices by cell and panel gene order."""
    cell_table = pd.read_csv(cache / f"01_{cohort}_cells.csv.gz", dtype={"cell_id": str})
    if cell_table.empty or not cell_table["cohort"].eq(cohort).all():
        raise ValueError(f"Unexpected cohort in the cell cache for {cohort}")
    ids = cell_table["cell_id"]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError(f"Missing or duplicate cell identifiers in {cohort}")

    with np.load(cache / f"02_{cohort}_expression.npz", allow_pickle=False) as saved:
        counts = saved["counts"]
        stored = saved["stored"]
        expected_shape = (len(ids), len(genes))
        if counts.shape != expected_shape or stored.shape != expected_shape:
            raise ValueError(f"Cell table and matrix dimensions differ for {cohort}")
        if not np.array_equal(saved["cell_ids"], ids.to_numpy()) or saved["genes"].tolist() != genes:
            raise ValueError(f"Cell or panel gene order differs for {cohort}")
    result = cell_table, counts, stored
    return result


# Marker order determines the four malignant-state score columns. TGFB2 is the outcome and is excluded.
state_markers = {
    "AC-like": ["GFAP", "S100B", "HOPX", "SLC1A3", "MLC1"],
    "MES-like": ["VIM", "CD44", "CHI3L1", "HILPDA", "DDIT3", "ENO2"],
    "OPC-like": ["PLP1", "OLIG1", "OMG", "TNR", "ALCAM"],
    "NPC-like": ["SOX4", "SOX11", "DCX", "CD24", "STMN1", "STMN2"],
}
lineage_markers = {
    "Endothelial": ["PECAM1", "VWF", "CLDN5", "KDR"],
    "Myeloid": ["PTPRC", "LST1", "TYROBP"],
    "Oligodendrocyte": ["MBP", "MOG", "MOBP"],
}
states = list(state_markers)
marker_genes = sum(state_markers.values(), []) + sum(lineage_markers.values(), [])
genes = list(dict.fromkeys(["TGFB2", *marker_genes]))


def assign_states(expression):
    """Score a cell-by-panel-gene matrix and return cell states and margins between the two highest scores."""
    state_scores = []
    for markers in state_markers.values():
        columns = [genes.index(gene) for gene in markers]
        state_scores.append(expression[:, columns].mean(axis=1))

    # Rank states within each cell. The original analysis required a minimum score difference of 0.05.
    scores = np.column_stack(state_scores)
    order = np.argsort(scores, axis=1)
    first = order[:, -1]
    rows = np.arange(len(expression))
    margin = scores[rows, first] - scores[rows, order[:, -2]]
    labels = np.array(states, dtype=object)[first]
    labels[margin < 0.05] = "Ambiguous"
    result = labels, margin
    return result


def summarize_patients(cell_table, expression, counts, labels, label_method, scenario, pseudobulk=False):
    """Return within-patient TGFB2 contrasts and eligibility records for all patients in one cohort."""
    cohort = cell_table["cohort"].iat[0]
    region = "Core" if cohort == "GSE274987" else "Tumor"
    annotated_table = cell_table.assign(state=labels)
    contrast_rows = []
    eligibility_rows = []
    gene_index = genes.index("TGFB2")

    # Retain matrix row indices. Primary labels require 20 cells per reference state. uniform labels pool other states.
    for patient, patient_table in annotated_table.loc[annotated_table["region"] == region].groupby("patient"):
        state_counts = patient_table["state"].value_counts()
        other_states = states[1:]
        if label_method == "primary":
            other_states = [state for state in other_states if state_counts.get(state, 0) >= 20]
        ac = patient_table.index[patient_table["state"] == "AC-like"].to_numpy()
        other = patient_table.index[patient_table["state"].isin(other_states)].to_numpy()
        eligible = len(ac) >= 20 and len(other) >= 20
        identity = {
            "scenario": scenario,
            "labels": label_method,
            "cohort": cohort,
            "patient": patient,
            "region": region,
            "n_ac": len(ac),
            "n_other": len(other),
        }
        eligibility_rows.append({**identity, "eligible": eligible})
        if not eligible:
            continue

        # Cell means and pooled counts use different expression scales, denominators and pseudocounts.
        if pseudobulk:
            ac_value = np.log2(
                (counts[ac, gene_index].sum() + 0.5) / (cell_table.loc[ac, "count_library"].sum() + 1) * 1e6
            )
            other_value = np.log2(
                (counts[other, gene_index].sum() + 0.5)
                / (cell_table.loc[other, "count_library"].sum() + 1)
                * 1e6
            )
            scale = "pseudobulk log2 CPM"
        else:
            ac_value = float(expression[ac, gene_index].mean())
            other_value = float(expression[other, gene_index].mean())
            scale = "mean log1p counts per 1000" if scenario == "counts_1k" else "mean log1p counts per 10000"

        # Each row compares the same patient's two cell groups. Compute detection and QC metrics within those groups.
        contrast_rows.append(
            {
                **identity,
                "ac_mean": ac_value,
                "other_mean": other_value,
                "difference": ac_value - other_value,
                "ac_detected_fraction": float(np.mean(counts[ac, gene_index] > 0)),
                "other_detected_fraction": float(np.mean(counts[other, gene_index] > 0)),
                "ac_median_library": float(np.median(cell_table.loc[ac, "count_library"])),
                "other_median_library": float(np.median(cell_table.loc[other, "count_library"])),
                "ac_median_genes": float(np.median(cell_table.loc[ac, "count_n_genes"])),
                "other_median_genes": float(np.median(cell_table.loc[other, "count_n_genes"])),
                "ac_median_mt_pct": float(np.median(cell_table.loc[ac, "count_mt_pct"])),
                "other_median_mt_pct": float(np.median(cell_table.loc[other, "count_mt_pct"])),
                "scale": scale,
            }
        )

    result = contrast_rows, eligibility_rows
    return result
