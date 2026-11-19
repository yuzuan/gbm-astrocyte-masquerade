#!/usr/bin/env python3
# Purpose: Reassign the four malignant states using the same markers in every cohort and test the sensitivity of
# within-patient TGFB2 differences to the annotation method.
# Inputs: GSE211376_raw_counts.tsv.gz and GSE211376_metadata.csv.gz (patient, predicted.high_hierarchy).
# The three AnnData files are scGBM_GSE162631_GSE273274.h5ad, snGBM_GSE276177_GSE286419.h5ad and
# regional_GSE274987_GSE241037.h5ad.
# Read X, dataset, patient, region and original malignant annotations. PUBLIC_INPUT_ROOT contains raw_inputs/ and
# processed_inputs/.
# Outputs: data/uniform_annotation_patient_contrasts.csv, data/uniform_annotation_cohort_results.csv and
# data/uniform_annotation_summary.json. Standalone releases write to 01_cohort_reanalysis/.
# Usage: Run python 02_uniform_annotation_sensitivity.py from code/. Python shares the patient-level routines and fits
# the statsmodels HC3 model.

from __future__ import annotations

from importlib import import_module
from itertools import chain
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

primary = import_module("01_patient_expression")
paths = import_module("00a_project_paths")


def main() -> None:
    """Use the same malignant cells for patient contrasts, cohort summaries and lineage-score adjustment. Returns None.
    """
    primary.stop_if_missing()
    cell_table_total = load_uniform_cells()
    patient_table = patient_contrasts(cell_table_total)
    cohort_table = cohort_results(patient_table)
    summary = {
        "uniform_marker_assignment": True,
        "tgfb2_not_used_for_state_definition": all("TGFB2" not in genes for genes in markers.values()),
        "minimum_score_difference": minimum_score_difference,
        "minimum_cells_per_group": primary.minimum_cells,
        "marker_gene_standardization": False,
        "meta": meta_result(cohort_table),
        "lineage_adjustment": lineage_adjustment(patient_table),
        "cohorts": cohort_table.to_dict("records"),
    }

    # Each eligible patient contributes one core-region contrast. Check patient counts before saving patient and cohort
    # tables.
    if int(cohort_table["n_patients"].sum()) != len(patient_table):
        raise ValueError("Cohort patient counts differ from the patient contrast table")
    output = paths.analysis_output_root() / "data"
    paths.write_table(patient_table, output / "uniform_annotation_patient_contrasts.csv", ["cohort", "patient"])
    paths.write_table(cohort_table, output / "uniform_annotation_cohort_results.csv", ["cohort"])
    paths.write_json(summary, output / "uniform_annotation_summary.json")
    print(f"{len(patient_table)} patient contrasts; {summary['meta']['n_cohorts']} cohorts eligible for pooling.")


# The baseline assignment requires a gap of at least 0.05 between the two highest mean marker scores. Stage 03 varies
# this threshold.
minimum_score_difference = 0.05


def load_uniform_cells(standardize: bool = False, score_threshold: float = minimum_score_difference) -> pd.DataFrame:
    """Score each cohort independently and combine cells in their original order. Include both GSE162631 and GSE273274
    from scGBM.
    """
    parts = [read_gse211376(standardize, score_threshold)]
    files = paths.primary_input_files()
    cohorts = [
        ("GSE162631", "scGBM", "patient_id", "tissue_type", sc_malignant),
        ("GSE273274", "scGBM", "patient_id", "tissue_type", sc_malignant),
        ("GSE276177", "snGBM", "patient", "tissue", celltype_malignant),
        ("GSE286419", "snGBM", "patient", "tissue", celltype_malignant),
        ("GSE274987", "GSE274987", "patient_id", "tissue_type", celltype_malignant),
    ]
    for cohort, source, patient_column, region_column, select_cells in cohorts:
        cell_table = read_h5ad_malignant(
            files[source],
            cohort,
            patient_column,
            region_column,
            "dataset",
            cohort,
            select_cells,
            standardize=standardize,
            score_threshold=score_threshold,
        )
        parts.append(cell_table)
    cell_table_total = pd.concat(parts, ignore_index=True)
    return cell_table_total


def sc_malignant(obs_table: pd.DataFrame) -> pd.Series:
    """Select scGBM candidates for uniform annotation using Tumor in the original Major_category field."""
    selected = obs_table["Major_category"].astype(str).eq("Tumor")
    return selected


def celltype_malignant(obs_table: pd.DataFrame) -> pd.Series:
    """Rescore only the four originally annotated malignant states in the single-nucleus and regional objects."""
    selected = obs_table["celltype"].astype(str).isin(primary.malignant_states)
    return selected


def read_gse211376(standardize: bool = False, score_threshold: float = minimum_score_difference) -> pd.DataFrame:
    """Read raw GEO counts. Select the original four malignant states before scoring markers in those cells."""
    metadata_table, matrix, _ = primary.read_count_expression(selected_genes)
    keep = metadata_table["predicted.high_hierarchy"].isin(primary.malignant_states).to_numpy()
    selected_table = metadata_table.loc[keep, ["patient"]].copy()
    selected_table["region"] = "Tumor"
    cell_table = scored_cells(
        matrix[keep], selected_table, "GSE211376", "patient", "region", standardize, score_threshold
    )
    return cell_table


def read_h5ad_malignant(
    path: Path,
    cohort: str,
    patient_col: str,
    region_col: str,
    dataset_col: str,
    dataset_filter: str,
    malignant_mask,
    *,
    standardize: bool = False,
    score_threshold: float = minimum_score_difference,
) -> pd.DataFrame:
    """Select genes from X by dataset and malignant label. Retain original patient, region and TGFB2 expression."""
    source = ad.read_h5ad(path, backed="r")
    try:
        obs_table = source.obs
        keep = obs_table[dataset_col].astype(str).eq(dataset_filter).to_numpy()
        positions = np.flatnonzero(keep & np.asarray(malignant_mask(obs_table), dtype=bool))
        matrix = primary.selected_expression(source, positions, selected_genes)
        selected_table = obs_table.iloc[positions]
    finally:
        source.file.close()
    cell_table = scored_cells(matrix, selected_table, cohort, patient_col, region_col, standardize, score_threshold)
    return cell_table


def scored_cells(
    matrix: np.ndarray,
    obs_table: pd.DataFrame,
    cohort: str,
    patient_column: str,
    region_column: str,
    standardize: bool,
    score_threshold: float,
) -> pd.DataFrame:
    """Return one row per malignant cell. Keep TGFB2 on its original outcome scale and standardize only marker scores.
    """
    labels, confidence, ac_scores = assign_uniform_states(matrix, selected_genes, standardize, score_threshold)
    cell_table = pd.DataFrame(
        {
            "cohort": cohort,
            "patient": obs_table[patient_column].astype(str).to_numpy(),
            "region": obs_table[region_column].astype(str).to_numpy(),
            "uniform_state": labels,
            "confidence": confidence,
            "ac_marker_score": ac_scores,
            "TGFB2": matrix[:, selected_genes.index("TGFB2")],
        }
    )
    return cell_table


# Keep the four published marker sets and state order. TGFB2 is an outcome only. CD44 is both a MES marker and a
# primary-analysis receptor.
markers = {
    "AC-like": ["GFAP", "S100B", "HOPX", "SLC1A3", "MLC1"],
    "MES-like": ["VIM", "CD44", "CHI3L1", "HILPDA", "DDIT3", "ENO2"],
    "OPC-like": ["PLP1", "OLIG1", "OMG", "TNR", "ALCAM"],
    "NPC-like": ["SOX4", "SOX11", "DCX", "CD24", "STMN1", "STMN2"],
}
selected_genes = list(dict.fromkeys(["TGFB2", *chain.from_iterable(markers.values())]))


def assign_uniform_states(
    expression: np.ndarray,
    genes: list[str],
    standardize: bool = False,
    score_threshold: float = minimum_score_difference,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the top state, its margin over the runner-up and the AC score. Label states below the margin threshold
    Ambiguous.
    """
    score_expression = np.asarray(expression)
    if standardize:
        score_expression = score_expression.astype(float, copy=False)
        deviations = score_expression.std(axis=0)
        deviations[deviations == 0] = 1.0
        score_expression = (score_expression - score_expression.mean(axis=0)) / deviations

    # Each row is a cell and each column a state. Use the last two argsort positions to preserve the original handling
    # of ties.
    positions = {gene: i for i, gene in enumerate(genes)}
    program_scores = []
    for program in markers.values():
        columns = [positions[gene] for gene in program]
        program_scores.append(score_expression[:, columns].mean(axis=1))
    score_matrix = np.column_stack(program_scores)
    order = np.argsort(score_matrix, axis=1)
    rows = np.arange(len(score_matrix))
    confidence = score_matrix[rows, order[:, -1]] - score_matrix[rows, order[:, -2]]
    states = list(markers)
    labels = np.asarray(states, dtype=object)[order[:, -1]]
    labels[confidence < score_threshold] = "Ambiguous"
    result = labels, confidence, score_matrix[:, states.index("AC-like")]
    return result


def patient_contrasts(cell_table: pd.DataFrame, minimum_cells: int = primary.minimum_cells) -> pd.DataFrame:
    """Return one row per patient and cohort. Pool all three non-AC states before applying the cell-count threshold."""
    rows = []
    eligible = cell_table["uniform_state"] != "Ambiguous"
    core = cell_table["cohort"].eq("GSE274987")
    target = (core & cell_table["region"].eq("Core")) | (~core & cell_table["region"].eq("Tumor"))
    eligible_table = cell_table.loc[eligible & target]
    for (cohort, patient), patient_table in eligible_table.groupby(["cohort", "patient"], observed=True):
        ac_table = patient_table.loc[patient_table["uniform_state"] == "AC-like"]
        other_table = patient_table.loc[patient_table["uniform_state"].isin(["MES-like", "OPC-like", "NPC-like"])]
        if len(ac_table) < minimum_cells or len(other_table) < minimum_cells:
            continue

        # Subtract at the original float32 precision before converting to Python float for output. Standardized AC
        # scores may be float64.
        ac_mean = ac_table["TGFB2"].mean()
        other_mean = other_table["TGFB2"].mean()
        score_difference = ac_table["ac_marker_score"].mean() - other_table["ac_marker_score"].mean()
        rows.append(
            {
                "cohort": cohort,
                "patient": patient,
                "region": "Core" if cohort == "GSE274987" else "Tumor",
                "n_ac": len(ac_table),
                "n_other": len(other_table),
                "tgfb2_ac": float(ac_mean),
                "tgfb2_other": float(other_mean),
                "tgfb2_difference": float(ac_mean - other_mean),
                "ac_score_difference": float(score_difference),
            }
        )
    patient_table = pd.DataFrame.from_records(
        rows,
        columns=[
            "cohort",
            "patient",
            "region",
            "n_ac",
            "n_other",
            "tgfb2_ac",
            "tgfb2_other",
            "tgfb2_difference",
            "ac_score_difference",
        ],
    )
    return patient_table


def cohort_results(patient_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize patient contrasts by cohort, retain descriptive rows for fewer than three patients, and BH-adjust
    estimable P values.
    """
    rows = []
    for cohort, cohort_table in patient_table.groupby("cohort", observed=True):
        differences = cohort_table["tgfb2_difference"].to_numpy(float)
        rows.append(
            {
                "cohort": cohort,
                "n_patients": len(cohort_table),
                "median_difference": float(np.median(differences)),
                "positive_patients": int((differences > 0).sum()),
                "p_value": primary.safe_wilcoxon(differences),
            }
        )
    cohort_table_total = pd.DataFrame.from_records(
        rows, columns=["cohort", "n_patients", "median_difference", "positive_patients", "p_value"]
    )
    cohort_table_total["q_value"] = primary.bh_adjust(cohort_table_total["p_value"])
    return cohort_table_total


def meta_result(cohort_table: pd.DataFrame) -> dict:
    """Pool cohorts with a P value and at least three patients. Record the omitted cohort's name at each iteration."""
    valid_table = cohort_table.loc[cohort_table["p_value"].notna() & cohort_table["n_patients"].ge(3)]
    meta_z, meta_p = primary.combine_meta(valid_table)
    leave_one_out = []
    for omitted in valid_table["cohort"]:
        z_value, p_value = primary.combine_meta(valid_table.loc[valid_table["cohort"] != omitted])
        leave_one_out.append({"omitted": omitted, "z": z_value, "p_value": p_value})
    result = {
        "n_cohorts": len(valid_table),
        "n_patients": int(valid_table["n_patients"].sum()),
        "positive_cohorts": int((valid_table["median_difference"] > 0).sum()),
        "meta_z": meta_z,
        "p_value": meta_p,
        "loo_max_p_value": max((row["p_value"] for row in leave_one_out), default=np.nan),
        "loo_results": leave_one_out,
    }
    return result


def lineage_adjustment(patient_table: pd.DataFrame) -> dict:
    """Adjust for the AC score in cohorts with at least three patients. Return HC3 intervals and rank correlation."""
    import statsmodels.formula.api as smf

    inferential = patient_table.groupby("cohort")["patient"].transform("count") >= 3
    model_table = patient_table.loc[inferential].copy()
    model_table["ac_score_centered"] = model_table["ac_score_difference"] - model_table["ac_score_difference"].mean()

    # Sum coding makes the intercept an equal-weight average across cohorts. Center the predictor at the mean of the
    # included patients.
    model = smf.ols("tgfb2_difference ~ ac_score_centered + C(cohort, Sum)", data=model_table)
    fit = model.fit(cov_type="HC3")
    rho, rho_p = spearmanr(model_table["tgfb2_difference"], model_table["ac_score_difference"])
    intervals_table = fit.conf_int()
    lower, upper = intervals_table.loc["Intercept"]
    result = {
        "n_patients": len(model_table),
        "grand_mean_at_average_ac_score": float(fit.params["Intercept"]),
        "hc3_p_value": float(fit.pvalues["Intercept"]),
        "hc3_ci_low": float(lower),
        "hc3_ci_high": float(upper),
        "ac_score_slope": float(fit.params["ac_score_centered"]),
        "ac_score_slope_p": float(fit.pvalues["ac_score_centered"]),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "interpretation": "Sensitivity model only; uniform states are intentionally defined by the AC marker program.",
    }
    return result


if __name__ == "__main__":
    main()
