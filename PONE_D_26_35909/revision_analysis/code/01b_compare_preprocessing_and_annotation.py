"""Compare patient-level TGFB2 contrasts under alternative normalization, filtering and annotation choices.
Input is the cohort cell cache from stage 01a, with aligned annotations, counts and stored expression.
Outputs are tables 02 through 08 and 10_python_versions.json in 01_counts_and_preprocessing.
Run python code/01b_compare_preprocessing_and_annotation.py from revision_analysis. SciPy provides signed-rank tests.
"""

import json
from importlib import import_module, metadata

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon

patient = import_module("00a_patient_expression")
cache = patient.result_root() / "01_counts_and_preprocessing/09_cohort_cell_cache"

# Each patient row belongs to one cohort, scenario and annotation method. The state summaries cover GSE162631.
contrast_rows = []
eligibility_rows = []
state_rows = []
crosswalk_tables = []
membership_rows = []
catalog_table = pd.read_csv(cache / "03_cohort_cell_manifest.csv")
source_table = patient.read_table("count_input_audit").set_index("object", verify_integrity=True)
totals = catalog_table.groupby("object")["n_cells"].sum().sort_index()
if catalog_table["cohort"].duplicated().any() or not totals.equals(source_table["n_cells"].sort_index()):
    raise ValueError("The manifest cell count differs from the full extraction total")
for entry in catalog_table.sort_values("cohort").itertuples():
    cohort = entry.cohort
    cell_table, counts, stored = patient.read_cohort_cache(cache, cohort)
    if len(cell_table) != entry.n_cells:
        raise ValueError(f"Cached cell count differs from the current extraction manifest for {cohort}")

    # Scale whole-gene library totals to 10,000 or 1,000. Strict QC requires at least 500 genes and at most 5% mtRNA.
    denominator = cell_table["count_library"].to_numpy()[:, None]
    expression_10k = np.log1p(counts / denominator * 1e4).astype(np.float32)
    expression_1k = np.log1p(counts / denominator * 1e3).astype(np.float32)
    all_cells = np.ones(len(cell_table), dtype=bool)
    strict = (cell_table["count_n_genes"] >= 500) & (cell_table["count_mt_pct"] <= 5)
    malignant = cell_table["malignant_pool"].to_numpy()
    baseline_labels, _ = patient.assign_states(stored)
    baseline_labels[~malignant] = "Excluded"
    settings = [
        ("stored_X", stored, all_cells),
        ("counts_10k", expression_10k, all_cells),
        ("counts_1k", expression_1k, all_cells),
        ("counts_10k_strictQC", expression_10k, strict.to_numpy()),
        ("pseudobulk", expression_10k, all_cells),
    ]
    for scenario, expression, keep in settings:
        labels, margin = patient.assign_states(expression)
        labels[~malignant | ~keep] = "Excluded"
        primary = cell_table["primary_state"].to_numpy().copy()
        primary[~keep] = "Excluded"
        frozen = baseline_labels.copy()
        frozen[~keep] = "Excluded"
        for method, selected_labels in [
            ("primary", primary),
            ("uniform", labels),
            ("uniform_frozen", frozen),
        ]:
            if method == "primary" and cohort == "GSE162631":
                continue
            contrasts, eligibility = patient.summarize_patients(
                cell_table, expression, counts, selected_labels, method, scenario, scenario == "pseudobulk"
            )
            contrast_rows.extend(contrasts)
            eligibility_rows.extend(eligibility)

        # Count cells separately in each scenario. Patient-level lineage summaries use the candidate malignant cells.
        membership_rows.append(
            {
                "cohort": cohort,
                "scenario": scenario,
                "retained_total": int(keep.sum()),
                "malignant_pool_retained": int((keep & malignant).sum()),
                "uniform_AC": int((labels == "AC-like").sum()),
                "uniform_ambiguous": int((labels == "Ambiguous").sum()),
            }
        )
        if cohort != "GSE162631":
            continue
        state_table = cell_table.assign(uniform_state=labels, margin=margin)
        groups = state_table.loc[malignant].groupby(["patient", "region", "uniform_state"])
        for (subject, region, state), group_table in groups:
            positions = group_table.index.to_numpy()
            index = patient.genes.index("TGFB2")
            row = {
                "scenario": scenario,
                "patient": subject,
                "region": region,
                "state": state,
                "n_cells": len(group_table),
                "tgfb2_detected": float((counts[positions, index] > 0).mean()),
                "tgfb2_mean_log10k": float(expression_10k[positions, index].mean()),
                "median_library": float(group_table["count_library"].median()),
                "median_genes": float(group_table["count_n_genes"].median()),
                "median_mt_pct": float(group_table["count_mt_pct"].median()),
                "median_assignment_margin": float(group_table["margin"].median()),
            }
            for lineage, markers in patient.lineage_markers.items():
                columns = [patient.genes.index(gene) for gene in markers]
                row[lineage + "_score"] = float(expression_10k[positions][:, columns].mean())
            state_rows.append(row)

        # Count cells by patient, region and annotation pair in the baseline scenario.
        if scenario != "stored_X":
            continue
        for field in [
            "Major_category",
            "Minor_cluster",
            "cell_type_original",
            "cell_type_cnv_corrected",
            "cnv_score",
            "cnv_leiden",
        ]:
            if field in state_table:
                group_columns = ["patient", "region", field, "uniform_state"]
                mapping_table = (
                    state_table.groupby(group_columns, dropna=False).size().reset_index(name="n_cells")
                )
                mapping_table = mapping_table.rename(columns={field: "original_value"})
                mapping_table["original_field"] = field
                crosswalk_tables.append(mapping_table)

if not contrast_rows or not crosswalk_tables:
    raise ValueError("No cells available for patient contrasts or the GSE162631 annotation audit. Run stage 01a.")
contrast_table_total = pd.DataFrame(contrast_rows)
patient.write_table("preprocessing_patient_contrasts", contrast_table_total)
patient.write_table("preprocessing_eligibility", pd.DataFrame(eligibility_rows))
patient.write_table("preprocessing_cell_membership", pd.DataFrame(membership_rows))
patient.write_table("GSE162631_state_audit", pd.DataFrame(state_rows))
patient.write_table("GSE162631_annotation_crosswalk", pd.concat(crosswalk_tables, ignore_index=True))


def summarize_cohorts(patient_table):
    """Return pooled and cohort-level estimates using signed-rank tests in cohorts with at least three patients."""
    cohort_rows = []
    for cohort, group_table in patient_table.groupby("cohort"):
        values = group_table["difference"].to_numpy()
        if len(values) < 3:
            continue
        probability = 1.0 if np.all(values == 0) else float(wilcoxon(values, method="auto").pvalue)
        cohort_rows.append(
            {
                "cohort": cohort,
                "n": len(values),
                "median": float(np.median(values)),
                "positive_patients": int((values > 0).sum()),
                "p": probability,
            }
        )
    pooled = None

    def combine_signed_ranks(cohort_rows):
        """Return a pooled Z and two-sided P, using cohort median signs and square-root patient-count weights."""
        terms = []
        for row in cohort_rows:
            quantile = norm.isf(max(row["p"], 1e-15) / 2)
            terms.append(np.sqrt(row["n"]) * np.sign(row["median"]) * quantile)
        score = sum(terms) / np.sqrt(sum(row["n"] for row in cohort_rows))
        result = float(score), float(2 * norm.sf(abs(score)))
        return result

    if len(cohort_rows) >= 2:
        score, probability = combine_signed_ranks(cohort_rows)
        leave_out = []
        for omitted in cohort_rows:
            remaining = [row for row in cohort_rows if row["cohort"] != omitted["cohort"]]
            _, omitted_probability = combine_signed_ranks(remaining)
            leave_out.append(omitted_probability)
        pooled = {
            "n": sum(row["n"] for row in cohort_rows),
            "k": len(cohort_rows),
            "positive_cohorts": sum(row["median"] > 0 for row in cohort_rows),
            "meta_z": score,
            "p_value": probability,
            "loo_max_p": max(leave_out),
        }
    result = pooled, cohort_rows
    return result


# Pool all eligible patients, then pool the patients shared with baseline using cohort and patient as the key.
meta_rows = []
cohort_rows = []
for method in ["primary", "uniform", "uniform_frozen"]:
    method_table = contrast_table_total.loc[contrast_table_total["labels"] == method]
    baseline_table = method_table.loc[method_table["scenario"] == "stored_X", ["cohort", "patient"]]
    for scenario, scenario_table in method_table.groupby("scenario"):
        intersection_table = scenario_table.merge(
            baseline_table, on=["cohort", "patient"], validate="one_to_one"
        )
        for population, analysis_table in [
            ("available", scenario_table),
            ("baseline_intersection", intersection_table),
        ]:
            pooled, cohort_statistics = summarize_cohorts(analysis_table)
            identity = {"labels": method, "scenario": scenario, "population": population}
            if pooled is not None:
                meta_rows.append({**identity, **pooled})
            cohort_rows.extend({**identity, **row} for row in cohort_statistics)

patient.write_table("preprocessing_meta_stouffer", pd.DataFrame(meta_rows))
patient.write_table("preprocessing_cohort_summary", pd.DataFrame(cohort_rows))
versions = {
    name: metadata.version(name) for name in ["numpy", "pandas", "scipy", "anndata", "h5py", "matplotlib"]
}
version_path = patient.result_root() / "01_counts_and_preprocessing/10_python_versions.json"
version_path.write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
if json.loads(version_path.read_text(encoding="utf-8")) != versions:
    raise ValueError("The written software version record is incomplete")
print("Patient contrasts:", len(contrast_table_total), "Pooled tests:", len(meta_rows))
