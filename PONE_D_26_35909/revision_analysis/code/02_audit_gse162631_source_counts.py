"""Check retained GSE162631 panel counts and full-library normalization against the original 10x matrices.
Inputs are the R*_T matrices, features and barcodes under GSE162631_RAW_ROOT and the outputs of 01a and 01b.
Outputs are tables 01 through 04 in 02_source_count_audit.
Run python code/02_audit_gse162631_source_counts.py from revision_analysis.
"""

import gzip
import os
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from anndata.utils import make_index_unique
from scipy.io import mmread

patient = import_module("00a_patient_expression")
source_root = Path(os.environ["GSE162631_RAW_ROOT"]).expanduser()
cache = patient.result_root() / "01_counts_and_preprocessing/09_cohort_cell_cache"
cell_table, cached_counts, _ = patient.read_cohort_cache(cache, "GSE162631")

# 10x matrices have genes in rows and barcodes in columns. Use sample feature names and retained cell barcodes.
audit_rows = []
contrast_rows = []
eligibility_rows = []
for folder in sorted(source_root.glob("R*_T")):
    features_path = folder / "features.tsv.gz"
    if not features_path.is_file():
        features_path = folder / "genes.tsv.gz"
    feature_table = pd.read_csv(features_path, sep="\t", header=None)
    genes = make_index_unique(pd.Index(feature_table.iloc[:, 1].astype(str)))
    barcode_table = pd.read_csv(folder / "barcodes.tsv.gz", header=None)
    barcodes = pd.Index(barcode_table.iloc[:, 0].astype(str))
    indices = genes.get_indexer(patient.genes)
    if np.any(indices < 0) or not barcodes.is_unique:
        raise ValueError(f"Missing panel genes or duplicate barcodes in {folder.name}")

    # Calculate library size and mitochondrial fraction from all genes. Subset panel rows only for direct comparison.
    with gzip.open(folder / "matrix.mtx.gz", "rb") as source:
        matrix = mmread(source).tocsc()
    valid_counts = (matrix.data >= 0) & (matrix.data == np.rint(matrix.data))
    if matrix.shape != (len(genes), len(barcodes)) or not valid_counts.all():
        raise ValueError(f"Invalid 10x matrix dimensions or counts in {folder.name}")
    library = np.asarray(matrix.sum(axis=0)).ravel()
    detected = np.asarray((matrix > 0).sum(axis=0)).ravel()
    mt_rows = genes.str.upper().str.startswith("MT-")
    mitochondrial = np.asarray(matrix[mt_rows].sum(axis=0)).ravel()

    # Barcodes can recur across samples. Select the patient and region before matching barcodes.
    subject = "GSE162631_" + folder.name.split("_")[0]
    tumor_cells = (cell_table["patient"] == subject) & (cell_table["region"] == "Tumor")
    subset_table = cell_table.loc[tumor_cells].copy()
    positions = subset_table.index.to_numpy()
    barcode_index = barcodes.get_indexer(subset_table["cell_id"].str.rsplit("_", n=1).str[-1])
    if subset_table.empty or np.any(barcode_index < 0):
        raise ValueError(f"Source barcodes differ from retained cell barcodes in {folder.name}")
    counts = matrix[indices, :][:, barcode_index].T.toarray().astype(np.float32)
    disagreement = np.abs(counts - cached_counts[positions]).max()
    if disagreement != 0:
        raise ValueError(f"Panel gene counts differ from cached counts in {folder.name}")

    # Reassign states using full-library totals. Also test the subset consistent with the original malignant labels.
    retained_library = subset_table["count_library"].to_numpy()
    subset_table["count_library"] = library[barcode_index]
    subset_table["count_n_genes"] = detected[barcode_index]
    subset_table["count_mt_pct"] = 100 * mitochondrial[barcode_index] / library[barcode_index]
    subset_table = subset_table.reset_index(drop=True)
    expression = np.log1p(counts / library[barcode_index, None] * 1e4).astype(np.float32)
    labels, _ = patient.assign_states(expression)
    malignant = subset_table["malignant_pool"].to_numpy()
    labels[~malignant] = "Excluded"
    original_tumor = subset_table["cell_type_original"].isin(["Tumor", "Prol.tumor"]).to_numpy()
    restricted = labels.copy()
    restricted[~original_tumor] = "Excluded"
    for scenario, selected in [
        ("source_full_gene_10k", labels),
        ("source_original_tumor_concordant", restricted),
    ]:
        contrasts, eligibility = patient.summarize_patients(
            subset_table, expression, counts, selected, "uniform", scenario
        )
        contrast_rows.extend(contrasts)
        eligibility_rows.extend(eligibility)

    ratio = library[barcode_index] / retained_library
    audit_rows.append(
        {
            "sample": folder.name,
            "source_cells": len(barcodes),
            "source_genes": len(genes),
            "retained_cells": len(subset_table),
            "retained_fraction": len(subset_table) / len(barcodes),
            "malignant_pool_cells": int(malignant.sum()),
            "max_panel_count_disagreement": float(disagreement),
            "median_full_over_retained_library_ratio": float(np.median(ratio)),
            "max_full_over_retained_library_ratio": float(np.max(ratio)),
            "n_original_tumor_concordant": int((original_tumor & malignant).sum()),
        }
    )

# Every patient with retained tumor cells must have a matching source file.
expected = set(cell_table.loc[cell_table["region"] == "Tumor", "patient"].unique())
observed = {"GSE162631_" + row["sample"].split("_")[0] for row in audit_rows}
if observed != expected:
    raise ValueError("Tumor samples and the retained cell table contain different patient sets")
patient.write_table("GSE162631_public_count_audit", pd.DataFrame(audit_rows))
patient.write_table("GSE162631_source_sensitivity", pd.DataFrame(contrast_rows))
patient.write_table("GSE162631_source_eligibility", pd.DataFrame(eligibility_rows))

# Omit one patient at a time from the baseline scenario and summarize the remaining contrasts.
contrast_table = patient.read_table("preprocessing_patient_contrasts")
baseline_table = contrast_table.query(
    "scenario == 'stored_X' and labels == 'uniform' and cohort == 'GSE162631'"
)
omission_rows = []
for subject in baseline_table["patient"]:
    remaining = baseline_table.loc[baseline_table["patient"] != subject, "difference"]
    omission_rows.append(
        {
            "omitted": subject,
            "n": len(remaining),
            "median_difference": remaining.median(),
            "mean_difference": remaining.mean(),
            "positive_patients": int((remaining > 0).sum()),
            "inference": f"descriptive n={len(remaining)}; no p value",
        }
    )
patient.write_table("GSE162631_leave_patient_out", pd.DataFrame(omission_rows))
print("Tumor samples checked:", len(audit_rows), "Leave-one-patient-out comparisons:", len(omission_rows))
