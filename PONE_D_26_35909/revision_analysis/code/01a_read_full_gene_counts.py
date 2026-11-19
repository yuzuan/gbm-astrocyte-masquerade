"""Extract TGFB2 and marker counts from full-gene libraries and check stored normalized expression.
Inputs under GLIOMA_VALIDATION_ROOT/public_inputs are three processed H5ADs and the GSE211376 counts and metadata.
Outputs are 01_counts_and_preprocessing/01_count_input_audit.csv and per-cohort cell and matrix files in the cache.
Run python code/01a_read_full_gene_counts.py from revision_analysis. AnnData and SciPy read the sparse layers.
"""

import gzip
import os
from importlib import import_module
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem
from scipy import sparse

patient = import_module("00a_patient_expression")
source_root = (
    Path(os.environ.get("GLIOMA_VALIDATION_ROOT", Path(__file__).resolve().parents[1])) / "public_inputs"
)
cache = patient.result_root() / "01_counts_and_preprocessing/09_cohort_cell_cache"
cache.mkdir(parents=True, exist_ok=True)
audit_rows = []
cache_rows = []


def csr_block(group, start, stop):
    """Read a contiguous range of cell rows using the AnnData CSR indptr array."""
    if group.attrs["encoding-type"] != "csr_matrix":
        raise ValueError("Expected an AnnData CSR matrix")
    pointers = group["indptr"][start : stop + 1]
    left, right = int(pointers[0]), int(pointers[-1])
    values = group["data"][left:right]
    columns = group["indices"][left:right]
    matrix = sparse.csr_matrix(
        (values, columns, pointers - left), shape=(stop - start, int(group.attrs["shape"][1]))
    )
    return matrix


# Process one source at a time. Library totals use all retained genes. only the panel is cached for scoring.
for filename in (
    "scGBM_GSE162631_GSE273274.h5ad",
    "snGBM_GSE276177_GSE286419.h5ad",
    "regional_GSE274987_GSE241037.h5ad",
    "GSE211376",
):
    if filename != "GSE211376":
        with h5py.File(source_root / "processed_inputs" / filename, "r") as source:
            cell_table = read_elem(source["obs"])
            gene_table = read_elem(source["var"])
            columns = gene_table.index.get_indexer(patient.genes)
            if not cell_table.index.is_unique or not gene_table.index.is_unique or np.any(columns < 0):
                raise ValueError(f"Duplicate cell or gene identifiers, or missing panel genes in {filename}")

            # Cells occupy H5AD rows. Compute library size, detected genes and mitochondrial fraction in blocks of rows.
            mitochondrial = gene_table.index.str.upper().str.startswith("MT-")
            n_genes = len(gene_table)
            selected = np.zeros((len(cell_table), len(columns)), dtype=np.float32)
            stored = np.zeros_like(selected)
            library = np.zeros(len(cell_table))
            detected = np.zeros(len(cell_table), dtype=int)
            mt_percent = np.zeros(len(cell_table))
            for start in range(0, len(cell_table), 2048):
                end = min(start + 2048, len(cell_table))
                block = csr_block(source["layers/counts"], start, end)
                valid = np.isfinite(block.data) & (block.data >= 0) & (block.data == np.rint(block.data))
                if not valid.all():
                    raise ValueError(f"Expected finite, nonnegative integer counts in {filename}")
                library[start:end] = np.asarray(block.sum(axis=1)).ravel()
                detected[start:end] = np.asarray((block > 0).sum(axis=1)).ravel()
                mt_percent[start:end] = (
                    100 * np.asarray(block[:, mitochondrial].sum(axis=1)).ravel() / library[start:end]
                )
                selected[start:end] = block[:, columns].toarray()
                stored[start:end] = csr_block(source["X"], start, end)[:, columns].toarray()

        # scGBM stores its state labels in Minor_cluster. The other objects provide states directly. Retain human
        # cohorts.
        cell_table["cell_id"] = cell_table.index.astype(str)
        cell_table["cohort"] = cell_table["dataset"].astype(str)
        cell_table["patient"] = cell_table["patient_id" if "patient_id" in cell_table else "patient"].astype(
            str
        )
        cell_table["region"] = cell_table["tissue_type" if "tissue_type" in cell_table else "tissue"].astype(
            str
        )
        if filename.startswith("scGBM"):
            names = {
                "AC": "AC-like",
                "MES1": "MES-like",
                "OPC": "OPC-like",
                "NPC1": "NPC-like",
                "NPC2": "NPC-like",
            }
            cell_table["primary_state"] = (
                cell_table["Minor_cluster"].astype(str).map(names).fillna("Excluded")
            )
            cell_table["malignant_pool"] = cell_table["Major_category"].astype(str).eq("Tumor")
        else:
            malignant = cell_table["celltype"].isin(patient.states)
            cell_table["primary_state"] = cell_table["celltype"].astype(str).where(malignant, "Excluded")
            cell_table["malignant_pool"] = malignant
        keep = cell_table["cohort"].ne("GSE241037").to_numpy()
        scope = "retained cells and genes in complete project AnnData"
        source_name = filename

    else:

        # GEO stores genes in rows and cells in the header. Accumulate library statistics without densifying the matrix.
        metadata_table = pd.read_csv(source_root / "raw_inputs/GSE211376_metadata.csv.gz", index_col=0)
        with gzip.open(source_root / "raw_inputs/GSE211376_raw_counts.tsv.gz", "rt") as source:
            header = source.readline().rstrip("\n").split("\t")
            cells = header[1:] if header[0] == "" else header
            selected = np.zeros((len(cells), len(patient.genes)), dtype=np.float32)
            library = np.zeros(len(cells))
            detected = np.zeros(len(cells), dtype=int)
            mt_counts = np.zeros(len(cells))
            found = set()
            n_genes = 0
            for line in source:
                gene, values = line.rstrip("\n").split("\t", 1)
                counts = np.fromstring(values, sep="\t", dtype=np.float64)
                valid = np.isfinite(counts) & (counts == np.rint(counts)) & (counts >= 0)
                if len(counts) != len(cells) or not valid.all():
                    raise ValueError(f"Incomplete or invalid GSE211376 count row for {gene}")
                library += counts
                detected += counts > 0
                n_genes += 1
                if gene.upper().startswith("MT-"):
                    mt_counts += counts
                if gene in found:
                    raise ValueError(f"Duplicate GSE211376 panel gene: {gene}")
                if gene in patient.genes:
                    selected[:, patient.genes.index(gene)] = counts
                    found.add(gene)

        # Align patient annotations to the cell order in the count table. GEO supplies no separate normalized X matrix.
        cell_table = metadata_table.reindex(cells).copy()
        if cell_table["patient"].isna().any() or found != set(patient.genes):
            raise ValueError("Missing GSE211376 patient identifiers or panel genes")
        cell_table["cell_id"] = cells
        cell_table["cohort"] = "GSE211376"
        cell_table["region"] = "Tumor"
        original_labels = cell_table["predicted.high_hierarchy"]
        cell_table["primary_state"] = original_labels.where(original_labels.isin(patient.states), "Excluded")
        cell_table["malignant_pool"] = cell_table["primary_state"].isin(patient.states)
        mt_percent = 100 * mt_counts / library
        stored = np.log1p(selected / library[:, None] * 1e4).astype(np.float32)
        keep = np.ones(len(cell_table), dtype=bool)
        scope = "public count matrix"
        source_name = "GSE211376 GEO counts"

    # Split each validated input by GEO accession. Write cell metadata in matrix row order.
    if np.any(library <= 0):
        raise ValueError(f"Cells with zero library size in {source_name}")
    cell_table["count_library"] = library
    cell_table["count_n_genes"] = detected
    cell_table["count_mt_pct"] = mt_percent
    cell_table = cell_table.loc[keep].reset_index(drop=True)
    selected = selected[keep]
    stored = stored[keep]
    rebuilt = np.log1p(selected / library[keep, None] * 1e4)
    difference = np.abs(rebuilt - stored)

    # The published GEO audit uses zero for unavailable X. Stage 05a checks reconstruction only for the three H5ADs.
    audit_rows.append(
        {
            "object": source_name,
            "n_cells": len(cell_table),
            "n_full_genes": n_genes,
            "nonnegative_integer_counts": True,
            "max_abs_X_reconstruction_error": float(difference.max()) if filename != "GSE211376" else 0.0,
            "mean_abs_X_reconstruction_error": float(difference.mean()) if filename != "GSE211376" else 0.0,
            "input_scope": scope,
        }
    )
    for cohort, cohort_table in cell_table.groupby("cohort", sort=True):
        positions = cohort_table.index.to_numpy()
        cell_ids = cohort_table["cell_id"].to_numpy(dtype=str)
        cohort_table.to_csv(cache / f"01_{cohort}_cells.csv.gz", index=False)
        matrix_path = cache / f"02_{cohort}_expression.npz"
        np.savez_compressed(
            matrix_path, counts=selected[positions], stored=stored[positions], genes=patient.genes, cell_ids=cell_ids
        )
        saved_table, saved_counts, saved_expression = patient.read_cohort_cache(cache, cohort)
        if not np.array_equal(saved_table["cell_id"].to_numpy(), cell_ids):
            raise ValueError(f"Cached cell order differs from the source in {cohort}")
        counts_match = np.array_equal(saved_counts, selected[positions])
        if not counts_match or not np.array_equal(saved_expression, stored[positions]):
            raise ValueError(f"Cached counts or expression differ from the source in {cohort}")
        cache_rows.append({"object": source_name, "cohort": cohort, "n_cells": len(saved_table)})

# Stage 01b uses the manifest from this run to identify the complete set of extracted cohorts.
cache_table = pd.DataFrame(cache_rows)
if cache_table["cohort"].duplicated().any():
    raise ValueError("A cohort occurs in multiple source objects")
catalog = cache / "03_cohort_cell_manifest.csv"
cache_table.to_csv(catalog, index=False)
pd.testing.assert_frame_equal(pd.read_csv(catalog), cache_table)

audit_table = pd.DataFrame(audit_rows)
patient.write_table("count_input_audit", audit_table)
print("Input sources:", len(audit_table), "Retained cells:", audit_table["n_cells"].sum())
