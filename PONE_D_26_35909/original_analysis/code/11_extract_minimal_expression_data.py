#!/usr/bin/env python3
# Purpose: Extract the cells, genes and metadata used by the primary and uniform-annotation analyses from the original
# public inputs.
# Inputs: GSE211376_raw_counts.tsv.gz, GSE211376_metadata.csv.gz and the three processed_inputs/*.h5ad files under
# PUBLIC_INPUT_ROOT.
# The published minimal_cell_data/minimal_cell_data_manifest.csv lists the historical cell counts, gene counts and obs
# columns.
# Outputs in minimal_cell_data/: GSE211376_minimal.h5ad, scGBM_minimal.h5ad, snGBM_minimal.h5ad,
# regional_GSE274987_minimal.h5ad and minimal_cell_data_manifest.csv. Extract patient IDs and cell names as recorded in
# the source files.
# The copies released with the paper underwent a separate recoding step.
# Usage: Run python 11_extract_minimal_expression_data.py from code/.
# Preserve AnnData's X, obs, var and uns and the original gzip storage. Standalone releases write to
# 01_cohort_reanalysis/.

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

primary = import_module("01_patient_expression")
uniform = import_module("02_uniform_annotation_sensitivity")
paths = import_module("00a_project_paths")


def main() -> None:
    """Verify every object before replacing the five output files. Keep other files in the directory. Returns None."""
    primary.stop_if_missing()
    files = paths.primary_input_files()
    output = paths.analysis_output_root() / "minimal_cell_data"
    output.mkdir(parents=True, exist_ok=True)

    # Stage the entire set on the same filesystem. If an object fails validation, existing objects and the manifest are
    # still intact.
    with TemporaryDirectory(dir=output) as scratch:
        staging = Path(scratch)
        rows = [write_gse211376(staging)]
        rows.append(
            write_minimal_h5ad(
                files["scGBM"],
                "scGBM_minimal.h5ad",
                ["dataset", "patient_id", "tissue_type", "Minor_cluster", "Major_category"],
                select_sc_cells,
                "GSE162631 malignant cells used by uniform-marker sensitivity "
                "and GSE273274 malignant or myeloid cells used by primary or uniform-marker analyses.",
                directory=staging,
            )
        )
        rows.append(
            write_minimal_h5ad(
                files["snGBM"],
                "snGBM_minimal.h5ad",
                ["dataset", "patient", "tissue", "celltype"],
                select_sn_cells,
                "GSE276177 and GSE286419 malignant or myeloid cells used by primary or uniform-marker analyses.",
                directory=staging,
            )
        )
        rows.append(
            write_minimal_h5ad(
                files["GSE274987"],
                "regional_GSE274987_minimal.h5ad",
                ["dataset", "patient_id", "tissue_type", "celltype", "Platform"],
                select_regional_cells,
                "Joint GSE274986/GSE274987 malignant or myeloid cells used by primary or uniform-marker analyses. "
                "The legacy dataset value is GSE274987; Platform maps Multiome to GSE274986 and ParseBio to GSE274987. "
                "All GSE241037 cells were excluded.",
                directory=staging,
            )
        )

        # Historical byte counts and digests describe files before recoding. Compare cell counts, gene counts and obs
        # columns to the public release.
        manifest_table = pd.DataFrame(rows)
        manifest_name = "minimal_cell_data_manifest.csv"
        reference = Path(__file__).resolve().parents[1] / "minimal_cell_data" / manifest_name
        reference_table = pd.read_csv(reference)
        columns = ["file", "n_cells", "n_genes", "obs_columns"]
        expected_table = reference_table[columns].sort_values("file").reset_index(drop=True)
        actual_table = manifest_table[columns].sort_values("file").reset_index(drop=True)
        pd.testing.assert_frame_equal(expected_table, actual_table)
        paths.write_table(manifest_table, staging / manifest_name, ["file"])

        for name in [*manifest_table["file"], manifest_name]:
            (staging / name).replace(output / name)
    print(f"Extracted {len(manifest_table)} objects with {manifest_table['n_cells'].sum()} cells and nuclei in total.")


def select_sc_cells(obs_table: pd.DataFrame) -> pd.Series:
    """Keep original Tumor cells for GSE162631. For GSE273274 take the union used by primary and uniform-marker
    analyses.
    """
    dataset = obs_table["dataset"].astype(str)
    tumor = uniform.sc_malignant(obs_table)
    annotated = obs_table["Minor_cluster"].astype(str).isin(primary.sc_state_labels)
    eligible = tumor | annotated | primary.sc_myeloid(obs_table)
    selected = (dataset.eq("GSE162631") & tumor) | (dataset.eq("GSE273274") & eligible)
    return selected


def select_sn_cells(obs_table: pd.DataFrame) -> pd.Series:
    """Keep only the four malignant states and three myeloid labels in the two single-nucleus cohorts."""
    dataset = obs_table["dataset"].astype(str).isin(["GSE276177", "GSE286419"])
    selected = dataset & (uniform.celltype_malignant(obs_table) | primary.celltype_myeloid(obs_table))
    return selected


def select_regional_cells(obs_table: pd.DataFrame) -> pd.Series:
    """Select the joint regional cohort by its legacy dataset value. Retain both Platform values and exclude mouse
    cells.
    """
    dataset = obs_table["dataset"].astype(str).eq("GSE274987")
    selected = dataset & (uniform.celltype_malignant(obs_table) | primary.celltype_myeloid(obs_table))
    return selected


def write_gse211376(directory: Path | None = None) -> dict:
    """Return a manifest row after extracting normalized target expression and pre-normalization cell totals from
    all-gene counts.
    """
    metadata_table, expression, totals = primary.read_count_expression(target_genes)
    selected = metadata_table["predicted.high_hierarchy"].isin(primary.malignant_states) | primary.tam_myeloid(
        metadata_table
    )
    obs_table = metadata_table.loc[selected, ["predicted.high_hierarchy", "patient"]].astype(str).copy()
    obs_table["dataset"] = "GSE211376"
    obs_table["region"] = "Tumor"
    obs_table["total_counts"] = totals[selected]
    files = paths.primary_input_files()
    sources = f"{files['GSE211376_counts'].name};{files['GSE211376_meta'].name}"

    annotations = {
        "data_role": (
            "Minimal normalized cell-level expression and metadata supporting the reported patient-level analyses."
        ),
        "normalization": "Library-size normalization to 10,000 followed by log1p.",
        "source_files": sources,
    }
    target = (directory or paths.analysis_output_root() / "minimal_cell_data") / "GSE211376_minimal.h5ad"

    # The old stage numbers are part of the published cell_scope field in the original manifest.
    record = save_minimal(
        target, expression[selected], obs_table, annotations, sources,
        "Malignant states and TAM-MG/TAM-BDM cells used by stage45 or stage48/51."
    )
    return record


# Keep primary ligands and receptors first, then append previously unseen uniform markers in their published order.
target_genes = list(dict.fromkeys([*primary.primary_genes, *uniform.selected_genes]))


def write_minimal_h5ad(
    source: Path, target_name: str, obs_columns: list[str], cell_mask, scope: str, *, directory: Path | None = None
) -> dict:
    """Select source cells and columns and preserve expression values. Require all genes and metadata columns. Return
    one manifest row.
    """
    source_object = ad.read_h5ad(source, backed="r")
    try:
        absent_genes = sorted(set(target_genes) - set(source_object.var_names))
        absent_columns = sorted(set(obs_columns) - set(source_object.obs.columns))
        if absent_genes or absent_columns:
            raise ValueError(f"{source.name}: missing genes {absent_genes} or metadata columns {absent_columns}")

        keep = np.asarray(cell_mask(source_object.obs), dtype=bool)
        obs_table = source_object.obs.loc[keep, obs_columns].astype(str).copy()
        obs_table.index = source_object.obs_names[keep].astype(str)
        matrix = source_object[keep, target_genes].X
    finally:
        source_object.file.close()

    annotations = {
        "data_role": "Minimal cell-level expression and metadata supporting the reported patient-level analyses; "
        "not a replacement for the public raw archive.",
        "cell_scope": scope,
        "source_file": source.name,
        "removed_components": (
            "Unused genes, layers, raw matrix, embeddings, graphs, and unneeded observation metadata."
        ),
    }
    target = (directory or paths.analysis_output_root() / "minimal_cell_data") / target_name
    record = save_minimal(target, matrix, obs_table, annotations, source.name, scope)
    return record


def save_minimal(
    target: Path, matrix, obs_table: pd.DataFrame, annotations: dict, source_name: str, scope: str
) -> dict:
    """Save float32 CSR expression. Reopen the object and verify ordering, obs, uns and expression before recording its
    provenance.
    """
    expression = normalize_matrix(matrix)
    if expression.shape != (len(obs_table), len(target_genes)):
        raise ValueError(f"{target.name}: expression matrix dimensions do not match the cell and gene counts")
    result = ad.AnnData(X=expression, obs=obs_table, var=pd.DataFrame(index=pd.Index(target_genes, name="gene")))
    result.uns.update(annotations)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(target, compression="gzip")

    # The minimal matrix fits in memory. An elementwise sparse comparison detects shifted values or incomplete writes.
    saved = ad.read_h5ad(target)
    if saved.var_names.tolist() != target_genes or dict(saved.uns) != annotations:
        raise ValueError(f"{target.name}: gene order or provenance fields changed on disk")
    pd.testing.assert_frame_equal(obs_table, saved.obs, check_dtype=False, check_categorical=False)
    if saved.shape != expression.shape or saved.X.dtype != np.float32 or (saved.X != expression).nnz:
        raise ValueError(f"{target.name}: expression dimensions, precision or values changed on disk")
    record = {
        "file": target.name,
        "source": source_name,
        "n_cells": saved.n_obs,
        "n_genes": saved.n_vars,
        "obs_columns": ";".join(saved.obs.columns),
        "cell_scope": scope,
        "bytes": target.stat().st_size,
        "sha256": paths.file_sha256(target),
    }
    return record


def normalize_matrix(matrix) -> sparse.csr_matrix:
    """Convert to CSR and remove explicit zeros. Each input already has its source-specific expression preprocessing."""
    if sparse.issparse(matrix):
        expression = matrix.tocsr().astype(np.float32)
    else:
        expression = sparse.csr_matrix(np.asarray(matrix, dtype=np.float32))
    expression.eliminate_zeros()
    return expression


if __name__ == "__main__":
    main()
