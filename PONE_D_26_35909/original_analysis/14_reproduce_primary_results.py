#!/usr/bin/env python3
# Purpose: Rebuild eight primary patient-level tables from the public recoded minimal expression objects and compare
# them with the reference results in data/.
# Inputs: minimal_cell_data/minimal_cell_data_manifest.csv and its four .h5ad files (X, patient, region, original state
# annotations).
# The eight reference CSVs are in data/. The result_names mapping in main pairs reference and output filenames.
# Outputs in 14_primary_results/: 01_patient_state_expression.csv, 02_patient_state_composition.csv,
# 03_patient_ligand_contrasts.csv,
# 04_pooled_ligand_results.csv, 05_patient_ligand_receptor_coavailability.csv, 06_regional_coavailability_tests.csv,
# 07_patient_ac_like_fraction.csv, 08_regional_ac_like_fraction_tests.csv and 09_reproduction_checks.json.
# Usage: Run python 14_reproduce_primary_results.py from original_analysis/. Inputs are relative to the script.

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import numpy as np
import pandas as pd


def main() -> None:
    """Compare columns, patient identifiers, values and nested records to the release. Record the result for each table.
    Returns None.
    """
    package = Path(__file__).resolve().parent
    code_directory = str(package / "code")
    if code_directory not in sys.path:
        sys.path.insert(0, code_directory)
    primary = import_module("01_patient_expression")
    paths = import_module("00a_project_paths")

    # Two cohorts share the same minimal snGBM file. The manifest records sizes before recoding, so compare dimensions
    # and columns.
    directory = package / "minimal_cell_data"
    manifest_table = pd.read_csv(directory / "minimal_cell_data_manifest.csv")
    expected_files = {
        "GSE211376_minimal.h5ad",
        "scGBM_minimal.h5ad",
        "snGBM_minimal.h5ad",
        "regional_GSE274987_minimal.h5ad",
    }
    if manifest_table["file"].duplicated().any() or set(manifest_table["file"]) != expected_files:
        raise ValueError("The manifest filenames do not match the four published minimal inputs")
    for row in manifest_table.itertuples(index=False):
        source = ad.read_h5ad(directory / row.file, backed="r")
        try:
            if source.shape != (row.n_cells, row.n_genes) or list(source.obs.columns) != row.obs_columns.split(";"):
                raise ValueError(f"{row.file}: dimensions or metadata columns differ from the manifest")
            if not source.obs_names.is_unique or not set(primary.primary_genes).issubset(source.var_names):
                raise ValueError(f"{row.file}: duplicate cell names or missing primary analysis genes")
        finally:
            source.file.close()

    expression_table_total, composition_table_total = primary.read_primary_cohorts(directory)
    tables = primary.primary_tables(expression_table_total, composition_table_total)
    result_names = {
        "patient_state_expression": "01_patient_state_expression.csv",
        "patient_state_composition": "02_patient_state_composition.csv",
        "ac_vs_other_ligand_contrasts": "03_patient_ligand_contrasts.csv",
        "meta_ligand_results": "04_pooled_ligand_results.csv",
        "patient_communication_scores": "05_patient_ligand_receptor_coavailability.csv",
        "regional_communication_tests": "06_regional_coavailability_tests.csv",
        "patient_ac_fraction": "07_patient_ac_like_fraction.csv",
        "regional_ac_fraction_tests": "08_regional_ac_like_fraction_tests.csv",
    }
    if tables.keys() != result_names.keys():
        raise ValueError("The reconstructed set of tables differs from the eight published references")

    # Read each table from disk before comparing it to its published reference. Record the result and mismatched fields
    # for review.
    output = package / "14_primary_results"
    output.mkdir(parents=True, exist_ok=True)
    checks = {}
    differences = []
    with TemporaryDirectory(dir=output) as scratch:
        staging = Path(scratch)
        for name, filename in result_names.items():
            keys = primary.primary_row_columns[name]
            paths.write_table(tables[name], staging / filename, keys)
            actual_table = pd.read_csv(staging / filename, float_precision="round_trip")
            reference_table = pd.read_csv(package / "data" / f"{name}.csv", float_precision="round_trip")
            try:
                compare_table(actual_table, reference_table, keys, name)
                checks[name] = True
            except (AssertionError, ValueError, KeyError, TypeError) as error:
                checks[name] = False
                differences.append(f"{name}: {error}")

        paths.write_json(checks, staging / "09_reproduction_checks.json")
        for filename in [*result_names.values(), "09_reproduction_checks.json"]:
            (staging / filename).replace(output / filename)
    print(f"Primary patient table comparisons: {sum(checks.values())} passed, {len(differences)} failed.")
    if differences:
        raise ValueError("\n".join(differences))


def compare_table(actual_table: pd.DataFrame, reference_table: pd.DataFrame, keys: list[str], name: str) -> None:
    """Align by the actual row keys and compare every published column. Duplicate keys or differences in schema, shape
    or values raise an error. Returns None.
    """
    if list(actual_table.columns) != list(reference_table.columns):
        raise ValueError(f"{name}: column names or order changed")
    actual_table = actual_table.set_index(keys, verify_integrity=True).sort_index()
    reference_table = reference_table.set_index(keys, verify_integrity=True).sort_index()
    pd.testing.assert_index_equal(actual_table.index, reference_table.index, obj=f"{name} row keys")

    # Keep the original floating-point tolerances (rtol=1e-5, atol=1e-7). Compare integers, booleans, text and missing
    # positions separately.
    nested_columns = {"patient_values", "loo_results", "cohort_effects", "patient_differences"}
    for column in reference_table:
        if column in nested_columns:
            for row_id in reference_table.index:
                actual = json.loads(actual_table.at[row_id, column])
                expected = json.loads(reference_table.at[row_id, column])
                compare_records(actual, expected, column, f"{name}/{row_id}/{column}")
            continue
        compare_column(actual_table[column], reference_table[column], f"{name}/{column}")


def compare_records(actual: list, expected: list, column: str, name: str) -> None:
    """Match identified patient or cohort records by name. Compare unlabelled regional differences in published order.
    Returns None.
    """
    if len(actual) != len(expected):
        raise ValueError(f"{name}: different numbers of nested records")
    if column == "patient_differences":
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7, equal_nan=True, err_msg=name)
    elif expected:
        identifier = {"patient_values": "patient", "loo_results": "omitted", "cohort_effects": "cohort"}[column]
        expected_fields = {row[identifier]: set(row) for row in expected}
        actual_fields = {row[identifier]: set(row) for row in actual}
        if actual_fields != expected_fields:
            raise ValueError(f"{name}: mismatched identifiers or fields within individual records")
        actual_table = pd.DataFrame(actual).set_index(identifier, verify_integrity=True).sort_index()
        expected_table = pd.DataFrame(expected).set_index(identifier, verify_integrity=True).sort_index()
        pd.testing.assert_index_equal(actual_table.index, expected_table.index, obj=name)
        if set(actual_table.columns) != set(expected_table.columns):
            raise ValueError(f"{name}: mismatched nested fields")
        for field in expected_table:
            compare_column(actual_table[field], expected_table[field], f"{name}/{field}")


def compare_column(actual: pd.Series, expected: pd.Series, name: str) -> None:
    """Use the reference field's type for comparison so numeric conversion cannot obscure missing values, count
    differences or text changes. Returns None.
    """
    if pd.api.types.is_float_dtype(expected):
        np.testing.assert_allclose(
            actual.to_numpy(float), expected.to_numpy(float), rtol=1e-5, atol=1e-7, equal_nan=True, err_msg=name
        )
    else:
        pd.testing.assert_series_equal(actual, expected, check_exact=True, obj=name)


if __name__ == "__main__":
    main()
