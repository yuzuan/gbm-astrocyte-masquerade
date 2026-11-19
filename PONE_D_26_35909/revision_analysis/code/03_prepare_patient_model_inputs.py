"""Prepare patient contrasts for REML models of annotation, preprocessing and source-count sensitivity.
Inputs are tables S3a and S7a with original study codes and the patient contrasts generated in stages 01b and 02.
Output 03_model_inputs/01_model_patient_contrasts.csv has the columns analysis, cohort, patient and difference.
Set GLIOMA_VALIDATION_ROOT and PLOS_REVISION_DIR, then run python code/03_prepare_patient_model_inputs.py.
"""

import json
import os
from importlib import import_module
from pathlib import Path

import pandas as pd

patient = import_module("00a_patient_expression")
tables = Path(os.environ["GLIOMA_VALIDATION_ROOT"]) / "claude_outputs/public_only_submission/tables"

# S3a nests patient contrasts within cohorts. S7a has one row per patient. Extract the same target and keys.
primary_table = pd.read_csv(tables / "TableS3a_ligand_contrasts.csv")
rows = []
for row in primary_table.loc[primary_table["gene"] == "TGFB2"].itertuples():
    for value in json.loads(row.patient_values):
        rows.append(
            {
                "analysis": "primary_labels",
                "cohort": row.cohort,
                "patient": value["patient"],
                "difference": value["difference"],
            }
        )
uniform_table = pd.read_csv(tables / "TableS7a_uniform_annotation_patient_contrasts.csv")
for row in uniform_table.itertuples():
    rows.append(
        {
            "analysis": "uniform_labels",
            "cohort": row.cohort,
            "patient": row.patient,
            "difference": row.tgfb2_difference,
        }
    )
baseline_table_total = pd.DataFrame(rows)
parts = [baseline_table_total]

# Keep each preprocessing and label combination. For source-count models, replace only the GSE162631 contrasts.
columns = ["cohort", "patient", "difference"]
preprocessing_table = patient.read_table("preprocessing_patient_contrasts")
sensitivity_table = preprocessing_table.loc[preprocessing_table["scenario"] != "stored_X"]
for (label, scenario), group_table in sensitivity_table.groupby(["labels", "scenario"]):
    scenario_table = group_table[columns].copy()
    scenario_table["analysis"] = "preproc__" + label + "__" + scenario
    parts.append(scenario_table)
source_table = patient.read_table("GSE162631_source_sensitivity")
other_table = baseline_table_total.query("analysis == 'uniform_labels' and cohort != 'GSE162631'")[columns]
for scenario, group_table in source_table.groupby("scenario"):
    scenario_table_total = pd.concat([other_table, group_table[columns]], ignore_index=True)
    scenario_table_total["analysis"] = "raw__" + scenario
    parts.append(scenario_table_total)

input_table_total = pd.concat(parts, ignore_index=True)
if input_table_total.duplicated(["analysis", "cohort", "patient"]).any():
    raise ValueError("A model contains duplicate cohort and patient keys")
patient.write_table("meta_patient_inputs", input_table_total)
print("Models:", input_table_total["analysis"].nunique(), "Patient contrasts:", len(input_table_total))
