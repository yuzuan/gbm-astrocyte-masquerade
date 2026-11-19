"""目的　将原注释、统一注释、计数预处理及来源计数四类患者对照整理为 REML 模型输入。
输入　项目补充表 S3a、S7a 的原研究编号，以及 01b 和 02 的患者表达差。每行是一位患者在一个队列的差值。
输出　03_患者模型输入/01_模型患者表达差.csv，列为 analysis、cohort、patient、difference。
运行　为全计数路线设置 GLIOMA_VALIDATION_ROOT 和 PLOS_REVISION_DIR，在 revision_analysis 下用 Python 执行本脚本。
"""

import json
import os
from importlib import import_module
from pathlib import Path

import pandas as pd

patient = import_module("00a_患者表达")
tables = Path(os.environ["GLIOMA_VALIDATION_ROOT"]) / "claude_outputs/public_only_submission/tables"

# S3a 在每个队列的一行内保存患者对照，S7a 已经一行一名患者。两者取相同的目标基因和患者列。
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

# 保存每种预处理和标签组合。来源计数仅重算 GSE162631，其余队列沿用统一注释对照。
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
    raise ValueError("同一模型内出现重复的队列和患者组合")
patient.write_table("meta_patient_inputs", input_table_total)
print("模型", input_table_total["analysis"].nunique(), "种，患者对照记录", len(input_table_total), "条")
