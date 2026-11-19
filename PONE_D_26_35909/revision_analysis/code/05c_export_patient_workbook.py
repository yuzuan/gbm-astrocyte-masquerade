"""Export the S14 workbook using patient identifiers assigned within each cohort.
Inputs are the patient-level and pooled CSVs from stages 01 through 04. Released table identifiers name the sheets.
Write 05_s14_patient_statistics.xlsx and 06_s14_sheet_index.csv in 05_supplementary_outputs.
From revision_analysis run python code/05c_export_patient_workbook.py. openpyxl writes and checks the workbook.
"""

from importlib import import_module

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font

patient = import_module("00a_patient_expression")

# Assign each patient a single code within its cohort across all scenarios. Keep existing public P codes.
eligibility_table = patient.read_table("preprocessing_eligibility")
patient_codes = {}
for cohort, group_table in eligibility_table.groupby("cohort"):
    subjects = sorted(group_table["patient"].astype(str).unique())
    for i, subject in enumerate(subjects, 1):
        is_public_code = subject.startswith("P") and subject[1:].isdigit()
        patient_codes[(cohort, subject)] = subject if is_public_code else f"P{i}"

path = patient.result_root() / "05_supplementary_outputs/05_s14_patient_statistics.xlsx"
path.parent.mkdir(parents=True, exist_ok=True)
sheet_rows = []
with pd.ExcelWriter(path, engine="openpyxl") as writer:
    for i, name in enumerate(sorted(patient.table_files), 1):
        result_table = patient.read_table(name)

        # Interpret patient codes within cohorts. GSE162631-only source tables may omit the cohort field.
        if "patient" in result_table:
            cohorts = (
                result_table["cohort"] if "cohort" in result_table else ["GSE162631"] * len(result_table)
            )
            result_table["patient"] = [
                patient_codes[(cohort, subject)] for cohort, subject in zip(cohorts, result_table["patient"])
            ]
        if name == "GSE162631_leave_patient_out":
            result_table["omitted"] = [
                patient_codes[("GSE162631", subject)] for subject in result_table["omitted"]
            ]
        if name == "GSE162631_public_count_audit":
            samples = []
            for sample in result_table["sample"]:
                subject = patient.patient_from_tumor_sample(sample)
                samples.append(patient_codes[("GSE162631", subject)] + "-S1")
            result_table["sample"] = samples

        # Use the supplementary table identifiers. A numeric prefix distinguishes names truncated to Excel's
        # 31-character limit.
        sheet = f"T{i:02d}_" + name[:26]
        result_table.to_excel(writer, sheet_name=sheet, index=False)
        sheet_rows.append(
            {
                "sheet": sheet,
                "source_table": name,
                "rows": len(result_table),
                "columns": len(result_table.columns),
            }
        )

    index_table = pd.DataFrame(sheet_rows)
    index_table[["sheet", "source_table", "rows"]].to_excel(writer, sheet_name="README", index=False)
    for worksheet in writer.book:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
            worksheet.column_dimensions[cell.column_letter].width = min(42, max(13, len(str(cell.value)) + 2))

# Open the workbook again and check each sheet's row count, including its header, and its column count.
workbook = load_workbook(path, read_only=True, data_only=True)
try:
    for row in index_table.itertuples():
        worksheet = workbook[row.sheet]
        if (worksheet.max_row, worksheet.max_column) != (row.rows + 1, row.columns):
            raise ValueError(f"Unexpected dimensions for S14 worksheet {row.sheet}")
finally:
    workbook.close()

# The CSV index and the workbook README use the same sheet order, sheet names and row counts.
sheet_index = path.with_name("06_s14_sheet_index.csv")
summary_table = index_table[["sheet", "source_table", "rows"]]
summary_table.to_csv(sheet_index, index=False)
pd.testing.assert_frame_equal(pd.read_csv(sheet_index), summary_table)
print("Data tables in the workbook:", len(sheet_rows))
