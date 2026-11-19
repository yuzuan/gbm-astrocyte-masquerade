"""目的　将返修统计表写入一个可核对的 S14 工作簿，使用各队列内部的患者编号。
输入　01 至 04 的逐患者及合并统计 CSV，冻结的数据表标识用于命名工作表。
输出　05_补充图与工作簿/05_S14患者统计表.xlsx。另写同目录的 06_S14工作表目录.csv。
运行　在 revision_analysis 下用 Python 运行本脚本。openpyxl 设置并回读工作簿。
"""

from importlib import import_module

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font

patient = import_module("00a_患者表达")

# 一位患者在不同预处理情景中可重复出现。先按队列去重，已经公开的 P 编号直接沿用。
eligibility_table = patient.read_table("preprocessing_eligibility")
patient_codes = {}
for cohort, group_table in eligibility_table.groupby("cohort"):
    subjects = sorted(group_table["patient"].astype(str).unique())
    for i, subject in enumerate(subjects, 1):
        is_public_code = subject.startswith("P") and subject[1:].isdigit()
        patient_codes[(cohort, subject)] = subject if is_public_code else f"P{i}"

path = patient.result_root() / "05_补充图与工作簿/05_S14患者统计表.xlsx"
path.parent.mkdir(parents=True, exist_ok=True)
sheet_rows = []
with pd.ExcelWriter(path, engine="openpyxl") as writer:
    for i, name in enumerate(sorted(patient.table_files), 1):
        result_table = patient.read_table(name)

        # patient 的含义由 cohort 决定。GSE162631 的单队列表可能没有 cohort 字段。
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

        # 沿用补充材料的工作表标识。Excel 表名最多31字符，序号区分可能重名的缩写。
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

# 按已写入的表清单重新打开文件，核对各张表行数（加表头）和列数。
workbook = load_workbook(path, read_only=True, data_only=True)
try:
    for row in index_table.itertuples():
        worksheet = workbook[row.sheet]
        if (worksheet.max_row, worksheet.max_column) != (row.rows + 1, row.columns):
            raise ValueError(f"S14 工作表维数不符：{row.sheet}")
finally:
    workbook.close()

# 单独的 CSV 目录与工作簿 README 使用相同的表序、表名和行数。
sheet_index = path.with_name("06_S14工作表目录.csv")
summary_table = index_table[["sheet", "source_table", "rows"]]
summary_table.to_csv(sheet_index, index=False)
pd.testing.assert_frame_equal(pd.read_csv(sheet_index), summary_table)
print("汇总数据表", len(sheet_rows), "张")
