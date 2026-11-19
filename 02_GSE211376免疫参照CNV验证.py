# 目的：按 GSE211376 作者的细胞注释，验证合并的 Astrocyte + AC-like 群及两组各自的 +7/−10 阈上率。
# 输入：data/intermediate/GSE211376_raw_counts.tsv.gz（基因×条码），同目录 GSE211376_metadata.csv.gz
#       （条码、predicted.high_hierarchy、patient），data/external/scGBM/adata2.h5ad 的 var/chromosome。
# 输出：03_result/02_GSE211376免疫参照CNV验证/01_GSE211376免疫参照CNV结果.json。
# 运行方式：在仓库根目录运行 python 02_GSE211376免疫参照CNV验证.py，沿用原分析的 Python 数值精度。

from importlib import import_module
from pathlib import Path

import numpy as np

common = import_module("00a_基因表达与聚类")

# 旧分析逐行按 float32 读取整数计数，再以 float64 累加。各染色体使用实际匹配的基因作分母。
expression_table = common.gse211376_expression(common.chromosome_gene_sets(), np.float32)
metadata_table = common.gse211376_metadata(expression_table.index)
if metadata_table["patient"].isna().any():
    raise ValueError("GSE211376 有计数条码缺少患者编号，无法确定患者分母。")
cell_types = metadata_table["predicted.high_hierarchy"]

# 中心化只用两类 TAM，第 95 百分位另加入 CD4/CD8。后续按未取位的严格阈值逐细胞判定。
tam = cell_types.isin(["TAM-MG", "TAM-BDM"])
immune = tam | cell_types.eq("CD4/CD8")
cnv_scores, threshold, center7, center10 = common.centered_cnv(
    expression_table["chr7"].to_numpy(), expression_table["chr10"].to_numpy(), tam, immune
)
positive = cnv_scores > threshold

# 原脚本逐类报告细胞数、阈上比例与中位分数。按真实注释留存这些统计，供查看较细的分组。
by_annotation = {}
for label in (
    "Astrocyte", "AC-like", "MES-like", "NPC-like", "OPC-like", "OPC", "Oligodendrocyte",
    "TAM-MG", "TAM-BDM", "CD4/CD8", "Endothelial", "Pericyte", "Neuron",
):
    selected = cell_types.eq(label)
    if selected.any():
        group = common.positive_summary(positive, selected)
        group["cnv_score_median"] = float(np.median(cnv_scores[selected]))
        by_annotation[label] = group

# 粗分类是两个互斥作者标签的并集。合并 TAM 的分母同样直接取细胞，避免对组百分比求平均。
coarse = common.positive_summary(positive, cell_types.isin(["Astrocyte", "AC-like"]))
refined = common.positive_summary(positive, cell_types.eq("Astrocyte"))
ac_like = common.positive_summary(positive, cell_types.eq("AC-like"))
reference_rates = {}
for label in ("TAM-MG", "TAM-BDM", "Oligodendrocyte", "CD4/CD8"):
    group = common.positive_summary(positive, cell_types.eq(label))
    reference_rates[label] = group["cnv_pos_pct"]
combined_tam = common.positive_summary(positive, tam)
reference_rates["TAM_combined"] = combined_tam["cnv_pos_pct"]

# IDH 是历史版本登记的队列描述。GEO 条码表只有作者注释和患者编号，不能据它逐人核实 IDH。
result = {
    "dataset": "GSE211376",
    "source": "GEO GSE211376 snRNA-seq counts and author cell annotations.",
    "n_cells": len(expression_table),
    "n_patients": int(metadata_table["patient"].nunique()),
    "IDH": "wildtype",
    "idh_source": "Historical cohort description; individual IDH status is absent from the cell metadata.",
    "strict_threshold": float(threshold),
    "tam_ref_center": {"chr7": float(center7), "chr10": float(center10)},
    "coarse_astro_cluster": {"definition": "Author labels Astrocyte + AC-like", **coarse},
    "refined_astrocyte_only": refined,
    "AC_like": ac_like,
    "reference_FPR": reference_rates,
    "by_annotation": by_annotation,
}

# 核实粗分类分母等于两个子组之和；回读 JSON 全部字段和精确阈值后发布。
if coarse["n"] != refined["n"] + ac_like["n"] or coarse["n"] == 0:
    raise ValueError("GSE211376 星形细胞合并分母为空或与两个作者标签之和不符。")
common.write_checked_json(Path("03_result/02_GSE211376免疫参照CNV验证/01_GSE211376免疫参照CNV结果.json"), result)
print(f"GSE211376 作者标签合并 {coarse['n']} 个细胞，免疫参照阈上比例 {coarse['cnv_pos_pct']}%。")
