# 目的：仅按谱系标志表达选择 GSE211376 星形胶质细胞，估算其 +7/−10 阈上比例，并与作者标签比较。
# 输入：data/intermediate/GSE211376_raw_counts.tsv.gz（基因×条码），同目录 GSE211376_metadata.csv.gz
#       （条码、predicted.high_hierarchy），data/external/scGBM/adata2.h5ad 的 var/chromosome。
# 输出：03_result/03_GSE211376标志基因定义星形细胞/01_GSE211376标志基因定义星形细胞CNV结果.json。
# 运行方式：在仓库根目录运行 python 03_GSE211376标志基因定义星形细胞.py，沿用原 Python 计数读取方式。

from importlib import import_module
from pathlib import Path

import numpy as np

common = import_module("00a_基因表达与聚类")

# 原面板中，髓系比 GSE174554 多 C1QA 与 C1QB。计数按 float64 读取，三类面板与染色体各自累加。
gene_sets = common.chromosome_gene_sets()
gene_sets.update(common.lineage_genes)
gene_sets["imm"] = common.lineage_genes["imm"] + ["C1QA", "C1QB"]
expression_table = common.gse211376_expression(gene_sets, np.float64)

# 原定义要求星形得分高于全队列上四分位、免疫得分不高于中位数、少突得分低于 1.0。
astro_score = expression_table["astro"].to_numpy()
immune_score = expression_table["imm"].to_numpy()
oligo_score = expression_table["oligo"].to_numpy()
astrocytes = (astro_score > np.quantile(astro_score, 0.75)) & (immune_score <= np.quantile(immune_score, 0.50))
astrocytes &= oligo_score < 1.0

# 髓系参照取免疫得分高于第 90 百分位的细胞，同时用于中心化和定阈值。少突只作比较。
myeloid = immune_score > np.quantile(immune_score, 0.90)
oligodendrocytes = (oligo_score > np.quantile(oligo_score, 0.75)) & (astro_score <= np.quantile(astro_score, 0.50))
cnv_scores, threshold, _, _ = common.centered_cnv(
    expression_table["chr7"].to_numpy(), expression_table["chr10"].to_numpy(), myeloid, myeloid
)
positive = cnv_scores > threshold
marker_astro = common.positive_summary(positive, astrocytes)
reference_myeloid = common.positive_summary(positive, myeloid)
reference_oligo = common.positive_summary(positive, oligodendrocytes)

# 作者注释只用于事后的两组比较。按条码对齐，即使元数据有额外行也不会用错顺序或静默丢字段。
metadata_table = common.gse211376_metadata(expression_table.index)
cell_types = metadata_table["predicted.high_hierarchy"]
result = {
    "dataset": "GSE211376",
    "n_cells": len(expression_table),
    "strict_threshold": float(threshold),
    "marker_defined_astrocyte": marker_astro,
    "reference_FPR": {"myeloid": reference_myeloid["cnv_pos_pct"], "oligodendrocyte": reference_oligo["cnv_pos_pct"]},
    "author_astrocyte_only": common.positive_summary(positive, cell_types.eq("Astrocyte")),
    "author_AClike": common.positive_summary(positive, cell_types.eq("AC-like")),
}

# 两个阈值筛选应让星形目标与髓系参照互斥；空目标也不能生成可解释的星形结果。
if marker_astro["n"] == 0 or np.any(astrocytes & myeloid):
    raise ValueError("GSE211376 标志定义的星形细胞为空或混入定阈值的髓系细胞。")
common.write_checked_json(Path("03_result/03_GSE211376标志基因定义星形细胞/01_GSE211376标志基因定义星形细胞CNV结果.json"), result)
print(f"GSE211376 标志定义星形细胞 {marker_astro['n']} 个，阈上比例 {marker_astro['cnv_pos_pct']}%。")
