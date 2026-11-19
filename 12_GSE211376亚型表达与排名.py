# 目的：计算 GSE211376 免疫面板在四种肿瘤亚型中的平均表达，将 AC 的队列内名次与 scGBM 比较。
# 输入：03_result/11_免疫相关基因跨队列排名/01_免疫基因表达与排名.csv（gene、scGBM_AC_rank 等已发字段），
# 也可用 data/intermediate/immune_evasion_cross_cohort.csv。data/intermediate/GSE211376_metadata.csv.gz
# 用首列细胞编号及 predicted.high_hierarchy；data/intermediate/GSE211376_raw_counts.tsv.gz 每行一个基因。
# 输出：03_result/12_GSE211376亚型表达与排名/01_四种亚型平均表达.csv、同目录的 02_跨队列AC排名对照.csv
# 运行方式：在仓库根目录用项目 Python 执行 python 12_GSE211376亚型表达与排名.py，沿用 NumPy 与 pandas 排序。

import gzip
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

communication = import_module("00a_细胞通讯")
upstream = import_module("00a_基因表达与聚类")

# 复用 11 实际产出的基因顺序和 scGBM 名次。旧分析外部提供的冻结面板沿用同名字段。
panel_path = upstream.result_or_frozen(
    Path("03_result/11_免疫相关基因跨队列排名/01_免疫基因表达与排名.csv"), "immune_evasion_cross_cohort.csv"
)
panel_table = pd.read_csv(panel_path)
genes = panel_table["gene"].tolist()
gene_set = set(genes)

# 原始计数文件首行直接列细胞编号；元数据首列为行名。留住全部细胞的文库总数，再按标签取亚型。
with gzip.open("data/intermediate/GSE211376_raw_counts.tsv.gz", "rt", encoding="utf-8") as stream:
    cells = communication.count_matrix_cells(stream)
    metadata_table = upstream.gse211376_metadata(cells)
    labels = metadata_table["predicted.high_hierarchy"]
    total_counts = np.zeros(len(cells))
    panel_counts = {}
    for line in stream:
        gene, counts = communication.read_gene_counts(line, len(cells))
        total_counts += counts
        if gene in gene_set:
            panel_counts[gene] = counts

# 分母包括面板外的基因。每个基因先逐细胞按每万计数取 log1p，再求亚型均值，缺测基因整行留空。
size_factors = communication.library_size_factors(total_counts)
states = {"AC": "AC-like", "MES": "MES-like", "OPC": "OPC-like", "NPC": "NPC-like"}
masks = {state: labels.eq(label).to_numpy() for state, label in states.items()}
expression_table = pd.DataFrame(np.nan, index=pd.Index(genes, name="gene"), columns=list(states))
for gene, counts in panel_counts.items():
    values = np.log1p(counts / size_factors)
    for state, mask in masks.items():
        if mask.any():
            expression_table.loc[gene, state] = values[mask].mean()

# 四种亚型都有值才排名。旧实现按 AC、MES、OPC、NPC 的原列序做降序 quicksort，并取 AC 的序号。
ac_ranks = pd.Series(np.nan, index=expression_table.index)
for gene, state_means in expression_table.dropna().iterrows():
    ordered = state_means.sort_values(ascending=False, kind="quicksort")
    ac_ranks.loc[gene] = ordered.index.get_loc("AC") + 1

# 以基因为单位一对一合并，保留已有面板全部字段；两列一致性指标在本表比较 GSE211376 与 scGBM。
gse_table = pd.DataFrame({"gse211376_AC": expression_table["AC"], "gse211376_AC_rank": ac_ranks})
comparison_table_total = panel_table.merge(
    gse_table, left_on="gene", right_index=True, how="left", validate="one_to_one"
)
rank_columns = ["gse211376_AC_rank", "scGBM_AC_rank"]
valid_table = comparison_table_total.dropna(subset=rank_columns).copy()
valid_table[rank_columns] = valid_table[rank_columns].astype(int)
valid_table["rank_concordant"] = valid_table["gse211376_AC_rank"].eq(valid_table["scGBM_AC_rank"])
valid_table["both_AC_top2"] = valid_table[rank_columns].le(2).all(axis=1)

# 对照输入面板与四亚型表达覆盖取得有效基因分母，两个输出都按原基因顺序逐列回读。
complete = expression_table.notna().all(axis=1)
expected = panel_table["scGBM_AC_rank"].notna() & panel_table["gene"].isin(expression_table.index[complete])
if not valid_table[rank_columns].isin(range(1, len(states) + 1)).all().all():
    raise ValueError("名次超出四个亚型的范围。")
output_dir = Path("03_result/12_GSE211376亚型表达与排名")
communication.write_gene_table(output_dir / "01_四种亚型平均表达.csv", expression_table.reset_index(), genes)
communication.write_gene_table(output_dir / "02_跨队列AC排名对照.csv", valid_table, panel_table.loc[expected, "gene"])
print(f"面板 {len(genes)} 个基因，两个队列可比较 AC 名次 {len(valid_table)} 个。")
