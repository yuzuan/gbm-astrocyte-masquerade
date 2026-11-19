# 目的：比较 scGBM 和自有队列中，AC-like 的免疫相关基因平均表达在四种肿瘤亚型中的名次。
# 输入：data/external/scGBM/adata2.h5ad 的 X、Minor_cluster，以及 data/malignant_gbm_cnv.h5ad 的 X、state。
# 输出：03_result/11_免疫相关基因跨队列排名/01_免疫基因表达与排名.csv
# 运行方式：在仓库根目录用项目 Python 执行 python 11_免疫相关基因跨队列排名.py，沿用 AnnData 和 NumPy 表达均值。

from contextlib import closing
from importlib import import_module

import anndata as ad
import numpy as np
import pandas as pd

communication = import_module("00a_细胞通讯")

# 已发面板按类别指定基因及顺序。每行最终对应两个表达矩阵都能测量的一个基因。
panel = {
    "MHC-I": ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2"],
    "MHC-II": ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1"],
    "Checkpoint ligands": ["CD274", "PDCD1LG2", "CD80", "CD86", "LGALS9", "NECTIN2", "PVR", "VTCN1"],
    "Immunosuppressive": ["TGFB1", "TGFB2", "IL10", "IDO1", "PTGS2", "ARG1", "NOS2"],
    "Complement regulators": ["CD46", "CD55", "CD59"],
    "Dont-eat-me": ["CD47", "CD24"],
    "Immunosuppressive chemokines": ["CCL2", "CSF1", "CXCL12", "CCL22"],
    "IFN response": ["IFITM1", "IFITM2", "IFITM3", "ISG15", "MX1", "STAT1", "IRF1"],
}
category_by_gene = {}
for category, genes in panel.items():
    for gene in genes:
        category_by_gene[gene] = category

# 先按基因名取交集。scGBM 每亚型最多抽 5000 个细胞，自有队列保留该亚型全部细胞。
scgbm = ad.read_h5ad("data/external/scGBM/adata2.h5ad", backed="r")
with closing(scgbm.file):
    own_dataset = ad.read_h5ad("data/malignant_gbm_cnv.h5ad", backed="r")
    with closing(own_dataset.file):
        shared_genes = [gene for gene in category_by_gene if gene in scgbm.var_names and gene in own_dataset.var_names]
        scgbm_means = {}
        for state, mask in communication.scgbm_state_masks(scgbm.obs["Minor_cluster"]).items():
            positions = communication.sampled_scgbm_positions(mask)
            expression_table = communication.cell_expression_table(scgbm, positions, shared_genes)
            scgbm_means[state] = {gene: np.mean(expression_table[gene].to_numpy()) for gene in shared_genes}

        # 自有标签用 AClike 等原始拼写。两个队列分别排名，只比较名次。
        own_means = {}
        for state in ("AClike", "MESlike", "OPClike", "NPClike"):
            positions = np.flatnonzero(own_dataset.obs["state"].eq(state))
            expression_table = communication.cell_expression_table(own_dataset, positions, shared_genes)
            own_means[state] = {gene: np.mean(expression_table[gene].to_numpy()) for gene in shared_genes}

# 基因在行、亚型在列。名次等于 1 加严格高于 AC 的亚型数，并列给相同名次。均值在判定后才取三位。
scgbm_table = pd.DataFrame(scgbm_means, index=shared_genes)
own_table = pd.DataFrame(own_means, index=shared_genes)
if not np.isfinite(scgbm_table.to_numpy()).all() or not np.isfinite(own_table.to_numpy()).all():
    raise ValueError("亚型均值有缺失，无法进行四亚型排名。")
scgbm_ranks = scgbm_table.rank(axis=1, ascending=False, method="min")["AC"].astype(int)
own_ranks = own_table.rank(axis=1, ascending=False, method="min")["AClike"].astype(int)
result_table = pd.DataFrame(
    {
        "category": [category_by_gene[gene] for gene in shared_genes],
        "gene": shared_genes,
        "scGBM_AC": scgbm_table["AC"].round(3),
        "scGBM_AC_rank": scgbm_ranks,
        "own_AC": own_table["AClike"].round(3),
        "own_AC_rank": own_ranks,
        "rank_concordant": scgbm_ranks == own_ranks,
        "both_AC_top2": (scgbm_ranks <= 2) & (own_ranks <= 2),
    }
)

# 当 AC 与本队列最大的表达均值相等时应排第一；分别用原始均值核实名次方向。
if not scgbm_ranks.eq(1).equals(scgbm_table["AC"].eq(scgbm_table.max(axis=1))):
    raise ValueError("scGBM 中 AC 的第一名与逐基因最大均值不符。")
if not own_ranks.eq(1).equals(own_table["AClike"].eq(own_table.max(axis=1))):
    raise ValueError("自有队列中 AC 的第一名与逐基因最大均值不符。")
communication.write_gene_table("03_result/11_免疫相关基因跨队列排名/01_免疫基因表达与排名.csv", result_table, shared_genes)
print(f"两队列共有 {len(shared_genes)} 个面板基因，AC 名次相同 {int(result_table['rank_concordant'].sum())} 个。")
