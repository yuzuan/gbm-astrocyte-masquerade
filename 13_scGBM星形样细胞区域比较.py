# 目的：比较 scGBM 肿瘤与瘤旁 AC 细胞的表达，另按同一队列 AC 的高低 CNV 分数组重建历史对照折数。
# 输入：data/external/scGBM/adata2.h5ad 的 X、Minor_cluster、tissue_type、Major_category、cnv_score。
# 输出：03_result/13_scGBM星形样细胞区域比较/01_肿瘤与瘤旁AC表达.csv
# 运行方式：在仓库根目录用项目 Python 执行 python 13_scGBM星形样细胞区域比较.py，使用 AnnData 读取 scGBM 对象。

from contextlib import closing
from importlib import import_module

import anndata as ad
import numpy as np
import pandas as pd

communication = import_module("00a_细胞通讯")

# 原始会话读取与 07、11 相同的 scGBM 对象。按实际可用基因保留历史面板顺序，不把未测基因当零表达。
scgbm = ad.read_h5ad("data/external/scGBM/adata2.h5ad", backed="r")
with closing(scgbm.file):
    panel = [
        "GFAP", "AQP4", "S100B", "SLC1A3", "CLU", "ALDOC", "CD44", "VIM", "CHI3L1", "CHI3L2", "EGFR", "PDGFRA",
        "SOX2", "NES", "HIF1A", "VEGFA", "LGALS1", "LGALS3", "SERPINE1", "ANXA1",
    ]
    genes = [gene for gene in panel if gene in scgbm.var_names]
    is_ac = scgbm.obs["Minor_cluster"].eq("AC")
    tumor = is_ac & scgbm.obs["tissue_type"].eq("Tumor")
    peri = is_ac & scgbm.obs["tissue_type"].eq("Peritumoral")

    # 已发 MES1_mean、OPC_mean 的参照组含两种组织。各组超过上限时独立抽样；检出定义为 X 大于零。
    region_masks = {
        "AC_Tumor": tumor,
        "AC_Peri": peri,
        "MES1": scgbm.obs["Minor_cluster"].eq("MES1"),
        "OPC": scgbm.obs["Minor_cluster"].eq("OPC"),
    }
    comparison_table = pd.DataFrame({"gene": genes})
    detection = {}
    for group, mask in region_masks.items():
        positions = communication.sampled_scgbm_positions(mask)
        expression_table = communication.cell_expression_table(scgbm, positions, genes)
        comparison_table[f"{group}_mean"] = [np.mean(expression_table[gene].to_numpy()) for gene in genes]
        if group in ("AC_Tumor", "AC_Peri"):
            detection[group] = [(expression_table[gene].to_numpy() > 0).mean() * 100 for gene in genes]

    # 只对 AC 的两个组织组报告检出率和比值。瘤旁均值为零时，区域倍数没有定义。
    comparison_table["AC_Tumor_pct"] = detection["AC_Tumor"]
    comparison_table["AC_Peri_pct"] = detection["AC_Peri"]
    peri_mean = comparison_table["AC_Peri_mean"].replace(0, np.nan)
    comparison_table["fold_T_vs_P"] = comparison_table["AC_Tumor_mean"] / peri_mean

    # 历史 own_fold 列逐基因对应 scGBM 的 AC 高/低分表。依该表生成脚本，用全体参考细胞的第 95 百分位划分全体 AC。
    cnv = scgbm.obs["cnv_score"]
    reference = scgbm.obs["Major_category"].isin(["TAM/Macrophage", "Microglia", "T/NK"])
    if not reference.any() or not np.isfinite(cnv.loc[reference | is_ac]).all():
        raise ValueError("参考细胞不足或 AC、参考组的 CNV 分数有缺失。")
    threshold = np.percentile(cnv.loc[reference].to_numpy(), 95)
    high = is_ac & cnv.gt(threshold)
    low = is_ac & cnv.le(threshold)
    historical_folds = pd.Series(np.nan, index=genes)

    # 该折数用两组全部细胞，低分组均值的下限 0.001 和发布精度两位均来自原比较表；沿用其小数再分档。
    if high.any() and low.any():
        high_table = communication.cell_expression_table(scgbm, np.flatnonzero(high), genes)
        low_table = communication.cell_expression_table(scgbm, np.flatnonzero(low), genes)
        for gene in genes:
            high_mean = np.mean(high_table[gene].to_numpy())
            low_mean = np.mean(low_table[gene].to_numpy())
            fold = high_mean / max(low_mean, 0.001)
            historical_folds.loc[gene] = float(str(round(fold, 2)))
    comparison_table["own_fold"] = comparison_table["gene"].map(historical_folds)

# 保留已发分类标签。own_interpret 的方向表示同队列 CNV 高/低分组；边界 0.8、1.3 以及其间均为 shared。
for column, target, above, below in (
    ("own_fold", "own_interpret", "malignant↑", "normal↑"),
    ("fold_T_vs_P", "scgbm_interpret", "Tumor↑", "Peri↑"),
):
    values = comparison_table[column]
    comparison_table[target] = np.select(
        [values.isna(), values.gt(1.3), values.lt(0.8)], ["NA", above, below], default="shared"
    )

# 两组实际 CNV 均值汇总全组，表达均值则来自上面的抽样。按可判定基因计算同方向数，不从均值判断个体恶性。
comparable = comparison_table["own_fold"].notna() & comparison_table["fold_T_vs_P"].notna()
site_directions = comparison_table["scgbm_interpret"].replace({"Tumor↑": "malignant↑", "Peri↑": "normal↑"})
same_direction = comparable & comparison_table["own_interpret"].eq(site_directions)
comparison_table["CAVEAT"] = (
    f"Among all AC cells, mean cnv_score was {cnv[tumor].mean():.6f} in Tumor (n={int(tumor.sum())}) "
    f"and {cnv[peri].mean():.6f} in Peritumoral (n={int(peri.sum())}). "
    "fold_T_vs_P compares locations. Historical own_fold and its direction labels compare high versus low CNV scores "
    f"in AC cells of this same scGBM cohort. Direction categories agree for {int(same_direction.sum())} "
    f"of {int(comparable.sum())} evaluable genes."
)

# 从两组检出率核实百分数单位，并检查零分母是否正确留空。回读完整 CSV 时另外核对面板顺序与全部字段。
for column in ("AC_Tumor_pct", "AC_Peri_pct"):
    if not comparison_table[column].between(0, 100).all():
        raise ValueError(f"{column} 的检出率不在百分数范围内。")
if not comparison_table["fold_T_vs_P"].isna().equals(comparison_table["AC_Peri_mean"].eq(0)):
    raise ValueError("区域折数的缺失与瘤旁零均值不对应。")
communication.write_gene_table("03_result/13_scGBM星形样细胞区域比较/01_肿瘤与瘤旁AC表达.csv", comparison_table, genes)
print(f"区域与 CNV 分组均可判定 {int(comparable.sum())} 个基因，方向类别相同 {int(same_direction.sum())} 个。")
