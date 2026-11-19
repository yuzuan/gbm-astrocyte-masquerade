# 目的：按 scGBM 发送亚型计算配体均值与接收群受体均值的乘积，再估计自有 GBM 星形细胞混合校正前后的分数。
# 输入：data/external/scGBM/adata2.h5ad（X、Minor_cluster、Major_category），data/malignant_gbm_cnv.h5ad（X、state），
# data/cellcell_communication.csv（condition、sender、receiver、ligand、receptor、mean_L_sender、score、perm_p）。
# CNV 比例取自 data/infercnv_cell_scores.csv（cell_type、cnv_pos）和 data/cell_metadata.csv（condition，首列为细胞编号）。
# 只有旧冻结汇总时可用 data/intermediate/communication_analysis_results.json 的 contamination_rates。
# 输出：03_result/07_scGBM细胞通讯与混合校正/01_通讯得分与混合校正.json
# 运行方式：在仓库根目录用项目 Python 执行 python 07_scGBM细胞通讯与混合校正.py，沿用 AnnData 和 NumPy 抽样。

import json
from contextlib import closing
from importlib import import_module
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

communication = import_module("00a_细胞通讯")
upstream = import_module("00a_基因表达与聚类")

# scGBM 用四种发送亚型分别对髓系和 T/NK 评分。共有受体在同一接收群内只求一次均值。
scgbm = ad.read_h5ad("data/external/scGBM/adata2.h5ad", backed="r")
with closing(scgbm.file):
    tam_scores = communication.scgbm_product_scores(scgbm, "TAM/Macrophage")
    tnk_scores = communication.scgbm_product_scores(scgbm, "T/NK")

# 自有 GBM 的混合比例以逐细胞 CNV 标记为依据。两表先按编号对齐，原文件的行序不同。
scores_path = Path("data/infercnv_cell_scores.csv")
metadata_path = Path("data/cell_metadata.csv")
if scores_path.exists() or metadata_path.exists():
    cnv_table = pd.read_csv(scores_path, index_col=0, dtype={"cnv_pos": "boolean"})
    metadata_table = pd.read_csv(metadata_path, index_col=0)
    cell_table_total = cnv_table.join(metadata_table[["condition"]], how="left", validate="one_to_one")
    if cell_table_total[["cnv_pos", "condition", "cell_type"]].isna().any().any():
        raise ValueError("混合校正需要每个 CNV 细胞的类型、阳性标记和组织条件。")

    # 该表逐细胞记录 CNV 阳性。总体比例与分条件比例的分母不同，不能把总体比例用于 GBM 通讯。
    astro_table = cell_table_total.loc[cell_table_total["cell_type"].eq("Astrocyte")]
    gbm_positive = astro_table.loc[astro_table["condition"].eq("GBM"), "cnv_pos"]
    normal_positive = astro_table.loc[astro_table["condition"].eq("Normal"), "cnv_pos"]
    myeloid_positive = cell_table_total.loc[cell_table_total["cell_type"].eq("Myeloid"), "cnv_pos"]
    oligo_positive = cell_table_total.loc[cell_table_total["cell_type"].eq("Oligodendrocyte"), "cnv_pos"]
    if any(group.empty for group in (gbm_positive, normal_positive, myeloid_positive, oligo_positive)):
        raise ValueError("计算历史比例所需的细胞类型或组织条件为空。")

    # 历史公式以三位小数的 GBM 比例作混合权重，展示的百分数保留一位。原字段供已发布的下游读取。
    rates = {
        "Astrocyte_cnv_pos": round(100 * astro_table["cnv_pos"].mean(), 1),
        "Astrocyte_total": len(astro_table),
        "Astrocyte_cnv_pos_n": int(astro_table["cnv_pos"].sum()),
        "Astrocyte_cleaned_n": int((~astro_table["cnv_pos"]).sum()),
        "Myeloid_cnv_pos": round(100 * myeloid_positive.mean(), 1),
        "Oligodendrocyte_cnv_pos": round(100 * oligo_positive.mean(), 1),
        "Astrocyte_cnv_pos_GBM": round(100 * gbm_positive.mean(), 1),
        "Astrocyte_cnv_pos_Normal": round(100 * normal_positive.mean(), 1),
        "f_malig_used": round(float(gbm_positive.mean()), 3),
    }
else:

    # 外来冻结 JSON 只提供历史比例。配受体分数和混合校正仍由本次读入的数据计算。
    with Path("data/intermediate/communication_analysis_results.json").open(encoding="utf-8") as stream:
        frozen_result = json.load(stream)
    rates = frozen_result["contamination_rates"].copy()

# 把分子、分母和用于校正的权重互相核对，避免误用缺少分条件比例的早期冻结表。
for key in (
    "Astrocyte_cnv_pos", "Myeloid_cnv_pos", "Oligodendrocyte_cnv_pos",
    "Astrocyte_cnv_pos_GBM", "Astrocyte_cnv_pos_Normal",
):
    if not 0 <= rates[key] <= 100:
        raise ValueError(f"历史字段 {key} 缺失或超出百分数范围。")
positive_fraction = rates["f_malig_used"]
if rates["Astrocyte_cnv_pos_n"] + rates["Astrocyte_cleaned_n"] != rates["Astrocyte_total"]:
    raise ValueError("星形细胞阳性数和保留数与总数不符。")
astro_percent = round(100 * rates["Astrocyte_cnv_pos_n"] / rates["Astrocyte_total"], 1)
if astro_percent != rates["Astrocyte_cnv_pos"] or not 0 <= positive_fraction < 1:
    raise ValueError("星形细胞总阳性率或 GBM 混合权重不合理。")
if not np.isclose(positive_fraction * 100, rates["Astrocyte_cnv_pos_GBM"]):
    raise ValueError("混合权重与 GBM 条件下的阳性率不符。")
rates["note"] = f"GBM-specific rate ({rates['Astrocyte_cnv_pos_GBM']:.1f}%) used for GBM communication decontamination"

# 自有数据按 AClike 标签取全组细胞。保留历史候选配体，只对既有通讯记录中的配受体进行校正。
own_dataset = ad.read_h5ad("data/malignant_gbm_cnv.h5ad", backed="r")
with closing(own_dataset.file):
    key_ligands = ["LGALS1", "ANXA1", "TGFB1", "TGFB2", "CCL2", "CSF1", "CD274", "VEGFA", "EGFR", "SERPINE1"]
    available_ligands = [gene for gene in key_ligands if gene in own_dataset.var_names]
    ac_positions = np.flatnonzero(own_dataset.obs["state"].eq("AClike"))
    ac_table = communication.cell_expression_table(own_dataset, ac_positions, available_ligands)

# 输入通讯表是一行一对配受体。保留 GBM、Astrocyte 到 Myeloid 的记录及原始行序。
comm_table = pd.read_csv("data/cellcell_communication.csv")
eligible = comm_table["condition"].eq("GBM") & comm_table["sender"].eq("Astrocyte")
eligible &= comm_table["receiver"].eq("Myeloid")
selected_table = comm_table.loc[eligible & comm_table["ligand"].isin(available_ligands)]
if not np.isfinite(selected_table[["mean_L_sender", "score", "perm_p"]].to_numpy()).all():
    raise ValueError("入选通讯记录缺少配体均值、得分或置换 P 值。")

# 每条记录沿用原始分数的计算精度，解出其非阳性部分的配体均值，负数按旧方案截到零。
pollution_rows = []
for row in selected_table.itertuples(index=False):
    malignant_mean = float(np.mean(ac_table[row.ligand].to_numpy()))
    corrected_mean = max(0, (row.mean_L_sender - positive_fraction * malignant_mean) / (1 - positive_fraction))

    # 用校正前后的配体均值之比缩放同一条分数。置换 P 值来自原始通讯表，校正后没有重新置换。
    corrected_score = row.score * (corrected_mean / row.mean_L_sender) if row.mean_L_sender > 0 else 0
    percent_change = (corrected_score - row.score) / row.score * 100 if row.score > 0 else 0
    pollution_rows.append(
        {
            "ligand": row.ligand,
            "receptor": row.receptor,
            "mean_L_before": round(row.mean_L_sender, 4),
            "mean_L_after": round(corrected_mean, 4),
            "mean_L_malig": round(malignant_mean, 4),
            "score_before": round(row.score, 4),
            "score_after": round(corrected_score, 4),
            "pct_change": round(percent_change, 1),
            "sig_before": "yes" if row.perm_p < 0.05 else "no",
        }
    )

# 核实每条 scGBM 得分都有全部发送亚型，自有通讯输出的配受体顺序与筛选后的输入完全对应。
for receiver_scores in (tam_scores, tnk_scores):
    for state_scores in receiver_scores.values():
        if set(state_scores) != {"AC", "MES1", "OPC", "NPC"}:
            raise ValueError("通讯得分缺少发送亚型。")
written_pairs = [(record["ligand"], record["receptor"]) for record in pollution_rows]
if written_pairs != list(zip(selected_table["ligand"], selected_table["receptor"])):
    raise ValueError("校正后的配受体与原始 GBM 通讯记录不匹配。")

result = {
    "tam_scores": tam_scores,
    "tnk_scores": tnk_scores,
    "pollution_results": pollution_rows,
    "contamination_rates": rates,
}
upstream.write_checked_json(Path("03_result/07_scGBM细胞通讯与混合校正/01_通讯得分与混合校正.json"), result)
print(f"髓系配受体 {len(tam_scores)} 对，T/NK 配受体 {len(tnk_scores)} 对，混合校正 {len(pollution_rows)} 条。")
