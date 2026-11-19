# 目的：读取 snGBM 已有 LIANA 结果，计算 AC-like 配体排名，并与 scGBM 髓系接收群的表达乘积分数对照。
# 输入：data/external/snGBM/results/6_CCC_liana/adata_with_liana.h5ad 的 uns/liana_res
# （source、target、ligand_complex、lr_means）。scGBM 的 tam_scores 优先取模块 07 的
# 03_result/07_scGBM细胞通讯与混合校正/01_通讯得分与混合校正.json，也可读 data/intermediate/communication_analysis_results.json。
# 单独运行且没有上述汇总时，使用 data/external/scGBM/adata2.h5ad（X、Minor_cluster、Major_category）。
# 输出：03_result/08_snGBM配体通讯排名/01_LIANA配体排名与方法对照.json
# 运行方式：在仓库根目录用项目 Python 执行 python 08_snGBM配体通讯排名.py，沿用 AnnData 的 LIANA 结果表。

import json
from contextlib import closing
from importlib import import_module
from math import isfinite
from pathlib import Path

import anndata as ad

communication = import_module("00a_细胞通讯")
upstream = import_module("00a_基因表达与聚类")

# 每条 LIANA 记录对应一组发送群、接收群和配受体复合物。沿用四种恶性亚型及 Macrophage 接收群。
sngbm = ad.read_h5ad("data/external/snGBM/results/6_CCC_liana/adata_with_liana.h5ad", backed="r")
with closing(sngbm.file):
    liana_table = sngbm.uns["liana_res"]
    cell_count = sngbm.n_obs
malignant_states = ["AC-like", "MES-like", "OPC-like", "NPC-like"]
selected = liana_table["source"].isin(malignant_states) & liana_table["target"].eq("Macrophage")
macrophage_table = liana_table.loc[selected]

# 每个候选配体先在同一发送亚型的不同受体间取最大 lr_means。AC-like 和另一种亚型都有值才进入排名。
key_ligands = [
    "TGFB1", "TGFB2", "TGFB3", "CSF1", "CCL2", "CD274", "PDCD1LG2", "NECTIN2", "PVR", "SPP1", "IL10", "LGALS9", "MIF",
]
rank_rows = []
peak_ligands = set()
for ligand in key_ligands:
    ligand_table = macrophage_table.loc[macrophage_table["ligand_complex"].eq(ligand)]
    maxima = ligand_table.groupby("source", observed=True)["lr_means"].max().reindex(malignant_states).dropna()
    if "AC-like" not in maxima.index or len(maxima) < 2:
        continue

    # 已发 AC_rank 先按并列平均名次排名，再转整数。跨方法比较用未取整的峰值，避免将并列的次名写成首名。
    ac_rank = int(maxima.rank(method="average", ascending=False)["AC-like"])
    if maxima["AC-like"] == maxima.max():
        peak_ligands.add(ligand)
    rank_rows.append(
        {
            "ligand": ligand,
            "AC_rank": ac_rank,
            "n_subtypes": len(maxima),
            "AC_lr_mean": float(round(maxima["AC-like"], 2)),
            "top_subtype": maxima.idxmax(),
            "top_val": float(round(maxima.max(), 2)),
        }
    )

# 首选 07 实际得分。单独重做 LIANA 对照时，公共 scGBM 对象也能提供同一得分，无需自有队列。
current_scores = Path("03_result/07_scGBM细胞通讯与混合校正/01_通讯得分与混合校正.json")
frozen_name = "communication_analysis_results.json"
if current_scores.is_file() or (Path("data/intermediate") / frozen_name).is_file() or Path(frozen_name).is_file():
    score_path = upstream.result_or_frozen(current_scores, frozen_name)
    with score_path.open(encoding="utf-8") as stream:
        score_result = json.load(stream)
    tam_scores = score_result["tam_scores"]
else:
    scgbm = ad.read_h5ad("data/external/scGBM/adata2.h5ad", backed="r")
    with closing(scgbm.file):
        tam_scores = communication.scgbm_product_scores(scgbm, "TAM/Macrophage")

# 比较单位是配体。scGBM 的单受体乘积与 snGBM 的跨受体最大值各自保留，不能解释为同一受体复合物的重复。
ranked_ligands = {row["ligand"] for row in rank_rows}
both_highest = []
other_results = []
for pair_name, state_scores in tam_scores.items():
    if set(state_scores) != {"AC", "MES1", "OPC", "NPC"} or not all(isfinite(value) for value in state_scores.values()):
        raise ValueError(f"{pair_name} 的 scGBM 分数不能比较全部发送亚型。")
    ligand = pair_name.partition("-")[0]
    if ligand not in key_ligands:
        continue
    if ligand not in ranked_ligands:
        other_results.append(f"{pair_name}: no eligible AC-like ligand rank in the snGBM macrophage table.")
        continue

    # 各队列只在本队列内比较。缺记录不能区分表达缺失、报告限制及上游筛选。
    if state_scores["AC"] == max(state_scores.values()) and ligand in peak_ligands:
        both_highest.append(f"{pair_name}: AC reaches the highest scGBM product and snGBM ligand-level lr_means.")
    else:
        other_results.append(f"{pair_name}: AC does not reach the highest score in both within-cohort comparisons.")

# 核实排名的唯一性、名次范围和亚型，保留已有 JSON 的字段名。旧的文字字段用当次实际比较生成。
if len(ranked_ligands) != len(rank_rows):
    raise ValueError("同一 LIANA 配体重复进入排名。")
for row in rank_rows:
    valid_rank = 1 <= row["AC_rank"] <= row["n_subtypes"] <= len(malignant_states)
    if not valid_rank or row["top_subtype"] not in malignant_states:
        raise ValueError(f"配体 {row['ligand']} 的排名与有值亚型数不匹配。")
top_count = sum(row["AC_rank"] == 1 for row in rank_rows)
result = {
    "tool": f"Precomputed LIANA lr_means from a snGBM object with {cell_count} cells/nuclei.",
    "malig_to_macrophage_pairs": len(macrophage_table),
    "AC_like_rank_key_ligands": rank_rows,
    "AC_like_top1": f"{top_count}/{len(rank_rows)} key immunosuppressive ligands",
    "robust_across_toolchange": both_highest,
    "NOT_recovered_by_real_tool": other_results,
    "interpretation": (
        f"{len(both_highest)} of {len(both_highest) + len(other_results)} scGBM macrophage-target pairs in the panel "
        "also reached an AC maximum at the ligand level in snGBM. The receptor sets and cohorts can differ."
    ),
}
upstream.write_checked_json(Path("03_result/08_snGBM配体通讯排名/01_LIANA配体排名与方法对照.json"), result)
print(f"髓系配受体记录 {len(macrophage_table)} 条，AC-like 有可比排名的配体 {len(rank_rows)} 个。")
