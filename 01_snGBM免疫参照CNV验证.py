# 目的：在 snGBM 冻结的 Leiden 簇 0 中，统计超过免疫参照 +7/−10 阈值的细胞比例。
# 输入：data/external/scGBM/adata2.h5ad 的 var/chromosome 和
#       data/external/snGBM/results/1_snRNA_processing/adata1.h5ad 的 X、var_names、obs/leiden_0.5。
#       自有队列字段优先用 data/infercnv_cell_scores.csv 和 data/cell_metadata.csv，缺两表时可读
#       data/intermediate/noncircular_cnv_validation.json 的冻结 own_cohort。
# 输出：03_result/01_snGBM免疫参照CNV验证/01_snGBM免疫参照CNV结果.json。
# 运行方式：在仓库根目录运行 python 01_snGBM免疫参照CNV验证.py，沿用原分析的 Python/AnnData 环境。

import json
from importlib import import_module
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

common = import_module("00a_基因表达与聚类")

# X 是已处理的表达量。保留冻结的簇号指代，不用作者的恶性标签决定哪一簇进入比较。
sn_data = ad.read_h5ad("data/external/snGBM/results/1_snRNA_processing/adata1.h5ad", backed="r")
cell_table = sn_data.obs[["leiden_0.5"]].copy()
clusters = cell_table["leiden_0.5"].astype(str)
if not {"0", "2", "5", "10", "13"}.issubset(set(clusters)):
    raise ValueError("snGBM 缺少历史目标簇或参照簇，请核对 Leiden 分辨率和输入版本。")


def mean_expression(data, genes):
    """逐细胞对命中的基因取算术均值。分块只限制内存，保留基因顺序和原表达精度。空面板会报错。"""
    positions = data.var_names.get_indexer(genes)
    positions = positions[positions >= 0]
    if not positions.size:
        raise ValueError("snGBM 表达矩阵未包含当前面板的任何基因。")

    # 先取连续细胞行，再在内存中取基因列，避免 AnnData 的后备 CSR 在离散列索引下整矩阵回读。
    block_means = []
    for start in range(0, data.n_obs, 1024):
        block = data[start:start + 1024, :].X
        block = block[:, positions]
        if sparse.issparse(block):
            block = block.toarray()
        block_means.append(np.asarray(block).mean(axis=1))
    means = np.concatenate(block_means)
    return means


# 三套面板分别衡量星形、髓系和少突表达。后两者用于描述簇，不参与严格 CNV 阈值运算。
try:
    cell_table["astro"] = mean_expression(
        sn_data, ["GFAP", "AQP4", "SLC1A3", "ALDH1L1", "GJA1", "CLU", "AGT", "SLC1A2", "GLUL"]
    )
    cell_table["immune"] = mean_expression(sn_data, ["PTPRC", "AIF1", "CD68", "P2RY12", "CX3CR1", "ITGAM", "C1QB"])
    cell_table["oligo"] = mean_expression(sn_data, ["MBP", "PLP1", "MOG", "MAG", "MOBP"])

    # 从独立 scGBM 注释取得染色体基因。按基因名排序后求均值，对齐旧分析的累加次序。
    chromosome_genes = common.chromosome_gene_sets()
    cell_table["chr7"] = mean_expression(sn_data, sorted(chromosome_genes["chr7"]))
    cell_table["chr10"] = mean_expression(sn_data, sorted(chromosome_genes["chr10"]))
finally:
    sn_data.file.close()

# 原分析固定簇 0。显示它在当前各簇中的实际星形面板名次，同时保留另两个面板的平均分。
marker_table = cell_table.groupby(clusters, observed=True)[["astro", "immune", "oligo"]].mean()
if not np.isfinite(marker_table.to_numpy()).all():
    raise ValueError("snGBM 簇标志基因均值有非有限值。")
astro_rank = int(marker_table["astro"].rank(ascending=False, method="min").at["0"])
marker_means = marker_table.loc["0"]
cluster_description = (
    f"Leiden 0; astro panel mean {marker_means['astro']:.3f} (rank {astro_rank}/{len(marker_table)}), "
    f"immune {marker_means['immune']:.3f}, oligo {marker_means['oligo']:.3f}"
)

# 巨噬簇 5 提供两条染色体的中心。严格阈值取簇 5 和 NK/T 簇 10 合并后的第 95 百分位，等于阈值算阴性。
macrophages = clusters.eq("5")
chr7_gain = cell_table["chr7"] - cell_table.loc[macrophages, "chr7"].mean()
chr10_change = cell_table["chr10"] - cell_table.loc[macrophages, "chr10"].mean()
cnv_scores = chr7_gain - chr10_change
if not np.isfinite(cnv_scores).all():
    raise ValueError("snGBM 目标或参照细胞的 CNV 分数缺失。")
threshold = np.quantile(cnv_scores[clusters.isin(["5", "10"])], 0.95)
strict_positive = cnv_scores > threshold

# 历史记录用少突簇 2、周细胞簇 13 各自作分母。旧字段 FPR 表示参照组阈上率，周细胞单列。
reference_ids = {"Macrophage": "5", "NK_T": "10", "Oligodendrocyte": "2", "Pericyte": "13"}
reference_rates = {}
for label, cluster_id in reference_ids.items():
    reference = common.positive_summary(strict_positive, clusters.eq(cluster_id))
    reference_rates[label] = reference["cnv_pos_pct"]
astrocytes = clusters.eq("0")
astro = common.positive_summary(strict_positive, astrocytes)
loose_macrophage = common.positive_summary((chr7_gain > 0) | (chr10_change < 0), macrophages)

# 旧 JSON 同时登记自有队列与 snGBM。没有两份自有输入时只转录明示的冻结来源，未提供的数字留空。
score_path = Path("data/infercnv_cell_scores.csv")
metadata_path = Path("data/cell_metadata.csv")
frozen_path = Path("data/intermediate/noncircular_cnv_validation.json")
if score_path.is_file() or metadata_path.is_file():
    own = common.inhouse_cnv_summary(score_path, metadata_path)
    own_source = "Computed from the in-house cell scores and condition metadata."
elif frozen_path.is_file():
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    own = frozen["own_cohort"].copy()
    own.setdefault("strict_matched_pct", None)
    own.setdefault("peritumoral_astro_strict_pct", None)
    own.setdefault("loose_cnv_pos_pct", None)
    own.setdefault("strict_ref_FPR", dict.fromkeys(["Myeloid", "T_NK", "Oligodendrocyte"]))
    own_source = "Transcribed from data/intermediate/noncircular_cnv_validation.json."
else:
    own = {
        "loose_cnv_pos_pct": None,
        "strict_matched_pct": None,
        "n": None,
        "strict_ref_FPR": dict.fromkeys(["Myeloid", "T_NK", "Oligodendrocyte"]),
        "peritumoral_astro_strict_pct": None,
    }
    own_source = "In-house cell inputs and frozen summary unavailable."

# 两个队列各用自己的免疫参照分位数。比较文本随输入生成，不能从表达代理分数推出每个细胞的恶性真值。
concordance = f"snGBM cluster 0: {astro['cnv_pos_pct']:.1f}% above its immune-reference threshold."
if own["strict_matched_pct"] is not None:
    concordance += f" In-house GBM astrocytes: {own['strict_matched_pct']:.1f}% above their own threshold."
result = {
    "method": "Frozen marker-defined Leiden cluster; macrophage-centered chr7 minus chr10 expression.",
    "snGBM_independent": {
        "astrocyte_cluster": cluster_description,
        "n": astro["n"],
        "strict_CNVpos_pct": astro["cnv_pos_pct"],
        "cnv_score_median": round(float(cnv_scores[astrocytes].median()), 4),
        "reference_FPR": {label: rate for label, rate in reference_rates.items() if label != "Pericyte"},
        "reference_FPR_full": reference_rates,
        "note_pericyte": f"Pericyte cluster 13: {reference_rates['Pericyte']:.1f}% above the strict threshold.",
        "strict_threshold": float(threshold),
    },
    "own_cohort": own,
    "own_cohort_source": own_source,
    "concordance": concordance,
    "reviewer_point_addressed": "Cluster membership precedes independent expression-based CNV scoring.",
    "loose_cnv_pos_FPR_warning": (
        f"The loose chr7-gain OR chr10-loss definition calls {loose_macrophage['cnv_pos_pct']:.1f}% of macrophages."
    ),
}

# 持久化前核对目标簇分母与分簇计数；写盘函数对完整 JSON 再读一次。
if result["snGBM_independent"]["n"] != clusters.value_counts().at["0"]:
    raise ValueError("snGBM 汇总分母与簇 0 的细胞数不一致。")
common.write_checked_json(Path("03_result/01_snGBM免疫参照CNV验证/01_snGBM免疫参照CNV结果.json"), result)
print(f"snGBM 簇 0 共 {astro['n']} 个细胞，免疫参照阈上比例 {astro['cnv_pos_pct']:.1f}%。")
