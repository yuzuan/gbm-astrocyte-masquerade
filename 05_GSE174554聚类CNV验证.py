# 目的：用发表分析的 GSE174554 冻结分区复现主、次星形细胞簇及参照细胞的 +7/−10 CNV 阈上比例。
# 输入：data/GSE174554 的 RNA 矩阵与条码、data/external/scGBM/adata2.h5ad 的染色体注释、
#       data/intermediate 的 GSE174554_metadata.txt.gz、gse174554_cluster.npz 及同目录 gse174554_cluster_markers.csv。
# 输出：03_result/05_GSE174554聚类CNV验证/01_GSE174554逐细胞CNV.npz 及同目录的 02_GSE174554聚类CNV结果.json。
# 运行方式：从含 data 的工作目录用 Python 运行本文件。可用 --partition 指向原分区的另一份拷贝，标志表须与它同目录。
#           根目录旧包支持相同的 NPZ 和 CSV 文件名。04 的当批重聚类独立保留。

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

common = import_module("00a_基因表达与聚类")

# 选择历史分区。每一份分区只配它所在目录的标志表，默认优先使用 data/intermediate 中的冻结档。
parser = argparse.ArgumentParser(description="在原 GSE174554 分区复现历史 CNV")
parser.add_argument("--partition", type=Path, help="历史分区 NPZ，配套标志表名为 gse174554_cluster_markers.csv")
partition_option = parser.parse_args().partition
frozen_path = Path("data/intermediate/gse174554_cluster.npz")
if not frozen_path.is_file():
    frozen_path = Path("gse174554_cluster.npz")
cluster_path = frozen_path if partition_option is None else partition_option.expanduser()
marker_path = cluster_path.with_name("gse174554_cluster_markers.csv")
if not cluster_path.is_file() or not marker_path.is_file():
    raise FileNotFoundError(f"须提供成对的历史分区和冻结标志表：{cluster_path}、{marker_path}")

# 从当前 RNA 文库生成完整“文库前缀:条码”。旧档的 SAMP 和 TN 可逐行核实，同样本同组织内的条码次序无法独立回证。
genes, matrix_paths, cell_table_total = common.gse174554_libraries()
cell_ids = cell_table_total["cell_id"].to_numpy(dtype=str)
samples = cell_table_total["sample"].to_numpy(dtype=str)
with np.load(cluster_path, allow_pickle=False) as cluster_archive:
    clusters = cluster_archive["leiden"]
    tissue_labels = cluster_archive["TN"]
    if not np.array_equal(cluster_archive["SAMP"], samples):
        raise ValueError("聚类档逐行样本与当前 RNA 文库顺序不符。")
    if clusters.shape != cell_ids.shape or tissue_labels.shape != cell_ids.shape:
        raise ValueError("聚类或组织标签数与当前 RNA 细胞数不符。")
    has_cell_ids = "cell_id" in cluster_archive.files
    if has_cell_ids and not np.array_equal(cluster_archive["cell_id"], cell_ids):
        raise ValueError("分区自带的完整细胞标识与当前 RNA 条码次序不同。")

# 用另一份分区时，如果本地同时有默认冻结档，逐细胞核它的成员与组织标签，防止同号簇来自其他分区。
if partition_option is not None and frozen_path.is_file() and not cluster_path.samefile(frozen_path):
    with np.load(frozen_path, allow_pickle=False) as reference:
        if not np.array_equal(reference["leiden"], clusters) or not np.array_equal(reference["TN"], tissue_labels):
            raise ValueError("指定分区的逐细胞成员或组织标签与默认冻结分区不同。")
        if not np.array_equal(reference["SAMP"], samples):
            raise ValueError("默认冻结分区的样本顺序与指定分区不同。")

# 与 04 沿用同一组织注释键。只有当前缺少匹配的条码才填历史的 NA 组。
tissue_table = pd.read_csv("data/intermediate/GSE174554_metadata.txt.gz", sep=r"\s+", dtype=str, keep_default_na=False)
tissue_table = tissue_table.drop_duplicates(["Sample#_Barcode", "Tumor_Normal_annotation"])
tissue_lookup = tissue_table.set_index("Sample#_Barcode", verify_integrity=True)["Tumor_Normal_annotation"]
observed_tissues = cell_table_total["tissue_id"].map(tissue_lookup).fillna("NA").to_numpy(dtype=str)
if not np.array_equal(tissue_labels, observed_tissues):
    raise ValueError("冻结 TN 与按当前 RNA 条码匹配的逐行组织标签不同。")

# 核对全部簇的细胞数，再按同一份标志谱认定发表分析的星形前二簇 5、10 与少突首位簇 1。
marker_table = pd.read_csv(marker_path, index_col="cluster")
counts = pd.Series(clusters).value_counts().sort_index()
if not marker_table["n"].sort_index().equals(counts):
    raise ValueError("分区和配套标志表的逐簇成员数不同。")
if not np.isfinite(marker_table[["astro", "oligo"]].to_numpy()).all():
    raise ValueError("冻结标志表含非有限的星形或少突表达。")
primary_astro_cluster, secondary_astro_cluster = marker_table["astro"].nlargest(2).index.tolist()
oligo_cluster = int(marker_table["oligo"].idxmax())
if (primary_astro_cluster, secondary_astro_cluster, oligo_cluster) != (5, 10, 1):
    raise ValueError("分区的星形前二簇或少突首簇与原研究身份不符。")

# 染色体分母按特征表中匹配的行数计；髓系基因按原面板取每个基因最后一次出现的位置。
chromosome_genes = common.chromosome_gene_sets()
chr7_mask = np.isin(genes, list(chromosome_genes["chr7"]))
chr10_mask = np.isin(genes, list(chromosome_genes["chr10"]))
chr7_count = chr7_mask.sum()
chr10_count = chr10_mask.sum()
gene_index = {gene: i for i, gene in enumerate(genes)}
immune_indices = [gene_index[gene] for gene in common.lineage_genes["imm"] if gene in gene_index]
if not chr7_count or not chr10_count or not immune_indices:
    raise ValueError("GSE174554 基因表有染色体或髓系面板完全未匹配。")

# 逐文库对细胞行计算每万计数，按基因数取均值后 log1p。这里先汇总再取对数，与 04 的标志均值口径不同。
chr7_blocks = []
chr10_blocks = []
immune_blocks = []
for matrix_path in matrix_paths:
    cell_matrix = common.read_cell_matrix(matrix_path)
    library_scale = np.asarray(cell_matrix.sum(axis=1)).ravel() / 1e4
    library_scale[library_scale == 0] = 1e-9

    # 旧代码的染色体分母是 NumPy 计数标量；髓系分母是面板的 Python 长度，保留相同的精度转换。
    chr7_sum = np.asarray(cell_matrix[:, chr7_mask].sum(axis=1)).ravel()
    chr10_sum = np.asarray(cell_matrix[:, chr10_mask].sum(axis=1)).ravel()
    immune_sum = np.asarray(cell_matrix[:, immune_indices].sum(axis=1)).ravel()
    chr7_blocks.append(np.log1p(chr7_sum / library_scale / chr7_count))
    chr10_blocks.append(np.log1p(chr10_sum / library_scale / chr10_count))
    immune_blocks.append(np.log1p(immune_sum / library_scale / len(immune_indices)))

# 免疫表达高于第 90 百分位的细胞既用于中心化，也用于取 CNV 第 95 百分位。阈值相等者归入阴性。
immune_score = np.concatenate(immune_blocks)
myeloid = immune_score > np.quantile(immune_score, 0.90)
cnv_scores, threshold, _, _ = common.centered_cnv(
    np.concatenate(chr7_blocks), np.concatenate(chr10_blocks), myeloid, myeloid
)
if cnv_scores.shape != clusters.shape:
    raise ValueError("逐细胞 CNV 分数未覆盖完整分区。")
positive = cnv_scores > threshold
astrocytes = clusters == primary_astro_cluster
astro = common.positive_summary(positive, astrocytes)
if not astro["n"]:
    raise ValueError("历史星形胶质细胞簇 5 在分区中不存在。")

# 组织标签 NA 也计入簇总人数，按实际出现的标签报告各自分母；另保留次级星形簇和两类参照结果。
by_tissue = {}
for tissue in np.unique(tissue_labels[astrocytes]):
    by_tissue[str(tissue)] = common.positive_summary(positive, astrocytes & (tissue_labels == tissue))
if sum(group["n"] for group in by_tissue.values()) != astro["n"]:
    raise ValueError("GSE174554 各组织星形细胞数之和与簇 5 分母不同。")
secondary_astro = clusters == secondary_astro_cluster
secondary = common.positive_summary(positive, secondary_astro)
myeloid_reference = common.positive_summary(positive, myeloid)
oligo = clusters == oligo_cluster
oligo_reference = common.positive_summary(positive, oligo)

# 对外仅记录工作目录内的相对位置，外部指定的拷贝记录文件名。旧档无条码时标出重建来源。
public_partition = Path(cluster_path.name)
if cluster_path.absolute().is_relative_to(Path.cwd()) and ".." not in cluster_path.parts:
    public_partition = cluster_path.absolute().relative_to(Path.cwd())
partition_source = public_partition.as_posix()
marker_source = public_partition.with_name(marker_path.name).as_posix()
cell_id_source = "RNA library prefix and full barcode rebuilt in historical matrix order"
cell_order_check = "SAMP and TN agree row by row; barcode order within each sample and tissue is unverified"
if has_cell_ids:
    cell_id_source = "partition cell_id matched to RNA library prefix and full barcode"
    cell_order_check = "SAMP, TN and full cell_id agree row by row"

# 10 从这一份 NPZ 读取 CNV、分区及掩码，表内的每一行对应同一个 RNA 细胞。
arrays = {
    "cnv": cnv_scores,
    "leiden": clusters,
    "TN": tissue_labels,
    "SAMP": samples,
    "strict": np.asarray(threshold),
    "primary_astro_cluster": np.asarray(primary_astro_cluster),
    "secondary_astro_cluster": np.asarray(secondary_astro_cluster),
    "oligo_cluster": np.asarray(oligo_cluster),
    "astro_mask": astrocytes,
    "secondary_astro_mask": secondary_astro,
    "oligo_mask": oligo,
    "myeloid_mask": myeloid,
    "cnv_positive_mask": positive,
    "partition_source": np.asarray(partition_source),
    "marker_source": np.asarray(marker_source),
    "cell_id_source": np.asarray(cell_id_source),
    "partition_has_cell_id": np.asarray(has_cell_ids),
    "cell_order_check": np.asarray(cell_order_check),
}
if has_cell_ids:
    arrays["cell_id"] = cell_ids
else:
    arrays["reconstructed_cell_id"] = cell_ids
output_dir = Path("03_result/05_GSE174554聚类CNV验证")
common.write_checked_arrays(output_dir / "01_GSE174554逐细胞CNV.npz", arrays)

# 记录本次实际使用的分区、匹配标志表、组织标签和条码可核范围。CNV 统计保留原发表字段。
result = {
    "method": "Leiden clustering (res=0.5, harmony batch-corrected), astrocyte=cluster5",
    "strict_threshold": float(threshold),
    "astrocyte_cluster": {"cluster": primary_astro_cluster, **astro},
    "astrocyte_by_tissue": by_tissue,
    "reference_FPR": {"myeloid": myeloid_reference["cnv_pos_pct"], "oligodendrocyte": oligo_reference["cnv_pos_pct"]},
    "secondary_astrocyte_cluster": {"cluster": secondary_astro_cluster, **secondary},
    "oligodendrocyte_cluster": {"cluster": oligo_cluster, **oligo_reference},
    "partition_source": partition_source,
    "marker_source": marker_source,
    "partition_cluster_count": len(marker_table),
    "partition_has_cell_id": has_cell_ids,
    "cell_id_source": cell_id_source,
    "cell_order_check": cell_order_check,
}
common.write_checked_json(output_dir / "02_GSE174554聚类CNV结果.json", result)
print(f"GSE174554 簇 5 共 {astro['n']} 个细胞，阈上比例 {astro['cnv_pos_pct']}%。")
