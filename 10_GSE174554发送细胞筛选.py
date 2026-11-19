# 目的：在 GSE174554 的发表分区内，比较主星形细胞簇去除 CNV 阈上细胞前后的髓系通讯得分。
# 输入：data/GSE174554 的矩阵、特征和条码；data/intermediate/GSE174554_metadata.txt.gz 的条码和组织。
# 默认只取 03_result/05_GSE174554聚类CNV验证/01_GSE174554逐细胞CNV.npz 的同分区标签、行键、阈值和掩码。
# 显式旧模式用 --frozen-dir 指定同时含 gse174554_cluster.npz、gse174554_cnv_leiden.npz、
# gse174554_cluster_markers.csv 的目录。旧归档没有条码，同一样本同组织内的条码顺序依赖历史归档。
# 输出：03_result/10_GSE174554发送细胞筛选/01_筛选前后通讯得分.json
# 运行方式：在含 data 和 05 产物的工作目录运行 python 10_GSE174554发送细胞筛选.py；使用冻结三件套时加 --frozen-dir 目录。

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

communication = import_module("00a_细胞通讯")
upstream = import_module("00a_基因表达与聚类")

# 默认分析只读 05 的单一逐细胞归档。只有明确指定冻结目录，才读取同目录的旧 CNV、聚类和标志表。
parser = argparse.ArgumentParser(description="用 GSE174554 已选分区分析发送群通讯")
parser.add_argument("--frozen-dir", type=Path, help="含原始配套聚类 NPZ、逐细胞 CNV NPZ 和簇标志 CSV 的目录")
frozen_dir = parser.parse_args().frozen_dir
cnv_path = Path("03_result/05_GSE174554聚类CNV验证/01_GSE174554逐细胞CNV.npz")
if frozen_dir is not None:
    cnv_path = frozen_dir / "gse174554_cnv_leiden.npz"

# 所有 RNA 文库沿用原始排序。逐条核实样本与组织，再按输入证据区分原档条码和按文库重建的行键。
gene_names, matrix_paths, cell_table_total = upstream.gse174554_libraries()
genes = pd.Index(gene_names)
cell_ids = cell_table_total["cell_id"].to_numpy(dtype=str)
samples = cell_table_total["sample"].to_numpy(dtype=str)
tissue_table = pd.read_csv("data/intermediate/GSE174554_metadata.txt.gz", sep=r"\s+", dtype=str, keep_default_na=False)
tissue_table = tissue_table.drop_duplicates(["Sample#_Barcode", "Tumor_Normal_annotation"])
tissue_lookup = tissue_table.set_index("Sample#_Barcode", verify_integrity=True)["Tumor_Normal_annotation"]
observed_tissues = cell_table_total["tissue_id"].map(tissue_lookup).fillna("NA").to_numpy(dtype=str)

# 两种输入都从选定的 CNV 归档取 CNV、组织和分区成员。新版另有明确的簇身份与双方可核的行键。
with np.load(cnv_path, allow_pickle=False) as archive:
    cnv = archive["cnv"]
    threshold = float(archive["strict"].item())
    leiden = archive["leiden"]
    tissues = archive["TN"]
    if cnv.shape != cell_ids.shape or leiden.shape != cell_ids.shape or not np.array_equal(tissues, observed_tissues):
        raise ValueError("逐细胞 CNV、分区或组织与 RNA 文库的行数或组织次序不符。")
    if not np.isfinite(cnv).all() or not np.isfinite(threshold):
        raise ValueError("归档中的 CNV 分数或阈值缺失。")
    if frozen_dir is None:
        primary_cluster = int(archive["primary_astro_cluster"].item())
        sender = archive["astro_mask"]
        positive = archive["cnv_positive_mask"]
        receiver_05 = archive["myeloid_mask"]
        has_ids = bool(archive["partition_has_cell_id"].item())
        row_key = "cell_id" if has_ids else "reconstructed_cell_id"
        available_keys = {"cell_id", "reconstructed_cell_id"}.intersection(archive.files)
        if available_keys != {row_key} or not np.array_equal(archive[row_key], cell_ids):
            raise ValueError("05 记录的行键来源与字段或当前 RNA 条码顺序不符。")
        if not np.array_equal(archive["SAMP"], samples):
            raise ValueError("05 的逐行样本与当前 RNA 文库不符。")
        if not np.array_equal(sender, leiden == primary_cluster) or not np.array_equal(positive, cnv > threshold):
            raise ValueError("05 的发送群或阈上掩码与同档的实际簇号、CNV 分数和阈值不符。")
        if sender.dtype != bool or positive.dtype != bool or receiver_05.dtype != bool:
            raise ValueError("05 的细胞群掩码必须为布尔向量。")
        partition_source = str(archive["partition_source"].item())
        marker_source = str(archive["marker_source"].item())
        cell_order_check = str(archive["cell_order_check"].item())

# 显式旧模式只能配同目录的两份原归档和标志表。按全簇细胞数及星形和少突面板的名次回查历史簇号。
if frozen_dir is not None:
    cluster_path = frozen_dir / "gse174554_cluster.npz"
    marker_path = frozen_dir / "gse174554_cluster_markers.csv"
    with np.load(cluster_path, allow_pickle=False) as partition:
        if not np.array_equal(partition["SAMP"], samples) or not np.array_equal(partition["TN"], tissues):
            raise ValueError("旧分区样本或组织顺序与此次 RNA 和 CNV 归档不符。")
        if not np.array_equal(partition["leiden"], leiden):
            raise ValueError("旧 CNV 归档与配套分区的逐细胞簇号不符。")
    marker_table = pd.read_csv(marker_path, index_col="cluster")
    cluster_counts = pd.Series(leiden).value_counts().sort_index()
    if not marker_table["n"].sort_index().equals(cluster_counts):
        raise ValueError("旧标志表的逐簇人数与指定分区不符。")
    if not np.isfinite(marker_table[["astro", "oligo"]].to_numpy()).all():
        raise ValueError("旧标志表的星形或少突得分缺失。")
    astro_clusters = marker_table["astro"].nlargest(2).index.tolist()
    if astro_clusters != [5, 10] or marker_table["oligo"].idxmax() != 1:
        raise ValueError("指定分区的标志谱与发表的主、次星形簇和少突簇不符。")
    primary_cluster = int(astro_clusters[0])
    sender = leiden == primary_cluster
    positive = cnv > threshold
    partition_source = cluster_path.name
    marker_source = marker_path.name
    cell_order_check = "SAMP, TN and Leiden matched. Barcode order within each sample and tissue is unverified."

# 通讯取原来的七种配体和八种受体。髓系标记集与 GSE211376 的定义不同，按此队列原方案选取存在的基因。
ligands = ["ANXA1", "LGALS1", "TGFB1", "TGFB2", "CSF1", "CCL2", "CD274"]
receptors = ["FPR1", "FPR3", "PTPRC", "TGFBR1", "TGFBR2", "CSF1R", "CCR2", "PDCD1"]
measured = [gene for gene in ligands + receptors if gene in genes]
positions = genes.get_indexer(measured)
immune_genes = [gene for gene in upstream.lineage_genes["imm"] if gene in genes]
if not immune_genes:
    raise ValueError("没有可用的髓系标记基因。")
immune_positions = genes.get_indexer(immune_genes)

# 按冻结分析的 RNA 文库排序，在同一矩阵上取出通讯基因和髓系标记。
expression_blocks = []
immune_blocks = []
for matrix_path in matrix_paths:

    # 从同一细胞的全部基因计算每万计数除数。配受体逐基因取对数；髓系分数先取标记计数均值，再取对数。
    cell_matrix = upstream.read_cell_matrix(matrix_path)
    total_counts = np.asarray(cell_matrix.sum(axis=1)).ravel()
    size_factors = communication.library_size_factors(total_counts)
    pair_matrix = cell_matrix[:, positions].toarray() / size_factors[:, None]
    expression_blocks.append(np.log1p(pair_matrix).astype(np.float32))
    immune_counts = np.asarray(cell_matrix[:, immune_positions].sum(axis=1)).ravel()
    immune_blocks.append(np.log1p(immune_counts / size_factors / len(immune_genes)).astype(np.float32))

expression_table_total = pd.DataFrame(np.vstack(expression_blocks), columns=measured)
immune_score = np.concatenate(immune_blocks)
if len(expression_table_total) != len(leiden) or immune_score.shape != leiden.shape:
    raise ValueError("纳入矩阵的细胞总数与逐细胞标签不符。")

# 使用已核定的主星形簇，去掉同一分区的 CNV 阈上细胞。接收群按髓系分数重算，并与新版 05 的掩码逐行比较。
clean = sender & ~positive
receiver = immune_score > np.quantile(immune_score, 0.90)
if frozen_dir is None and not np.array_equal(receiver, receiver_05):
    raise ValueError("本次表达矩阵得到的髓系接收群与 05 同一分区的掩码不符。")
if not sender.any() or not clean.any() or not receiver.any():
    raise ValueError("至少一个发送群或接收群为空，通讯均值无法估计。")
receptor_means = {}
for gene in receptors:
    if gene in measured:
        receptor_means[gene] = float(expression_table_total.loc[receiver, gene].mean())

# 旧 GSE174554 口径把面板中缺基因的一端按 0 计分。百分比按未取位乘积计算，原始分数为零时按旧约定记 0。
pairs = [
    ("ANXA1", "FPR1"), ("ANXA1", "FPR3"), ("LGALS1", "PTPRC"), ("TGFB1", "TGFBR1"), ("TGFB2", "TGFBR1"),
    ("TGFB2", "TGFBR2"), ("CSF1", "CSF1R"), ("CCL2", "CCR2"), ("CD274", "PDCD1"),
]
pair_scores = {}
for ligand, receptor in pairs:
    ligand_before = ligand_after = 0.0
    if ligand in measured:
        ligand_before = float(expression_table_total.loc[sender, ligand].mean())
        ligand_after = float(expression_table_total.loc[clean, ligand].mean())
    before = ligand_before * receptor_means.get(receptor, 0.0)
    after = ligand_after * receptor_means.get(receptor, 0.0)
    change = (after - before) / before * 100 if before > 0 else 0.0
    pair_scores[f"{ligand}->{receptor}"] = {
        "before": round(before, 4),
        "after": round(after, 4),
        "pct_change": round(change, 1),
        "ligand_before": round(ligand_before, 4),
        "ligand_after": round(ligand_after, 4),
    }

# 将实际超过阈值的发送细胞数与两次分母之差核对，并确认每对配受体均有记录。
sender_count = int(sender.sum())
clean_count = int(clean.sum())
if sender_count - clean_count != int((sender & (cnv > threshold)).sum()) or len(pair_scores) != len(pairs):
    raise ValueError("发送群分母或配受体数不符。")
sender_method = f"Leiden astrocyte cluster{primary_cluster} sender"
result = {
    "dataset": "GSE174554",
    "method": f"{sender_method}; myeloid(immune-high) receiver; remove CNV+ from sender",
    "primary_astro_cluster": primary_cluster,
    "partition_source": partition_source,
    "marker_source": marker_source,
    "cell_order_check": cell_order_check,
    "n_astro_before": sender_count,
    "n_astro_after_clean": clean_count,
    "strict_threshold": threshold,
    "pairs": pair_scores,
}
upstream.write_checked_json(Path("03_result/10_GSE174554发送细胞筛选/01_筛选前后通讯得分.json"), result)
print(f"发送细胞 {sender_count} 个，按 CNV 阈值保留 {clean_count} 个。")
