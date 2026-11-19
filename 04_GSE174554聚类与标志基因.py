# 目的：按原 RNA 文库顺序完成 GSE174554 的 Harmony 整合和 Leiden 聚类，报告每簇七组标志基因表达。
# 输入：data/GSE174554 的 *_matrix.mtx.gz、*_features.tsv.gz、*_barcodes.tsv.gz，
#       data/intermediate/GSE174554_metadata.txt.gz 的 Sample#_Barcode、Tumor_Normal_annotation。
# 输出：03_result/04_GSE174554聚类与标志基因/01_GSE174554细胞聚类.npz 和同目录的
#       02_GSE174554各簇标志基因表达.csv，记录本次重聚类；历史 CNV 另从冻结分区复现。
# 运行方式：在仓库根目录运行 python 04_GSE174554聚类与标志基因.py，沿用原 Python 稀疏矩阵与聚类库。

from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

import harmonypy
import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

common = import_module("00a_基因表达与聚类")

# 每行是一个入选 RNA 条码，次序与所有矩阵拼接后的行序一致。同一 SF 样本的不同文库批次单独保留。
genes, matrix_paths, cell_table_total = common.gse174554_libraries()
tissue_table = pd.read_csv("data/intermediate/GSE174554_metadata.txt.gz", sep=r"\s+", dtype=str, keep_default_na=False)
tissue_table = tissue_table.drop_duplicates(["Sample#_Barcode", "Tumor_Normal_annotation"])
if not tissue_table["Sample#_Barcode"].is_unique:
    raise ValueError("同一个 GSE174554 组织注释条码有互相矛盾的标签。")
tissue_lookup = tissue_table.set_index("Sample#_Barcode")["Tumor_Normal_annotation"]
tissue_labels = cell_table_total["tissue_id"].map(tissue_lookup).fillna("NA").to_numpy(dtype=str)
samples = cell_table_total["sample"].to_numpy(dtype=str)

# 逐文库转为细胞×基因，再按原顺序合成 CSR。保留全零细胞；文库总量为零时只替换除数。
matrices = []
for matrix_path in matrix_paths:
    matrices.append(common.read_cell_matrix(matrix_path))
expression_matrix = sparse.vstack(matrices).tocsr()
del matrices
if expression_matrix.shape != (len(cell_table_total), len(genes)):
    raise ValueError("合并表达矩阵与入选条码或共同基因数不符。")
library_sizes = np.asarray(expression_matrix.sum(axis=1)).ravel()
library_sizes[library_sizes == 0] = 1
expression_matrix = expression_matrix.multiply(1e4 / library_sizes[:, None]).tocsr()
expression_matrix.data = np.log1p(expression_matrix.data)

# 原分析按 log 每万表达的方差/(均值+1e-8) 排序；取前 2000 后恢复基因原序，再按列标准化。
gene_means = np.asarray(expression_matrix.mean(axis=0)).ravel()
mean_squares = np.asarray(expression_matrix.multiply(expression_matrix).mean(axis=0)).ravel()
dispersion = (mean_squares - gene_means ** 2) / (gene_means + 1e-8)
variable_indices = np.sort(np.argsort(dispersion)[-2000:])
variable_matrix = expression_matrix[:, variable_indices].toarray()
variable_matrix -= variable_matrix.mean(axis=0)
gene_sd = variable_matrix.std(axis=0)
gene_sd[gene_sd == 0] = 1
variable_matrix /= gene_sd
np.clip(variable_matrix, -10, 10, out=variable_matrix)

# 对截到 ±10 的标准化矩阵取 50 个成分。SVD 与 Harmony 都沿用原随机状态 0。
pca = TruncatedSVD(n_components=50, random_state=0).fit_transform(variable_matrix)
del variable_matrix
batch_table = pd.DataFrame({"sample": samples})
harmony = harmonypy.run_harmony(pca, batch_table, ["sample"], max_iter_harmony=10, random_state=0)
coordinates = harmony.Z_corr.T
if coordinates.shape != pca.shape:
    coordinates = harmony.Z_corr
if coordinates.shape != pca.shape or not np.isfinite(coordinates).all():
    raise ValueError("Harmony 坐标与原细胞×成分矩阵的维度不符，或含非有限值。")
del pca, harmony

# 每个细胞按欧氏距离查 15 个近邻。显式传入查询矩阵，保留原含自身的查找方式，再去自环和重复边。
neighbors = NearestNeighbors(n_neighbors=15, metric="euclidean", n_jobs=-1).fit(coordinates)
knn = neighbors.kneighbors_graph(coordinates, mode="connectivity")
source, target = knn.nonzero()
graph = ig.Graph(n=len(samples), edges=list(zip(source.tolist(), target.tolist())), directed=False)
graph.simplify()

# 使用原 RBConfiguration 分区、分辨率 0.5 和种子 0；簇编号由此次分区产生，便于对照冻结的标志谱。
partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, resolution_parameter=0.5, seed=0)
clusters = np.asarray(partition.membership)
if clusters.shape != (len(samples),):
    raise ValueError("Leiden 标签未逐一覆盖入选细胞。")

# 每簇一行，每类面板先取匹配基因，再对该簇的细胞×基因条目求均值。索引与旧代码同取末次同名基因。
panels = {
    **common.lineage_genes,
    "tumor": ["EGFR", "SOX2", "OLIG1", "OLIG2"],
    "tcell": ["CD3D", "CD3E", "CD2"],
    "endo": ["PECAM1", "VWF", "CLDN5"],
    "neuron": ["RBFOX3", "SYT1", "SNAP25"],
}
gene_index = {gene: i for i, gene in enumerate(genes)}
panel_indices = {}
for name, panel in panels.items():
    panel_indices[name] = [gene_index[gene] for gene in panel if gene in gene_index]
    if not panel_indices[name]:
        raise ValueError(f"GSE174554 没有匹配 {name} 面板的基因。")

# 每个簇只切片一次后，按历史次序写入七组面板；细胞数按实际成员计数。
marker_rows = []
for cluster in np.unique(clusters):
    selected = clusters == cluster
    cluster_matrix = expression_matrix[selected]
    row = {"cluster": int(cluster), "n": int(selected.sum())}
    for name, indices in panel_indices.items():
        row[name] = float(cluster_matrix[:, indices].mean())
    marker_rows.append(row)
marker_table = pd.DataFrame(marker_rows)

# 逐簇核对表中分母与分区标签，并要求每个面板均有有限结果。保留实际簇号和分母供与归档分区比较。
member_counts = pd.Series(clusters).value_counts().sort_index()
reported_counts = marker_table.set_index("cluster")["n"]
if not reported_counts.equals(member_counts) or not np.isfinite(marker_table[list(panels)].to_numpy()).all():
    raise ValueError("标志基因表与分区的逐簇细胞数不符，或面板表达含非有限值。")

# 保留原 NPZ 的全部字段和类型，另外保存文库前缀和完整条码组成的 cell_id，供下游逐细胞核对。
arrays = {
    "leiden": clusters,
    "TN": tissue_labels,
    "SAMP": samples,
    "hvg": variable_indices,
    "Z": coordinates.astype(np.float32),
    "cell_id": cell_table_total["cell_id"].to_numpy(dtype=str),
}
output_dir = Path("03_result/04_GSE174554聚类与标志基因")
common.write_checked_arrays(output_dir / "01_GSE174554细胞聚类.npz", arrays)

# 标志表各列是历史 ASCII 字段。按可往返的浮点精度回读，逐行核对全部列后发布。
marker_path = output_dir / "02_GSE174554各簇标志基因表达.csv"
with TemporaryDirectory(dir=output_dir) as temporary:
    draft = Path(temporary) / marker_path.name
    marker_table.to_csv(draft, index=False)
    stored_table = pd.read_csv(draft, float_precision="round_trip")
    pd.testing.assert_frame_equal(stored_table, marker_table, check_exact=True)
    draft.replace(marker_path)
print(f"GSE174554 共 {len(clusters)} 个细胞、{len(marker_table)} 个簇，组织未匹配 {np.sum(tissue_labels == 'NA')} 个。")
