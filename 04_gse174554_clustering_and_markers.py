# Purpose: Integrate GSE174554 RNA libraries in their original order with Harmony, run Leiden clustering, and report
#          mean expression for seven marker panels per cluster.
# Inputs: *_matrix.mtx.gz, *_features.tsv.gz, and *_barcodes.tsv.gz in data/GSE174554, plus Sample#_Barcode and
#         Tumor_Normal_annotation in data/intermediate/GSE174554_metadata.txt.gz.
# Outputs: 03_result/04_gse174554_clustering_and_markers/01_gse174554_clusters.npz and 02_gse174554_cluster_markers.csv
#          beside it record the new partition. Stage 05 reproduces historical CNV from the frozen partition.
# Usage: Run python 04_gse174554_clustering_and_markers.py from the repository root with the original sparse-matrix
#        and clustering libraries.

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

common = import_module("00a_expression_and_clustering")

# Each retained RNA barcode has one row in concatenated matrix order. Library batches from the same SF sample retain
# distinct cell identifiers.
genes, matrix_paths, cell_table_total = common.gse174554_libraries()
tissue_table = pd.read_csv("data/intermediate/GSE174554_metadata.txt.gz", sep=r"\s+", dtype=str, keep_default_na=False)
tissue_table = tissue_table.drop_duplicates(["Sample#_Barcode", "Tumor_Normal_annotation"])
if not tissue_table["Sample#_Barcode"].is_unique:
    raise ValueError("A GSE174554 tissue-annotation barcode has conflicting labels.")
tissue_lookup = tissue_table.set_index("Sample#_Barcode")["Tumor_Normal_annotation"]
tissue_labels = cell_table_total["tissue_id"].map(tissue_lookup).fillna("NA").to_numpy(dtype=str)
samples = cell_table_total["sample"].to_numpy(dtype=str)

# Stack cell-by-gene library matrices in original order as CSR. Retain all-zero cells by replacing only their divisor.
matrices = []
for matrix_path in matrix_paths:
    matrices.append(common.read_cell_matrix(matrix_path))
expression_matrix = sparse.vstack(matrices).tocsr()
del matrices
if expression_matrix.shape != (len(cell_table_total), len(genes)):
    raise ValueError("The combined expression matrix does not match the retained barcode and shared gene counts.")
library_sizes = np.asarray(expression_matrix.sum(axis=1)).ravel()
library_sizes[library_sizes == 0] = 1
expression_matrix = expression_matrix.multiply(1e4 / library_sizes[:, None]).tocsr()
expression_matrix.data = np.log1p(expression_matrix.data)

# Rank genes by variance / (mean + 1e-8) of log1p counts per 10,000. Restore the top 2,000 to their original order,
# then standardize each gene.
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

# Extract 50 components after clipping standardized values at +/-10. Both SVD and Harmony use random state 0.
pca = TruncatedSVD(n_components=50, random_state=0).fit_transform(variable_matrix)
del variable_matrix
batch_table = pd.DataFrame({"sample": samples})
harmony = harmonypy.run_harmony(pca, batch_table, ["sample"], max_iter_harmony=10, random_state=0)
coordinates = harmony.Z_corr.T
if coordinates.shape != pca.shape:
    coordinates = harmony.Z_corr
if coordinates.shape != pca.shape or not np.isfinite(coordinates).all():
    raise ValueError("Harmony coordinates have the wrong cell-by-component shape or contain nonfinite values.")
del pca, harmony

# Query 15 Euclidean neighbors per cell. Passing the coordinates explicitly includes self-neighbors as in the original
# search. Simplify the graph to remove self-loops and duplicate edges.
neighbors = NearestNeighbors(n_neighbors=15, metric="euclidean", n_jobs=-1).fit(coordinates)
knn = neighbors.kneighbors_graph(coordinates, mode="connectivity")
source, target = knn.nonzero()
graph = ig.Graph(n=len(samples), edges=list(zip(source.tolist(), target.tolist())), directed=False)
graph.simplify()

# Use the original RBConfiguration partition, resolution 0.5 and seed 0. The resulting cluster IDs can be compared
# with marker profiles from the frozen partition.
partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, resolution_parameter=0.5, seed=0)
clusters = np.asarray(partition.membership)
if clusters.shape != (len(samples),):
    raise ValueError("Leiden did not assign exactly one label to each retained cell.")

# For each cluster and panel, average expression across all cells and matched genes. For duplicate gene names,
# the historical index selects the last occurrence.
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
        raise ValueError(f"No GSE174554 genes match the {name} panel.")

# Slice each cluster once, count its members, and write the seven panels in historical order.
marker_rows = []
for cluster in np.unique(clusters):
    selected = clusters == cluster
    cluster_matrix = expression_matrix[selected]
    row = {"cluster": int(cluster), "n": int(selected.sum())}
    for name, indices in panel_indices.items():
        row[name] = float(cluster_matrix[:, indices].mean())
    marker_rows.append(row)
marker_table = pd.DataFrame(marker_rows)

# Check cluster denominators against membership and require finite panel means. Keep the observed cluster IDs
# and counts for comparison with the archived partition.
member_counts = pd.Series(clusters).value_counts().sort_index()
reported_counts = marker_table.set_index("cluster")["n"]
if not reported_counts.equals(member_counts) or not np.isfinite(marker_table[list(panels)].to_numpy()).all():
    raise ValueError("Marker-table counts differ from cluster membership, or a panel mean is nonfinite.")

# Preserve the original NPZ fields and dtypes. Add cell_id (library prefix and full barcode) for per-cell alignment.
arrays = {
    "leiden": clusters,
    "TN": tissue_labels,
    "SAMP": samples,
    "hvg": variable_indices,
    "Z": coordinates.astype(np.float32),
    "cell_id": cell_table_total["cell_id"].to_numpy(dtype=str),
}
output_dir = Path("03_result/04_gse174554_clustering_and_markers")
common.write_checked_arrays(output_dir / "01_gse174554_clusters.npz", arrays)

# Read all historical ASCII columns back at round-trip floating-point precision and check every row before publication.
marker_path = output_dir / "02_gse174554_cluster_markers.csv"
with TemporaryDirectory(dir=output_dir) as temporary:
    draft = Path(temporary) / marker_path.name
    marker_table.to_csv(draft, index=False)
    stored_table = pd.read_csv(draft, float_precision="round_trip")
    pd.testing.assert_frame_equal(stored_table, marker_table, check_exact=True)
    draft.replace(marker_path)
print(
    f"GSE174554 has {len(clusters)} cells in {len(marker_table)} clusters, "
    f"with {np.sum(tissue_labels == 'NA')} unmatched tissue labels."
)
