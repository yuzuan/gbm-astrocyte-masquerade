# Purpose: Reproduce +7/-10 CNV exceedance rates for the primary and secondary astrocyte clusters and reference cells
#          in the published, frozen GSE174554 partition.
# Inputs: RNA matrices and barcodes in data/GSE174554, chromosome annotations in data/external/scGBM/adata2.h5ad, and
#         GSE174554_metadata.txt.gz, gse174554_cluster.npz, and gse174554_cluster_markers.csv in data/intermediate.
# Outputs: 03_result/05_gse174554_cluster_cnv/01_gse174554_cell_cnv.npz and 02_gse174554_cluster_cnv.json beside it.
# Usage: Run python 05_gse174554_cluster_cnv.py from the directory containing data. Use --partition for a copy of the
#        original partition, with its marker table alongside it.
#        Legacy packages support the same NPZ and CSV names at the root. Stage 04 produces a separate new partition.

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

common = import_module("00a_expression_and_clustering")

# Select the historical partition, preferring the frozen file in data/intermediate. Use its directory's marker table.
parser = argparse.ArgumentParser(description="Reproduce historical CNV rates in the original GSE174554 partition.")
parser.add_argument(
    "--partition", type=Path, help="Historical partition NPZ located alongside gse174554_cluster_markers.csv."
)
partition_option = parser.parse_args().partition
frozen_path = Path("data/intermediate/gse174554_cluster.npz")
if not frozen_path.is_file():
    frozen_path = Path("gse174554_cluster.npz")
cluster_path = frozen_path if partition_option is None else partition_option.expanduser()
marker_path = cluster_path.with_name("gse174554_cluster_markers.csv")
if not cluster_path.is_file() or not marker_path.is_file():
    raise FileNotFoundError(f"Missing historical partition or paired marker table: {cluster_path}, {marker_path}")

# Reconstruct library-prefix:barcode from the RNA inputs. The historical SAMP and TN can be checked row by row.
# Without archived barcodes, within-sample, within-tissue barcode order cannot be independently verified.
genes, matrix_paths, cell_table_total = common.gse174554_libraries()
cell_ids = cell_table_total["cell_id"].to_numpy(dtype=str)
samples = cell_table_total["sample"].to_numpy(dtype=str)
with np.load(cluster_path, allow_pickle=False) as cluster_archive:
    clusters = cluster_archive["leiden"]
    tissue_labels = cluster_archive["TN"]
    if not np.array_equal(cluster_archive["SAMP"], samples):
        raise ValueError("Per-row samples in the cluster archive differ from the current RNA library order.")
    if clusters.shape != cell_ids.shape or tissue_labels.shape != cell_ids.shape:
        raise ValueError("The number of cluster or tissue labels differs from the current RNA cell count.")
    has_cell_ids = "cell_id" in cluster_archive.files
    if has_cell_ids and not np.array_equal(cluster_archive["cell_id"], cell_ids):
        raise ValueError("Full cell identifiers in the partition differ from the current RNA barcode order.")

# When using a copy with a local frozen reference, compare membership and tissue labels cell by cell. Cluster numbers
# alone cannot identify the historical partition.
if partition_option is not None and frozen_path.is_file() and not cluster_path.samefile(frozen_path):
    with np.load(frozen_path, allow_pickle=False) as reference:
        if not np.array_equal(reference["leiden"], clusters) or not np.array_equal(reference["TN"], tissue_labels):
            raise ValueError("Per-cell membership or tissue labels differ from the default frozen partition.")
        if not np.array_equal(reference["SAMP"], samples):
            raise ValueError("The frozen reference and the specified partition have different sample orders.")

# Use the same tissue-annotation key as stage 04. Assign the historical NA label to barcodes without a tissue match.
tissue_table = pd.read_csv("data/intermediate/GSE174554_metadata.txt.gz", sep=r"\s+", dtype=str, keep_default_na=False)
tissue_table = tissue_table.drop_duplicates(["Sample#_Barcode", "Tumor_Normal_annotation"])
tissue_lookup = tissue_table.set_index("Sample#_Barcode", verify_integrity=True)["Tumor_Normal_annotation"]
observed_tissues = cell_table_total["tissue_id"].map(tissue_lookup).fillna("NA").to_numpy(dtype=str)
if not np.array_equal(tissue_labels, observed_tissues):
    raise ValueError("Frozen TN differs from the tissue labels matched to the current RNA barcodes in row order.")

# Verify every cluster size. Rank this partition's markers to identify the published primary and secondary astrocyte
# clusters (5, 10) and the top oligodendrocyte cluster (1).
marker_table = pd.read_csv(marker_path, index_col="cluster")
counts = pd.Series(clusters).value_counts().sort_index()
if not marker_table["n"].sort_index().equals(counts):
    raise ValueError("Cluster membership counts differ between the partition and its paired marker table.")
if not np.isfinite(marker_table[["astro", "oligo"]].to_numpy()).all():
    raise ValueError("The frozen marker table contains nonfinite astrocyte or oligodendrocyte expression.")
primary_astro_cluster, secondary_astro_cluster = marker_table["astro"].nlargest(2).index.tolist()
oligo_cluster = int(marker_table["oligo"].idxmax())
if (primary_astro_cluster, secondary_astro_cluster, oligo_cluster) != (5, 10, 1):
    raise ValueError("The two top astrocyte clusters or top oligodendrocyte cluster differ from the original study.")

# Chromosome denominators count matching feature rows. The original myeloid panel uses each gene's last occurrence.
chromosome_genes = common.chromosome_gene_sets()
chr7_mask = np.isin(genes, list(chromosome_genes["chr7"]))
chr10_mask = np.isin(genes, list(chromosome_genes["chr10"]))
chr7_count = chr7_mask.sum()
chr10_count = chr10_mask.sum()
gene_index = {gene: i for i, gene in enumerate(genes)}
immune_indices = [gene_index[gene] for gene in common.lineage_genes["imm"] if gene in gene_index]
if not chr7_count or not chr10_count or not immune_indices:
    raise ValueError("A chromosome or the myeloid panel has no matches in the GSE174554 feature table.")

# Within each library and cell, scale summed counts to 10,000, divide by the number of genes, then take log1p.
# Stage 04 averages marker expression after taking the logarithm.
chr7_blocks = []
chr10_blocks = []
immune_blocks = []
for matrix_path in matrix_paths:
    cell_matrix = common.read_cell_matrix(matrix_path)
    library_scale = np.asarray(cell_matrix.sum(axis=1)).ravel() / 1e4
    library_scale[library_scale == 0] = 1e-9

    # Preserve the original precision conversions: chromosome denominators are NumPy count scalars and the myeloid
    # denominator is the Python length of the matched panel.
    chr7_sum = np.asarray(cell_matrix[:, chr7_mask].sum(axis=1)).ravel()
    chr10_sum = np.asarray(cell_matrix[:, chr10_mask].sum(axis=1)).ravel()
    immune_sum = np.asarray(cell_matrix[:, immune_indices].sum(axis=1)).ravel()
    chr7_blocks.append(np.log1p(chr7_sum / library_scale / chr7_count))
    chr10_blocks.append(np.log1p(chr10_sum / library_scale / chr10_count))
    immune_blocks.append(np.log1p(immune_sum / library_scale / len(immune_indices)))

# Cells above the immune-score 90th percentile center the CNV score and set its 95th-percentile cutoff. Ties at
# the CNV cutoff are negative.
immune_score = np.concatenate(immune_blocks)
myeloid = immune_score > np.quantile(immune_score, 0.90)
cnv_scores, threshold, _, _ = common.centered_cnv(
    np.concatenate(chr7_blocks), np.concatenate(chr10_blocks), myeloid, myeloid
)
if cnv_scores.shape != clusters.shape:
    raise ValueError("Per-cell CNV scores do not cover the full partition.")
positive = cnv_scores > threshold
astrocytes = clusters == primary_astro_cluster
astro = common.positive_summary(positive, astrocytes)
if not astro["n"]:
    raise ValueError("The historical astrocyte cluster 5 is absent from the partition.")

# Count tissue label NA in the cluster total and report each observed tissue denominator. Keep the secondary
# astrocyte cluster and both reference groups.
by_tissue = {}
for tissue in np.unique(tissue_labels[astrocytes]):
    by_tissue[str(tissue)] = common.positive_summary(positive, astrocytes & (tissue_labels == tissue))
if sum(group["n"] for group in by_tissue.values()) != astro["n"]:
    raise ValueError("GSE174554 astrocyte counts by tissue do not sum to the cluster 5 denominator.")
secondary_astro = clusters == secondary_astro_cluster
secondary = common.positive_summary(positive, secondary_astro)
myeloid_reference = common.positive_summary(positive, myeloid)
oligo = clusters == oligo_cluster
oligo_reference = common.positive_summary(positive, oligo)

# Record paths within the working directory as relative paths and external copies by filename. Record how cell IDs
# were reconstructed when the archive lacks barcodes.
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

# Stage 10 reads CNV scores, partition membership, and group masks from this one NPZ. Each row refers to one RNA cell.
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
output_dir = Path("03_result/05_gse174554_cluster_cnv")
common.write_checked_arrays(output_dir / "01_gse174554_cell_cnv.npz", arrays)

# Record the selected partition, its paired marker table, the tissue checks, and the scope of barcode verification.
# Keep the published CNV fields.
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
common.write_checked_json(output_dir / "02_gse174554_cluster_cnv.json", result)
print(f"GSE174554 cluster 5 contains {astro['n']} cells, with {astro['cnv_pos_pct']}% above the cutoff.")
