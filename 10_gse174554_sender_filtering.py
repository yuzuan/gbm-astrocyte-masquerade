# Purpose: Compare myeloid signaling scores before and after removing cells above the CNV threshold from the primary
# astrocyte cluster of the published GSE174554 partition.
# Inputs: matrices, features and barcodes in data/GSE174554; barcodes and tissue labels in
# data/intermediate/GSE174554_metadata.txt.gz.
# By default, take partition labels, row keys, threshold and masks from
# 03_result/05_gse174554_cluster_cnv/01_gse174554_cell_cnv.npz.
# In legacy mode, --frozen-dir points to the directory containing gse174554_cluster.npz, gse174554_cnv_leiden.npz
# and gse174554_cluster_markers.csv. The older archives lack barcodes; order within a sample and tissue depends on the
# archived ordering.
# Output: 03_result/10_gse174554_sender_filtering/01_sender_filtering_scores.json
# Run python 10_gse174554_sender_filtering.py from the directory with data and the stage 05 output. For the three
# frozen files, add --frozen-dir DIRECTORY.

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

communication = import_module("00a_cell_communication")
upstream = import_module("00a_expression_and_clustering")

# Read the single stage 05 cell archive by default. Read the legacy CNV, partition and marker files together only when
# a frozen directory is explicitly requested.
parser = argparse.ArgumentParser(description="Analyze sender signaling in the selected GSE174554 partition")
parser.add_argument("--frozen-dir", type=Path, help="Directory with matched cluster NPZ, cell CNV NPZ and marker CSV")
frozen_dir = parser.parse_args().frozen_dir
cnv_path = Path("03_result/05_gse174554_cluster_cnv/01_gse174554_cell_cnv.npz")
if frozen_dir is not None:
    cnv_path = frozen_dir / "gse174554_cnv_leiden.npz"

# Keep the original RNA library order. Check sample and tissue row by row and distinguish archived barcodes from row
# keys reconstructed from the RNA libraries.
gene_names, matrix_paths, cell_table_total = upstream.gse174554_libraries()
genes = pd.Index(gene_names)
cell_ids = cell_table_total["cell_id"].to_numpy(dtype=str)
samples = cell_table_total["sample"].to_numpy(dtype=str)
tissue_table = pd.read_csv("data/intermediate/GSE174554_metadata.txt.gz", sep=r"\s+", dtype=str, keep_default_na=False)
tissue_table = tissue_table.drop_duplicates(["Sample#_Barcode", "Tumor_Normal_annotation"])
tissue_lookup = tissue_table.set_index("Sample#_Barcode", verify_integrity=True)["Tumor_Normal_annotation"]
observed_tissues = cell_table_total["tissue_id"].map(tissue_lookup).fillna("NA").to_numpy(dtype=str)

# In both modes the selected CNV archive supplies scores, tissues and partition membership. Stage 05 also records
# cluster identities and row keys that can be checked against RNA.
with np.load(cnv_path, allow_pickle=False) as archive:
    cnv = archive["cnv"]
    threshold = float(archive["strict"].item())
    leiden = archive["leiden"]
    tissues = archive["TN"]
    if cnv.shape != cell_ids.shape or leiden.shape != cell_ids.shape or not np.array_equal(tissues, observed_tissues):
        raise ValueError("CNV or partition lengths or the tissue order disagree with the RNA libraries.")
    if not np.isfinite(cnv).all() or not np.isfinite(threshold):
        raise ValueError("The archive contains nonfinite CNV scores or a nonfinite threshold.")
    if frozen_dir is None:
        primary_cluster = int(archive["primary_astro_cluster"].item())
        sender = archive["astro_mask"]
        positive = archive["cnv_positive_mask"]
        receiver_05 = archive["myeloid_mask"]
        has_ids = bool(archive["partition_has_cell_id"].item())
        row_key = "cell_id" if has_ids else "reconstructed_cell_id"
        available_keys = {"cell_id", "reconstructed_cell_id"}.intersection(archive.files)
        if available_keys != {row_key} or not np.array_equal(archive[row_key], cell_ids):
            raise ValueError("The stage 05 row-key source, stored field or current RNA barcode order disagrees.")
        if not np.array_equal(archive["SAMP"], samples):
            raise ValueError("Stage 05 sample IDs differ from the current RNA libraries in row order.")
        if not np.array_equal(sender, leiden == primary_cluster) or not np.array_equal(positive, cnv > threshold):
            raise ValueError("Stage 05 masks disagree with its sender cluster, CNV scores or threshold.")
        if sender.dtype != bool or positive.dtype != bool or receiver_05.dtype != bool:
            raise ValueError("Stage 05 cell group masks must be boolean vectors.")
        partition_source = str(archive["partition_source"].item())
        marker_source = str(archive["marker_source"].item())
        cell_order_check = str(archive["cell_order_check"].item())

# Match the two original archives and marker table in the specified directory. Verify all cluster sizes and the
# astrocyte and oligodendrocyte marker ranks before using the historical cluster IDs.
if frozen_dir is not None:
    cluster_path = frozen_dir / "gse174554_cluster.npz"
    marker_path = frozen_dir / "gse174554_cluster_markers.csv"
    with np.load(cluster_path, allow_pickle=False) as partition:
        if not np.array_equal(partition["SAMP"], samples) or not np.array_equal(partition["TN"], tissues):
            raise ValueError("The archived sample or tissue order differs from the current RNA and CNV records.")
        if not np.array_equal(partition["leiden"], leiden):
            raise ValueError("Cell-level cluster IDs differ between the legacy CNV archive and its partition.")
    marker_table = pd.read_csv(marker_path, index_col="cluster")
    cluster_counts = pd.Series(leiden).value_counts().sort_index()
    if not marker_table["n"].sort_index().equals(cluster_counts):
        raise ValueError("Cluster sizes in the legacy marker table differ from the selected partition.")
    if not np.isfinite(marker_table[["astro", "oligo"]].to_numpy()).all():
        raise ValueError("The frozen marker table contains nonfinite astrocyte or oligodendrocyte scores.")
    astro_clusters = marker_table["astro"].nlargest(2).index.tolist()
    if astro_clusters != [5, 10] or marker_table["oligo"].idxmax() != 1:
        raise ValueError("Marker ranks differ from the published primary and secondary astrocyte and oligo clusters.")
    primary_cluster = int(astro_clusters[0])
    sender = leiden == primary_cluster
    positive = cnv > threshold
    partition_source = cluster_path.name
    marker_source = marker_path.name
    cell_order_check = "SAMP, TN and Leiden matched. Barcode order within each sample and tissue is unverified."

# Use the original seven ligands and eight receptors. This cohort has a different myeloid panel from GSE211376; select
# its measured markers.
ligands = ["ANXA1", "LGALS1", "TGFB1", "TGFB2", "CSF1", "CCL2", "CD274"]
receptors = ["FPR1", "FPR3", "PTPRC", "TGFBR1", "TGFBR2", "CSF1R", "CCR2", "PDCD1"]
measured = [gene for gene in ligands + receptors if gene in genes]
positions = genes.get_indexer(measured)
immune_genes = [gene for gene in upstream.lineage_genes["imm"] if gene in genes]
if not immune_genes:
    raise ValueError("No myeloid marker genes are available.")
immune_positions = genes.get_indexer(immune_genes)

# Read signaling genes and myeloid markers from the same matrix in the RNA library order used for the frozen analysis.
expression_blocks = []
immune_blocks = []
for matrix_path in matrix_paths:

    # Compute the divisor for counts per 10,000 from all genes in each cell. Log-transform signaling genes separately;
    # average myeloid marker counts before the log transform.
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
    raise ValueError("The combined matrix and cell-level labels have different cell counts.")

# Start with the verified primary astrocyte cluster and remove cells above its CNV threshold. Recompute myeloid
# receivers and compare their mask with stage 05 row by row.
clean = sender & ~positive
receiver = immune_score > np.quantile(immune_score, 0.90)
if frozen_dir is None and not np.array_equal(receiver, receiver_05):
    raise ValueError("Myeloid receivers from the expression matrix differ from the matching stage 05 mask.")
if not sender.any() or not clean.any() or not receiver.any():
    raise ValueError("A sender or receiver group is empty; the signaling mean cannot be estimated.")
receptor_means = {}
for gene in receptors:
    if gene in measured:
        receptor_means[gene] = float(expression_table_total.loc[receiver, gene].mean())

# The historical GSE174554 method assigns zero to missing panel genes at either end. Compute percentage changes from
# unrounded products and use zero for a zero baseline.
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

# Check the decrease in sender count against the number above threshold and verify that every pair has a record.
sender_count = int(sender.sum())
clean_count = int(clean.sum())
if sender_count - clean_count != int((sender & (cnv > threshold)).sum()) or len(pair_scores) != len(pairs):
    raise ValueError("The sender denominator or ligand-receptor pair count is inconsistent.")
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
upstream.write_checked_json(Path("03_result/10_gse174554_sender_filtering/01_sender_filtering_scores.json"), result)
print(f"Sender cells: {sender_count}; retained at the CNV threshold: {clean_count}.")
