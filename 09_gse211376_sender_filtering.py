# Purpose: Select GSE211376 sender cells by lineage markers and compare ligand-receptor scores before and after
# removing cells above the +7/-10 CNV threshold.
# Inputs: var gene names and chromosome assignments from data/external/scGBM/adata2.h5ad, and
# data/intermediate/GSE211376_raw_counts.tsv.gz (cell IDs in the header, followed by one raw count row per gene).
# Output: 03_result/09_gse211376_sender_filtering/01_sender_filtering_scores.json
# Run from the repository root with the project Python, HDF5 and NumPy: python 09_gse211376_sender_filtering.py

import gzip
from importlib import import_module
from pathlib import Path

import numpy as np

communication = import_module("00a_cell_communication")
upstream = import_module("00a_expression_and_clustering")

# Use the chromosome and lineage definitions from the preceding CNV analysis. The immune panel for this cohort also
# includes C1QA and C1QB.
gene_sets = upstream.chromosome_gene_sets()
gene_sets["astro"] = set(upstream.lineage_genes["astro"])
gene_sets["immune"] = set(upstream.lineage_genes["imm"]) | {"C1QA", "C1QB"}
gene_sets["oligo"] = set(upstream.lineage_genes["oligo"])

# Use the five historical ligand-receptor pairs. All genes contribute to library sizes; retain cell-level count
# vectors for genes in these pairs.
pairs = [("TGFB2", "TGFBR1"), ("TGFB2", "TGFBR2"), ("TGFB1", "TGFBR1"), ("ANXA1", "FPR1"), ("CSF1", "CSF1R")]
pair_genes = set()
for ligand, receptor in pairs:
    pair_genes.update((ligand, receptor))
with gzip.open("data/intermediate/GSE211376_raw_counts.tsv.gz", "rt", encoding="utf-8") as stream:
    cells = communication.count_matrix_cells(stream)
    total_counts = np.zeros(len(cells))
    group_sums = {name: np.zeros(len(cells)) for name in gene_sets}
    gene_counts = dict.fromkeys(gene_sets, 0)
    pair_counts = {}

    # A gene may belong to both a chromosome and a marker group. Accumulate each group in count-file order.
    for line in stream:
        gene, counts = communication.read_gene_counts(line, len(cells))
        total_counts += counts
        for name, members in gene_sets.items():
            if gene in members:
                group_sums[name] += counts
                gene_counts[name] += 1
        if gene in pair_genes:
            pair_counts[gene] = counts

# Normalize each cell to 10,000 counts, divide each group total by the number of matched genes, then take log1p of
# that group mean.
size_factors = communication.library_size_factors(total_counts)
marker_scores = {}
for name, summed_counts in group_sums.items():
    if gene_counts[name] == 0:
        raise ValueError(f"No genes matched {name} in the counts; cell groups or CNV scores cannot be determined.")
    marker_scores[name] = np.log1p(summed_counts / size_factors / gene_counts[name])

# Select senders above the astrocyte upper quartile, at or below the immune median and below an oligodendrocyte score
# of 1. Myeloid receivers come from the upper tail of the immune scores.
astro = marker_scores["astro"] > np.quantile(marker_scores["astro"], 0.75)
astro &= marker_scores["immune"] <= np.quantile(marker_scores["immune"], 0.5)
astro &= marker_scores["oligo"] < 1.0
myeloid = marker_scores["immune"] > np.quantile(marker_scores["immune"], 0.90)
if not myeloid.any():
    raise ValueError("The immune-high reference is empty; the CNV threshold cannot be estimated.")

# Center chr7 and chr10 on their respective myeloid means and subtract the centered chr10 score from chr7. Retain
# senders at the reference 95th percentile.
cnv_score, threshold, _, _ = upstream.centered_cnv(marker_scores["chr7"], marker_scores["chr10"], myeloid, myeloid)
clean = astro & (cnv_score <= threshold)

# Take log1p separately for each gene. Multiply each sender ligand mean by the same myeloid receptor mean. Missing
# genes and empty groups yield missing scores.
expression = {gene: np.log1p(counts / size_factors) for gene, counts in pair_counts.items()}
pair_scores = {}
for ligand, receptor in pairs:
    pair_name = f"{ligand}->{receptor}"
    record = {"before": None, "after": None, "pct_change": None}
    if ligand in expression and receptor in expression and astro.any():
        receptor_mean = expression[receptor][myeloid].mean()
        before = float(expression[ligand][astro].mean() * receptor_mean)
        record["before"] = round(before, 4)
        if clean.any():
            after = float(expression[ligand][clean].mean() * receptor_mean)
            record["after"] = round(after, 4)
            record["pct_change"] = round(100 * (after - before) / before, 1) if before else 0.0
    pair_scores[pair_name] = record

# Check how many senders were removed and retain a record for every pair, including unmeasured ones.
sender_count = int(astro.sum())
clean_count = int(clean.sum())
if sender_count - clean_count != int((astro & (cnv_score > threshold)).sum()) or len(pair_scores) != len(pairs):
    raise ValueError("The number of removed senders or ligand-receptor records is inconsistent.")
result = {"dataset": "GSE211376", "n_sender": sender_count, "n_clean": clean_count, "pairs": pair_scores}
upstream.write_checked_json(Path("03_result/09_gse211376_sender_filtering/01_sender_filtering_scores.json"), result)
print(f"Sender cells: {sender_count}; retained at the CNV threshold: {clean_count}.")
