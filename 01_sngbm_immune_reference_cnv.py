# Purpose: Measure the fraction of cells in frozen snGBM Leiden cluster 0 above the immune-reference +7/-10 cutoff.
# Inputs: var/chromosome in data/external/scGBM/adata2.h5ad and X, var_names, and obs/leiden_0.5 in
#         data/external/snGBM/results/1_snRNA_processing/adata1.h5ad.
#         The in-house inputs are data/infercnv_cell_scores.csv and data/cell_metadata.csv. If both are absent,
#         own_cohort can be read from data/intermediate/noncircular_cnv_validation.json.
# Output: 03_result/01_sngbm_immune_reference_cnv/01_sngbm_immune_reference_cnv.json
# Usage: Run python 01_sngbm_immune_reference_cnv.py from the repo root in the original Python/AnnData environment.

import json
from importlib import import_module
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

common = import_module("00a_expression_and_clustering")

# X is processed expression. Frozen cluster IDs define the comparison independently of author-assigned malignancy.
sn_data = ad.read_h5ad("data/external/snGBM/results/1_snRNA_processing/adata1.h5ad", backed="r")
cell_table = sn_data.obs[["leiden_0.5"]].copy()
clusters = cell_table["leiden_0.5"].astype(str)
if not {"0", "2", "5", "10", "13"}.issubset(set(clusters)):
    raise ValueError("Missing historical snGBM target or reference cluster. Check Leiden resolution and input version.")


def mean_expression(data, genes):
    """Return the arithmetic mean across matched genes per cell. Blocks preserve gene order and expression precision.

    An empty panel raises an error.
    """
    positions = data.var_names.get_indexer(genes)
    positions = positions[positions >= 0]
    if not positions.size:
        raise ValueError("The snGBM expression matrix contains no genes from this panel.")

    # Read consecutive cell rows before selecting gene columns in memory. This avoids loading the whole backed CSR
    # matrix through noncontiguous column indexing in AnnData.
    block_means = []
    for start in range(0, data.n_obs, 1024):
        block = data[start:start + 1024, :].X
        block = block[:, positions]
        if sparse.issparse(block):
            block = block.toarray()
        block_means.append(np.asarray(block).mean(axis=1))
    means = np.concatenate(block_means)
    return means


# The panels measure astrocyte, myeloid, and oligodendrocyte expression. The latter two characterize clusters only.
try:
    cell_table["astro"] = mean_expression(
        sn_data, ["GFAP", "AQP4", "SLC1A3", "ALDH1L1", "GJA1", "CLU", "AGT", "SLC1A2", "GLUL"]
    )
    cell_table["immune"] = mean_expression(sn_data, ["PTPRC", "AIF1", "CD68", "P2RY12", "CX3CR1", "ITGAM", "C1QB"])
    cell_table["oligo"] = mean_expression(sn_data, ["MBP", "PLP1", "MOG", "MAG", "MOBP"])

    # Independent scGBM annotations supply the chromosome genes. Sorted names preserve the original summation order.
    chromosome_genes = common.chromosome_gene_sets()
    cell_table["chr7"] = mean_expression(sn_data, sorted(chromosome_genes["chr7"]))
    cell_table["chr10"] = mean_expression(sn_data, sorted(chromosome_genes["chr10"]))
finally:
    sn_data.file.close()

# The original analysis fixed cluster 0. Report its astrocyte-panel rank across clusters and its other two panel means.
marker_table = cell_table.groupby(clusters, observed=True)[["astro", "immune", "oligo"]].mean()
if not np.isfinite(marker_table.to_numpy()).all():
    raise ValueError("The snGBM cluster marker means contain nonfinite values.")
astro_rank = int(marker_table["astro"].rank(ascending=False, method="min").at["0"])
marker_means = marker_table.loc["0"]
cluster_description = (
    f"Leiden 0; astro panel mean {marker_means['astro']:.3f} (rank {astro_rank}/{len(marker_table)}), "
    f"immune {marker_means['immune']:.3f}, oligo {marker_means['oligo']:.3f}"
)

# Macrophage cluster 5 centers both chromosomes. The strict cutoff is the pooled 95th percentile of clusters 5 and 10
# (NK/T). A score equal to the cutoff is negative.
macrophages = clusters.eq("5")
chr7_gain = cell_table["chr7"] - cell_table.loc[macrophages, "chr7"].mean()
chr10_change = cell_table["chr10"] - cell_table.loc[macrophages, "chr10"].mean()
cnv_scores = chr7_gain - chr10_change
if not np.isfinite(cnv_scores).all():
    raise ValueError("A target or reference cell in snGBM has a missing or infinite CNV score.")
threshold = np.quantile(cnv_scores[clusters.isin(["5", "10"])], 0.95)
strict_positive = cnv_scores > threshold

# Oligodendrocyte cluster 2 and pericyte cluster 13 each have their own denominator. Historical FPR fields give
# the percentage above threshold in each reference group. Pericytes are also reported separately.
reference_ids = {"Macrophage": "5", "NK_T": "10", "Oligodendrocyte": "2", "Pericyte": "13"}
reference_rates = {}
for label, cluster_id in reference_ids.items():
    reference = common.positive_summary(strict_positive, clusters.eq(cluster_id))
    reference_rates[label] = reference["cnv_pos_pct"]
astrocytes = clusters.eq("0")
astro = common.positive_summary(strict_positive, astrocytes)
loose_macrophage = common.positive_summary((chr7_gain > 0) | (chr10_change < 0), macrophages)

# The historical JSON covers both cohorts. When both in-house tables are absent, copy only the documented frozen
# summary if available and leave unreported values empty.
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

# Each cohort uses its own immune-reference percentile. The comparison reports the observed fractions. These
# expression-based proxy scores do not establish an individual cell's malignancy.
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

# Check the target denominator against the cluster counts. The writer reads back the entire JSON before publication.
if result["snGBM_independent"]["n"] != clusters.value_counts().at["0"]:
    raise ValueError("The snGBM summary denominator differs from the cluster 0 cell count.")
common.write_checked_json(Path("03_result/01_sngbm_immune_reference_cnv/01_sngbm_immune_reference_cnv.json"), result)
print(f"snGBM cluster 0: {astro['n']} cells, with {astro['cnv_pos_pct']:.1f}% above the immune-reference cutoff.")
