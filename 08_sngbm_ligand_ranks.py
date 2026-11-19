# Purpose: Rank AC-like ligands in precomputed snGBM LIANA results and compare them with scGBM myeloid-receiver
# expression products.
# Inputs: data/external/snGBM/results/6_CCC_liana/adata_with_liana.h5ad, uns/liana_res
# (source, target, ligand_complex, lr_means). Read scGBM tam_scores preferentially from the stage 07 file
# 03_result/07_scgbm_communication_and_mixture_adjustment/01_communication_and_mixture_adjustment.json or use
# data/intermediate/communication_analysis_results.json.
# For standalone analysis without either summary, use data/external/scGBM/adata2.h5ad (X, Minor_cluster,
# Major_category).
# Output: 03_result/08_sngbm_ligand_ranks/01_liana_ligand_ranks.json
# Run from the repository root with the project Python and AnnData: python 08_sngbm_ligand_ranks.py

import json
from contextlib import closing
from importlib import import_module
from math import isfinite
from pathlib import Path

import anndata as ad

communication = import_module("00a_cell_communication")
upstream = import_module("00a_expression_and_clustering")

# Each LIANA record describes a sender, receiver and ligand-receptor complex. Select the four malignant sender states
# and the Macrophage receiver.
sngbm = ad.read_h5ad("data/external/snGBM/results/6_CCC_liana/adata_with_liana.h5ad", backed="r")
with closing(sngbm.file):
    liana_table = sngbm.uns["liana_res"]
    cell_count = sngbm.n_obs
malignant_states = ["AC-like", "MES-like", "OPC-like", "NPC-like"]
selected = liana_table["source"].isin(malignant_states) & liana_table["target"].eq("Macrophage")
macrophage_table = liana_table.loc[selected]

# For each ligand, take the maximum lr_means across receptors within each sender state. Rank ligands reported in
# AC-like and at least one other state.
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

    # The published AC_rank truncates the average tied rank to an integer. Use the actual maximum in cross-method
    # comparisons so truncation cannot create a spurious first place.
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

# Prefer the scores produced by stage 07. For a standalone comparison, the public scGBM object supplies the same
# scores without in-house data.
current_scores = Path(
    "03_result/07_scgbm_communication_and_mixture_adjustment/01_communication_and_mixture_adjustment.json"
)
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

# Compare at the ligand level. scGBM scores a single receptor per pair, whereas snGBM takes the maximum across
# reported receptors; the receptor complexes can differ.
ranked_ligands = {row["ligand"] for row in rank_rows}
both_highest = []
other_results = []
for pair_name, state_scores in tam_scores.items():
    if set(state_scores) != {"AC", "MES1", "OPC", "NPC"} or not all(isfinite(value) for value in state_scores.values()):
        raise ValueError(f"scGBM scores for {pair_name} cannot be compared across all sender states.")
    ligand = pair_name.partition("-")[0]
    if ligand not in key_ligands:
        continue
    if ligand not in ranked_ligands:
        other_results.append(f"{pair_name}: no eligible AC-like ligand rank in the snGBM macrophage table.")
        continue

    # Compare sender states within each cohort. An absent record can reflect unmeasured expression, reporting limits
    # or upstream filtering.
    if state_scores["AC"] == max(state_scores.values()) and ligand in peak_ligands:
        both_highest.append(f"{pair_name}: AC reaches the highest scGBM product and snGBM ligand-level lr_means.")
    else:
        other_results.append(f"{pair_name}: AC does not reach the highest score in both within-cohort comparisons.")

# Check ligand uniqueness, rank range and subtype. Keep published JSON keys and build their descriptions from the
# current comparison.
if len(ranked_ligands) != len(rank_rows):
    raise ValueError("A LIANA ligand appears more than once in the ranking.")
for row in rank_rows:
    valid_rank = 1 <= row["AC_rank"] <= row["n_subtypes"] <= len(malignant_states)
    if not valid_rank or row["top_subtype"] not in malignant_states:
        raise ValueError(f"The rank for {row['ligand']} is inconsistent with its observed sender states.")
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
upstream.write_checked_json(Path("03_result/08_sngbm_ligand_ranks/01_liana_ligand_ranks.json"), result)
print(f"Myeloid-target pairs: {len(macrophage_table)}; eligible AC-like ligand ranks: {len(rank_rows)}.")
