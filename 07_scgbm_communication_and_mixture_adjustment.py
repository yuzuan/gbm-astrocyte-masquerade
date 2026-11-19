# Purpose: Compute scGBM sender ligand means times receiver receptor means and estimate in-house GBM astrocyte scores
# before and after mixture adjustment.
# Inputs: data/external/scGBM/adata2.h5ad (X, Minor_cluster, Major_category), data/malignant_gbm_cnv.h5ad (X, state),
# and data/cellcell_communication.csv (condition, sender, receiver, ligand, receptor, mean_L_sender, score, perm_p).
# CNV fractions use data/infercnv_cell_scores.csv (cell_type, cnv_pos) and data/cell_metadata.csv (condition); the
# first column is the cell ID.
# If only the frozen summary is available, take contamination_rates from
# data/intermediate/communication_analysis_results.json.
# Output: 03_result/07_scgbm_communication_and_mixture_adjustment/01_communication_and_mixture_adjustment.json
# From the repository root with the project AnnData and NumPy environment, run:
# python 07_scgbm_communication_and_mixture_adjustment.py

import json
from contextlib import closing
from importlib import import_module
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

communication = import_module("00a_cell_communication")
upstream = import_module("00a_expression_and_clustering")

# Score all four scGBM sender states against myeloid and T/NK receivers, averaging each shared receptor once per
# receiver group.
scgbm = ad.read_h5ad("data/external/scGBM/adata2.h5ad", backed="r")
with closing(scgbm.file):
    tam_scores = communication.scgbm_product_scores(scgbm, "TAM/Macrophage")
    tnk_scores = communication.scgbm_product_scores(scgbm, "T/NK")

# Estimate the in-house GBM mixture fraction from cell-level CNV calls. The source tables have different row orders,
# so align them by cell ID.
scores_path = Path("data/infercnv_cell_scores.csv")
metadata_path = Path("data/cell_metadata.csv")
if scores_path.exists() or metadata_path.exists():
    cnv_table = pd.read_csv(scores_path, index_col=0, dtype={"cnv_pos": "boolean"})
    metadata_table = pd.read_csv(metadata_path, index_col=0)
    cell_table_total = cnv_table.join(metadata_table[["condition"]], how="left", validate="one_to_one")
    if cell_table_total[["cnv_pos", "condition", "cell_type"]].isna().any().any():
        raise ValueError("Mixture adjustment requires cell type, CNV call and tissue condition for each scored cell.")

    # Pooling tissue conditions changes the denominator. Use the GBM-specific fraction for GBM signaling.
    astro_table = cell_table_total.loc[cell_table_total["cell_type"].eq("Astrocyte")]
    gbm_positive = astro_table.loc[astro_table["condition"].eq("GBM"), "cnv_pos"]
    normal_positive = astro_table.loc[astro_table["condition"].eq("Normal"), "cnv_pos"]
    myeloid_positive = cell_table_total.loc[cell_table_total["cell_type"].eq("Myeloid"), "cnv_pos"]
    oligo_positive = cell_table_total.loc[cell_table_total["cell_type"].eq("Oligodendrocyte"), "cnv_pos"]
    if any(group.empty for group in (gbm_positive, normal_positive, myeloid_positive, oligo_positive)):
        raise ValueError("A cell type or tissue condition required for the historical fractions is empty.")

    # The historical formula uses the GBM fraction rounded to three decimals; displayed percentages use one. Preserve
    # published fields for downstream users.
    rates = {
        "Astrocyte_cnv_pos": round(100 * astro_table["cnv_pos"].mean(), 1),
        "Astrocyte_total": len(astro_table),
        "Astrocyte_cnv_pos_n": int(astro_table["cnv_pos"].sum()),
        "Astrocyte_cleaned_n": int((~astro_table["cnv_pos"]).sum()),
        "Myeloid_cnv_pos": round(100 * myeloid_positive.mean(), 1),
        "Oligodendrocyte_cnv_pos": round(100 * oligo_positive.mean(), 1),
        "Astrocyte_cnv_pos_GBM": round(100 * gbm_positive.mean(), 1),
        "Astrocyte_cnv_pos_Normal": round(100 * normal_positive.mean(), 1),
        "f_malig_used": round(float(gbm_positive.mean()), 3),
    }
else:

    # The external frozen JSON supplies only the historical fractions. Compute pair scores and mixture adjustments
    # from the current inputs.
    with Path("data/intermediate/communication_analysis_results.json").open(encoding="utf-8") as stream:
        frozen_result = json.load(stream)
    rates = frozen_result["contamination_rates"].copy()

# Cross-check numerators, denominators and the mixture weight. Earlier frozen tables omitted the condition-specific
# fractions.
for key in (
    "Astrocyte_cnv_pos", "Myeloid_cnv_pos", "Oligodendrocyte_cnv_pos",
    "Astrocyte_cnv_pos_GBM", "Astrocyte_cnv_pos_Normal",
):
    if not 0 <= rates[key] <= 100:
        raise ValueError(f"Historical field {key} is missing or outside the percentage range.")
positive_fraction = rates["f_malig_used"]
if rates["Astrocyte_cnv_pos_n"] + rates["Astrocyte_cleaned_n"] != rates["Astrocyte_total"]:
    raise ValueError("The CNV-positive and retained astrocyte counts do not sum to the total.")
astro_percent = round(100 * rates["Astrocyte_cnv_pos_n"] / rates["Astrocyte_total"], 1)
if astro_percent != rates["Astrocyte_cnv_pos"] or not 0 <= positive_fraction < 1:
    raise ValueError("The overall astrocyte CNV-positive percentage or the GBM mixture weight is inconsistent.")
if not np.isclose(positive_fraction * 100, rates["Astrocyte_cnv_pos_GBM"]):
    raise ValueError("The mixture weight disagrees with the GBM-specific CNV-positive percentage.")
rates["note"] = f"GBM-specific rate ({rates['Astrocyte_cnv_pos_GBM']:.1f}%) used for GBM communication decontamination"

# Include all in-house cells labeled AClike. Adjust observed signaling records for ligands on the historical panel.
own_dataset = ad.read_h5ad("data/malignant_gbm_cnv.h5ad", backed="r")
with closing(own_dataset.file):
    key_ligands = ["LGALS1", "ANXA1", "TGFB1", "TGFB2", "CCL2", "CSF1", "CD274", "VEGFA", "EGFR", "SERPINE1"]
    available_ligands = [gene for gene in key_ligands if gene in own_dataset.var_names]
    ac_positions = np.flatnonzero(own_dataset.obs["state"].eq("AClike"))
    ac_table = communication.cell_expression_table(own_dataset, ac_positions, available_ligands)

# Each input row is a ligand-receptor pair. Keep GBM Astrocyte-to-Myeloid records in their original order.
comm_table = pd.read_csv("data/cellcell_communication.csv")
eligible = comm_table["condition"].eq("GBM") & comm_table["sender"].eq("Astrocyte")
eligible &= comm_table["receiver"].eq("Myeloid")
selected_table = comm_table.loc[eligible & comm_table["ligand"].isin(available_ligands)]
if not np.isfinite(selected_table[["mean_L_sender", "score", "perm_p"]].to_numpy()).all():
    raise ValueError("Selected signaling records contain a nonfinite ligand mean, score or permutation P value.")

# Use the original score precision. Solve for the ligand mean of the CNV-negative fraction and clip negative means to
# zero as in the original analysis.
pollution_rows = []
for row in selected_table.itertuples(index=False):
    malignant_mean = float(np.mean(ac_table[row.ligand].to_numpy()))
    corrected_mean = max(0, (row.mean_L_sender - positive_fraction * malignant_mean) / (1 - positive_fraction))

    # Scale each score by the ratio of adjusted to unadjusted ligand means. The permutation P value comes from the
    # original, unadjusted record.
    corrected_score = row.score * (corrected_mean / row.mean_L_sender) if row.mean_L_sender > 0 else 0
    percent_change = (corrected_score - row.score) / row.score * 100 if row.score > 0 else 0
    pollution_rows.append(
        {
            "ligand": row.ligand,
            "receptor": row.receptor,
            "mean_L_before": round(row.mean_L_sender, 4),
            "mean_L_after": round(corrected_mean, 4),
            "mean_L_malig": round(malignant_mean, 4),
            "score_before": round(row.score, 4),
            "score_after": round(corrected_score, 4),
            "pct_change": round(percent_change, 1),
            "sig_before": "yes" if row.perm_p < 0.05 else "no",
        }
    )

# Check all sender states for each scGBM score and compare the in-house pair order with the filtered input.
for receiver_scores in (tam_scores, tnk_scores):
    for state_scores in receiver_scores.values():
        if set(state_scores) != {"AC", "MES1", "OPC", "NPC"}:
            raise ValueError("A signaling score is missing a sender state.")
written_pairs = [(record["ligand"], record["receptor"]) for record in pollution_rows]
if written_pairs != list(zip(selected_table["ligand"], selected_table["receptor"])):
    raise ValueError("Adjusted pairs do not match the original GBM signaling records.")

result = {
    "tam_scores": tam_scores,
    "tnk_scores": tnk_scores,
    "pollution_results": pollution_rows,
    "contamination_rates": rates,
}
upstream.write_checked_json(
    Path("03_result/07_scgbm_communication_and_mixture_adjustment/01_communication_and_mixture_adjustment.json"),
    result,
)
print(f"Myeloid pairs: {len(tam_scores)}; T/NK pairs: {len(tnk_scores)}; mixture-adjusted rows: {len(pollution_rows)}.")
