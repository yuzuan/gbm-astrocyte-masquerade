# Purpose: Compare tumor and peritumoral AC expression in scGBM and reconstruct published fold changes between high-
# and low-CNV-score AC cells in the same cohort.
# Input: data/external/scGBM/adata2.h5ad (X, Minor_cluster, tissue_type, Major_category, cnv_score).
# Output: 03_result/13_scgbm_ac_like_regional_comparison/01_tumor_peritumor_ac_expression.csv
# Run from the repository root with the project Python and AnnData: python 13_scgbm_ac_like_regional_comparison.py

from contextlib import closing
from importlib import import_module

import anndata as ad
import numpy as np
import pandas as pd

communication = import_module("00a_cell_communication")

# The original session used the same scGBM object as stages 07 and 11. Keep measured genes in the historical panel
# order and distinguish unmeasured genes from zero expression.
scgbm = ad.read_h5ad("data/external/scGBM/adata2.h5ad", backed="r")
with closing(scgbm.file):
    panel = [
        "GFAP", "AQP4", "S100B", "SLC1A3", "CLU", "ALDOC", "CD44", "VIM", "CHI3L1", "CHI3L2", "EGFR", "PDGFRA",
        "SOX2", "NES", "HIF1A", "VEGFA", "LGALS1", "LGALS3", "SERPINE1", "ANXA1",
    ]
    genes = [gene for gene in panel if gene in scgbm.var_names]
    is_ac = scgbm.obs["Minor_cluster"].eq("AC")
    tumor = is_ac & scgbm.obs["tissue_type"].eq("Tumor")
    peri = is_ac & scgbm.obs["tissue_type"].eq("Peritumoral")

    # The published MES1_mean and OPC_mean pool both tissue locations. Sample independently within each group above
    # the cap. Detection means X > 0.
    region_masks = {
        "AC_Tumor": tumor,
        "AC_Peri": peri,
        "MES1": scgbm.obs["Minor_cluster"].eq("MES1"),
        "OPC": scgbm.obs["Minor_cluster"].eq("OPC"),
    }
    comparison_table = pd.DataFrame({"gene": genes})
    detection = {}
    for group, mask in region_masks.items():
        positions = communication.sampled_scgbm_positions(mask)
        expression_table = communication.cell_expression_table(scgbm, positions, genes)
        comparison_table[f"{group}_mean"] = [np.mean(expression_table[gene].to_numpy()) for gene in genes]
        if group in ("AC_Tumor", "AC_Peri"):
            detection[group] = [(expression_table[gene].to_numpy() > 0).mean() * 100 for gene in genes]

    # Report detection rates and the regional ratio for the two AC tissue groups. The fold change is undefined when
    # the peritumoral mean is zero.
    comparison_table["AC_Tumor_pct"] = detection["AC_Tumor"]
    comparison_table["AC_Peri_pct"] = detection["AC_Peri"]
    peri_mean = comparison_table["AC_Peri_mean"].replace(0, np.nan)
    comparison_table["fold_T_vs_P"] = comparison_table["AC_Tumor_mean"] / peri_mean

    # Historical own_fold values come from the scGBM high/low AC score table. Following its source script, use the
    # 95th percentile across all reference cells to partition all AC cells.
    cnv = scgbm.obs["cnv_score"]
    reference = scgbm.obs["Major_category"].isin(["TAM/Macrophage", "Microglia", "T/NK"])
    if not reference.any() or not np.isfinite(cnv.loc[reference | is_ac]).all():
        raise ValueError("The reference group is empty or AC or reference cells have nonfinite CNV scores.")
    threshold = np.percentile(cnv.loc[reference].to_numpy(), 95)
    high = is_ac & cnv.gt(threshold)
    low = is_ac & cnv.le(threshold)
    historical_folds = pd.Series(np.nan, index=genes)

    # Use all cells in both score groups. Retain the published lower bound of 0.001 on the low-group mean and round
    # the fold to two decimals before assigning its historical category.
    if high.any() and low.any():
        high_table = communication.cell_expression_table(scgbm, np.flatnonzero(high), genes)
        low_table = communication.cell_expression_table(scgbm, np.flatnonzero(low), genes)
        for gene in genes:
            high_mean = np.mean(high_table[gene].to_numpy())
            low_mean = np.mean(low_table[gene].to_numpy())
            fold = high_mean / max(low_mean, 0.001)
            historical_folds.loc[gene] = float(str(round(fold, 2)))
    comparison_table["own_fold"] = comparison_table["gene"].map(historical_folds)

# Keep the published direction labels. own_interpret compares high versus low CNV scores within scGBM; the interval
# from 0.8 through 1.3, including its endpoints, is shared.
for column, target, above, below in (
    ("own_fold", "own_interpret", "malignant↑", "normal↑"),
    ("fold_T_vs_P", "scgbm_interpret", "Tumor↑", "Peri↑"),
):
    values = comparison_table[column]
    comparison_table[target] = np.select(
        [values.isna(), values.gt(1.3), values.lt(0.8)], ["NA", above, below], default="shared"
    )

# CNV means use all AC cells in each location; expression means use the samples above. Count direction agreement among
# evaluable genes. A group mean cannot establish an individual cell's malignancy.
comparable = comparison_table["own_fold"].notna() & comparison_table["fold_T_vs_P"].notna()
site_directions = comparison_table["scgbm_interpret"].replace({"Tumor↑": "malignant↑", "Peri↑": "normal↑"})
same_direction = comparable & comparison_table["own_interpret"].eq(site_directions)
comparison_table["CAVEAT"] = (
    f"Among all AC cells, mean cnv_score was {cnv[tumor].mean():.6f} in Tumor (n={int(tumor.sum())}) "
    f"and {cnv[peri].mean():.6f} in Peritumoral (n={int(peri.sum())}). "
    "fold_T_vs_P compares locations. Historical own_fold and its direction labels compare high versus low CNV scores "
    f"in AC cells of this same scGBM cohort. Direction categories agree for {int(same_direction.sum())} "
    f"of {int(comparable.sum())} evaluable genes."
)

# Check percentage units for both detection rates and missing ratios for zero denominators. Verify gene order and all
# fields when reading back the full CSV.
for column in ("AC_Tumor_pct", "AC_Peri_pct"):
    if not comparison_table[column].between(0, 100).all():
        raise ValueError(f"The detection rate in {column} lies outside the percentage range.")
if not comparison_table["fold_T_vs_P"].isna().equals(comparison_table["AC_Peri_mean"].eq(0)):
    raise ValueError("Missing regional fold changes do not match zero peritumoral means.")
communication.write_gene_table(
    "03_result/13_scgbm_ac_like_regional_comparison/01_tumor_peritumor_ac_expression.csv", comparison_table, genes
)
print(f"Evaluable genes (region and CNV): {int(comparable.sum())}; matching directions: {int(same_direction.sum())}.")
