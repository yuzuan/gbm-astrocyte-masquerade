# Purpose: Compare in-house GBM and snGBM astrocyte +7/-10 exceedance rates using each cohort's immune-reference
#          95th percentile.
# Inputs: data/infercnv_cell_scores.csv (barcode in the first column, cell_type, chr7, chr10, cnv_pos),
#         data/cell_metadata.csv (barcode in the first column, condition), and snGBM_independent in
#         03_result/01_sngbm_immune_reference_cnv/01_sngbm_immune_reference_cnv.json.
#         Legacy runs can read noncircular_cnv_validation.json from data/intermediate or the working directory.
# Output: 03_result/06_cross_cohort_cnv_thresholds/01_matched_cnv_thresholds.json
# Usage: Run python 06_cross_cohort_cnv_thresholds.py from the repository root with the original cutoff calculation.

import json
from importlib import import_module
from pathlib import Path

common = import_module("00a_expression_and_clustering")

# Recompute in-house rates from cell scores. All tissues enter the three reference groups. Only Astrocyte cells with
# condition=GBM enter the primary denominator.
own = common.inhouse_cnv_summary(Path("data/infercnv_cell_scores.csv"), Path("data/cell_metadata.csv"))

# Stage 01 computes snGBM results against its own immune reference. A legacy frozen summary may contain only the
# positive percentage and denominator, leaving the numerical cutoff unavailable.
sn_path = common.result_or_frozen(
    Path("03_result/01_sngbm_immune_reference_cnv/01_sngbm_immune_reference_cnv.json"),
    "noncircular_cnv_validation.json",
)
sn_result = json.loads(sn_path.read_text(encoding="utf-8"))
sn_group = sn_result["snGBM_independent"]
sn_count = sn_group["n"]
sn_percent = sn_group["strict_CNVpos_pct"]
if type(sn_count) is not int or sn_count <= 0 or type(sn_percent) not in (int, float) or not 0 <= sn_percent <= 100:
    raise ValueError("The snGBM input lacks a valid astrocyte denominator or percentage above the strict cutoff.")

# The cohorts share a reference-percentile rule. Upstream cnv_pos defines the loose in-house calls. Reference FPR
# is the percentage above cutoff in each reference cell type.
# The historical peritumoral field uses in-house Normal astrocytes. The condition column gives no finer sampling site.
result = {
    "own_strict_pct": own["strict_matched_pct"],
    "own_loose_pct": own["loose_cnv_pos_pct"],
    "own_n": own["n"],
    "snGBM_strict_pct": sn_percent,
    "snGBM_n": sn_count,
    "own_ref_FPR_strict": own["strict_ref_FPR"],
    "peritumoral_astro_strict_pct": own["peritumoral_astro_strict_pct"],
    "own_strict_threshold": own["strict_threshold"],
    "snGBM_strict_threshold": sn_group.get("strict_threshold"),
    "own_ref_n": own["reference_n"],
    "own_normal_astro_n": own["normal_astro_n"],
    "sn_summary_source": sn_path.as_posix(),
    "note": (
        f"In-house GBM astrocytes (n={own['n']}): {own['loose_cnv_pos_pct']:.1f}% positive by upstream cnv_pos, "
        f"{own['strict_matched_pct']:.1f}% above the in-house immune 95th percentile. "
        f"snGBM Leiden cluster 0 (n={sn_count}): {sn_percent:.1f}% above the snGBM immune 95th percentile."
    ),
}

# Check that each reference percentage has a matching denominator. A missing cell type must have a null percentage.
# Read back the full result before publication.
for label, count in result["own_ref_n"].items():
    percent = result["own_ref_FPR_strict"][label]
    if (count == 0) != (percent is None):
        raise ValueError(f"In-house {label} has inconsistent denominator and false-positive rate missingness.")
common.write_checked_json(Path("03_result/06_cross_cohort_cnv_thresholds/01_matched_cnv_thresholds.json"), result)
print(
    f"Above the immune-reference cutoff: in-house GBM {result['own_strict_pct']}% (n={own['n']}), "
    f"snGBM {sn_percent}% (n={sn_count})."
)
