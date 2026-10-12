#!/usr/bin/env python3
"""用统一 marker 规则重新标注公共恶性细胞，检验 TGFB2 结果对标签来源的敏感性。"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm, spearmanr, wilcoxon
import statsmodels.formula.api as smf

from stage_public_paths import project_root, public_input_root, submission_package


ROOT = project_root()
PACKAGE = submission_package(ROOT)
INPUT_ROOT = public_input_root(ROOT)
OUT = PACKAGE / "data"

COUNTS_211 = INPUT_ROOT / "raw_inputs" / "GSE211376_raw_counts.tsv.gz"
META_211 = INPUT_ROOT / "raw_inputs" / "GSE211376_metadata.csv.gz"
SCGBM = INPUT_ROOT / "processed_inputs" / "scGBM_GSE162631_GSE273274.h5ad"
SNGBM = INPUT_ROOT / "processed_inputs" / "snGBM_GSE276177_GSE286419.h5ad"
GSE274 = INPUT_ROOT / "processed_inputs" / "regional_GSE274987_GSE241037.h5ad"

MARKERS = {
    "AC-like": ["GFAP", "S100B", "HOPX", "SLC1A3", "MLC1"],
    "MES-like": ["VIM", "CD44", "CHI3L1", "HILPDA", "DDIT3", "ENO2"],
    "OPC-like": ["PLP1", "OLIG1", "OMG", "TNR", "ALCAM"],
    "NPC-like": ["SOX4", "SOX11", "DCX", "CD24", "STMN1", "STMN2"],
}
STATES = list(MARKERS)
SELECTED_GENES = list(dict.fromkeys(["TGFB2", *[gene for genes in MARKERS.values() for gene in genes]]))
MIN_CELLS = 20
MIN_SCORE_DIFF = 0.05
STANDARDIZE_MARKER_GENES = False


def dense(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def bh_adjust(values: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().astype(float)
    if valid.empty:
        return out
    order = np.argsort(valid.to_numpy())
    ranked = valid.to_numpy()[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    out.loc[valid.index] = restored
    return out


def safe_wilcoxon(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3 or np.allclose(values, 0):
        return np.nan
    return float(wilcoxon(values, alternative="two-sided", method="auto").pvalue)


def assign_uniform_states(expression: np.ndarray, genes: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用相同 marker 平均表达和置信差阈值分配四种恶性状态。"""
    index = {gene: i for i, gene in enumerate(genes)}
    score_expression = np.asarray(expression)
    if STANDARDIZE_MARKER_GENES:
        score_expression = score_expression.astype(float, copy=False)
        standard_deviation = score_expression.std(axis=0)
        standard_deviation[standard_deviation == 0] = 1.0
        score_expression = (score_expression - score_expression.mean(axis=0)) / standard_deviation
    score_matrix = np.column_stack(
        [score_expression[:, [index[gene] for gene in markers]].mean(axis=1) for markers in MARKERS.values()]
    )
    order = np.argsort(score_matrix, axis=1)
    top = order[:, -1]
    confidence = score_matrix[np.arange(len(score_matrix)), order[:, -1]] - score_matrix[np.arange(len(score_matrix)), order[:, -2]]
    labels = np.asarray(STATES, dtype=object)[top]
    labels[confidence < MIN_SCORE_DIFF] = "Ambiguous"
    return labels, confidence, score_matrix[:, STATES.index("AC-like")]


def read_h5ad_malignant(
    path: Path,
    cohort: str,
    patient_col: str,
    region_col: str,
    dataset_col: str,
    dataset_filter: str,
    malignant_mask,
) -> pd.DataFrame:
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs.copy()
    keep = obs[dataset_col].astype(str).eq(dataset_filter).to_numpy() & np.asarray(malignant_mask(obs), dtype=bool)
    obs_index = np.flatnonzero(keep)
    missing = [gene for gene in SELECTED_GENES if gene not in adata.var_names]
    if missing:
        adata.file.close()
        raise ValueError(f"{cohort} 缺少统一注释基因: {missing}")
    gene_index = adata.var_names.get_indexer(SELECTED_GENES)
    expression = dense(adata.X[obs_index, :][:, gene_index]).astype(np.float32)
    labels, confidence, ac_score = assign_uniform_states(expression, SELECTED_GENES)
    tgfb2 = expression[:, SELECTED_GENES.index("TGFB2")]
    selected = obs.iloc[obs_index]
    result = pd.DataFrame(
        {
            "cohort": cohort,
            "patient": selected[patient_col].astype(str).to_numpy(),
            "region": selected[region_col].astype(str).to_numpy(),
            "uniform_state": labels,
            "confidence": confidence,
            "ac_marker_score": ac_score,
            "TGFB2": tgfb2,
        }
    )
    adata.file.close()
    return result


def read_gse211376() -> pd.DataFrame:
    meta = pd.read_csv(META_211, index_col=0)
    with gzip.open(COUNTS_211, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        cells = header[1:] if header[0] == "" else header
        n_cells = len(cells)
        total = np.zeros(n_cells, dtype=np.float64)
        selected = {gene: np.zeros(n_cells, dtype=np.float32) for gene in SELECTED_GENES}
        seen: set[str] = set()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            gene = parts[0]
            values = np.asarray(parts[1:], dtype=np.float32)
            if len(values) != n_cells:
                raise ValueError(f"GSE211376 行长度错误: {gene}")
            total += values
            if gene in selected:
                selected[gene] = values
                seen.add(gene)
    missing = sorted(set(SELECTED_GENES) - seen)
    if missing:
        raise ValueError(f"GSE211376 缺少统一注释基因: {missing}")
    scale = total / 1e4
    scale[scale == 0] = 1.0
    expression = np.column_stack([np.log1p(selected[gene] / scale) for gene in SELECTED_GENES]).astype(np.float32)
    meta = meta.reindex(cells)
    malignant = meta["predicted.high_hierarchy"].isin(STATES).to_numpy()
    labels, confidence, ac_score = assign_uniform_states(expression[malignant], SELECTED_GENES)
    return pd.DataFrame(
        {
            "cohort": "GSE211376",
            "patient": meta.loc[malignant, "patient"].astype(str).to_numpy(),
            "region": "Tumor",
            "uniform_state": labels,
            "confidence": confidence,
            "ac_marker_score": ac_score,
            "TGFB2": expression[malignant, SELECTED_GENES.index("TGFB2")],
        }
    )


def patient_contrasts(cells: pd.DataFrame) -> pd.DataFrame:
    """按患者比较统一 AC-like 与其他统一恶性状态。"""
    rows = []
    eligible = cells[cells["uniform_state"] != "Ambiguous"].copy()
    for cohort, cohort_frame in eligible.groupby("cohort", observed=True):
        target_region = "Core" if cohort == "GSE274987" else "Tumor"
        cohort_frame = cohort_frame[cohort_frame["region"] == target_region]
        for patient, frame in cohort_frame.groupby("patient", observed=True):
            ac = frame[frame["uniform_state"] == "AC-like"]
            other = frame[frame["uniform_state"].isin(["MES-like", "OPC-like", "NPC-like"])]
            if len(ac) < MIN_CELLS or len(other) < MIN_CELLS:
                continue
            rows.append(
                {
                    "cohort": cohort,
                    "patient": patient,
                    "region": target_region,
                    "n_ac": len(ac),
                    "n_other": len(other),
                    "tgfb2_ac": float(ac["TGFB2"].mean()),
                    "tgfb2_other": float(other["TGFB2"].mean()),
                    "tgfb2_difference": float(ac["TGFB2"].mean() - other["TGFB2"].mean()),
                    "ac_score_difference": float(ac["ac_marker_score"].mean() - other["ac_marker_score"].mean()),
                }
            )
    return pd.DataFrame(rows)


def cohort_results(patient: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, frame in patient.groupby("cohort", observed=True):
        diff = frame["tgfb2_difference"].to_numpy(float)
        rows.append(
            {
                "cohort": cohort,
                "n_patients": len(frame),
                "median_difference": float(np.median(diff)),
                "positive_patients": int((diff > 0).sum()),
                "p_value": safe_wilcoxon(diff),
            }
        )
    results = pd.DataFrame(rows)
    results["q_value"] = bh_adjust(results["p_value"])
    return results


def combine_meta(frame: pd.DataFrame) -> tuple[float, float]:
    p = np.clip(frame["p_value"].to_numpy(float), 1e-15, 1.0)
    signs = np.sign(frame["median_difference"].to_numpy(float))
    z = norm.isf(p / 2) * signs
    weights = np.sqrt(frame["n_patients"].to_numpy(float))
    meta_z = float(np.sum(weights * z) / np.sqrt(np.sum(weights**2)))
    return meta_z, float(2 * norm.sf(abs(meta_z)))


def meta_result(cohorts: pd.DataFrame) -> dict[str, object]:
    valid = cohorts.dropna(subset=["p_value"])
    valid = valid[valid["n_patients"] >= 3]
    z, p = combine_meta(valid)
    loo = []
    for omitted in valid["cohort"]:
        loo_z, loo_p = combine_meta(valid[valid["cohort"] != omitted])
        loo.append({"omitted": omitted, "z": loo_z, "p_value": loo_p})
    return {
        "n_cohorts": len(valid),
        "n_patients": int(valid["n_patients"].sum()),
        "positive_cohorts": int((valid["median_difference"] > 0).sum()),
        "meta_z": z,
        "p_value": p,
        "loo_max_p_value": max(item["p_value"] for item in loo),
        "loo_results": loo,
    }


def lineage_adjustment(patient: pd.DataFrame) -> dict[str, object]:
    """检验 TGFB2 差值是否完全由 AC marker 差值解释。"""
    inferential = patient.groupby("cohort")["patient"].transform("count") >= 3
    frame = patient[inferential].copy()
    frame["ac_score_centered"] = frame["ac_score_difference"] - frame["ac_score_difference"].mean()
    model = smf.ols(
        "tgfb2_difference ~ ac_score_centered + C(cohort, Sum)", data=frame
    ).fit(cov_type="HC3")
    rho, rho_p = spearmanr(frame["tgfb2_difference"], frame["ac_score_difference"])
    ci = model.conf_int().loc["Intercept"]
    return {
        "n_patients": len(frame),
        "grand_mean_at_average_ac_score": float(model.params["Intercept"]),
        "hc3_p_value": float(model.pvalues["Intercept"]),
        "hc3_ci_low": float(ci.iloc[0]),
        "hc3_ci_high": float(ci.iloc[1]),
        "ac_score_slope": float(model.params["ac_score_centered"]),
        "ac_score_slope_p": float(model.pvalues["ac_score_centered"]),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "interpretation": "Sensitivity model only; uniform states are intentionally defined by the AC marker program.",
    }


def main() -> None:
    required = [COUNTS_211, META_211, SCGBM, SNGBM, GSE274]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少公共输入：\n" + "\n".join(missing))

    parts = [read_gse211376()]
    for accession in ["GSE162631", "GSE273274"]:
        parts.append(
            read_h5ad_malignant(
                SCGBM,
                cohort=accession,
                patient_col="patient_id",
                region_col="tissue_type",
                dataset_col="dataset",
                dataset_filter=accession,
                malignant_mask=lambda obs: obs["Major_category"].astype(str).eq("Tumor"),
            )
        )
    for accession in ["GSE276177", "GSE286419"]:
        parts.append(
            read_h5ad_malignant(
                SNGBM,
                cohort=accession,
                patient_col="patient",
                region_col="tissue",
                dataset_col="dataset",
                dataset_filter=accession,
                malignant_mask=lambda obs: obs["celltype"].astype(str).isin(STATES),
            )
        )
    parts.append(
        read_h5ad_malignant(
            GSE274,
            cohort="GSE274987",
            patient_col="patient_id",
            region_col="tissue_type",
            dataset_col="dataset",
            dataset_filter="GSE274987",
            malignant_mask=lambda obs: obs["celltype"].astype(str).isin(STATES),
        )
    )

    cells = pd.concat(parts, ignore_index=True)
    patient = patient_contrasts(cells)
    cohorts = cohort_results(patient)
    meta = meta_result(cohorts)
    adjusted = lineage_adjustment(patient)
    summary = {
        "uniform_marker_assignment": True,
        "tgfb2_not_used_for_state_definition": True,
        "minimum_score_difference": MIN_SCORE_DIFF,
        "minimum_cells_per_group": MIN_CELLS,
        "marker_gene_standardization": STANDARDIZE_MARKER_GENES,
        "meta": meta,
        "lineage_adjustment": adjusted,
        "cohorts": cohorts.to_dict(orient="records"),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    patient.to_csv(OUT / "uniform_annotation_patient_contrasts.csv", index=False)
    cohorts.to_csv(OUT / "uniform_annotation_cohort_results.csv", index=False)
    (OUT / "uniform_annotation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
