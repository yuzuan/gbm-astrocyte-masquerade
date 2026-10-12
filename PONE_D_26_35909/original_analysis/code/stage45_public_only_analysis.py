#!/usr/bin/env python3
"""仅使用公共数据集重算 AC-like GBM 与髓系细胞的患者级关联。"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm, wilcoxon

from stage_public_paths import project_root, public_input_root, submission_package


ROOT = project_root()
PACKAGE = submission_package(ROOT)
INPUT_ROOT = public_input_root(ROOT)
OUT = PACKAGE / "data"

INPUTS = {
    "GSE211376_counts": INPUT_ROOT / "raw_inputs" / "GSE211376_raw_counts.tsv.gz",
    "GSE211376_meta": INPUT_ROOT / "raw_inputs" / "GSE211376_metadata.csv.gz",
    "scGBM": INPUT_ROOT / "processed_inputs" / "scGBM_GSE162631_GSE273274.h5ad",
    "snGBM": INPUT_ROOT / "processed_inputs" / "snGBM_GSE276177_GSE286419.h5ad",
    "GSE274987": INPUT_ROOT / "processed_inputs" / "regional_GSE274987_GSE241037.h5ad",
}

INPUT_DESCRIPTORS = {
    "GSE211376_counts": ("GSE211376", "GEO raw count matrix", "raw_inputs/GSE211376_raw_counts.tsv.gz"),
    "GSE211376_meta": ("GSE211376", "GEO cell metadata", "raw_inputs/GSE211376_metadata.csv.gz"),
    "scGBM": ("GSE162631; GSE273274", "project-generated processed AnnData", "processed_inputs/scGBM_GSE162631_GSE273274.h5ad"),
    "snGBM": ("GSE276177; GSE286419", "project-generated processed AnnData", "processed_inputs/snGBM_GSE276177_GSE286419.h5ad"),
    "GSE274987": (
        "GSE274986; GSE274987; GSE241037",
        (
            "project-generated processed AnnData; joint GSE274986/GSE274987 "
            "cohort selected through legacy dataset value GSE274987; "
            "GSE241037 excluded"
        ),
        "processed_inputs/regional_GSE274987_GSE241037.h5ad",
    ),
}

MALIGNANT_STATES = ["AC-like", "MES-like", "OPC-like", "NPC-like"]
LIGANDS = ["TGFB2", "TGFB1", "ANXA1", "SPP1", "CSF1", "CCL2"]
RECEPTORS = ["TGFBR1", "TGFBR2", "FPR1", "CD44", "CSF1R", "CCR2"]
GENES = list(dict.fromkeys(LIGANDS + RECEPTORS))
PAIRS = [
    ("TGFB2", "TGFBR1"),
    ("TGFB2", "TGFBR2"),
    ("TGFB1", "TGFBR1"),
    ("ANXA1", "FPR1"),
    ("SPP1", "CD44"),
    ("CSF1", "CSF1R"),
    ("CCL2", "CCR2"),
]
MIN_CELLS = 20


def stop_if_missing() -> None:
    """输入不完整时立即停止，禁止使用旧汇总值替代。"""
    missing = [f"{name}: {path}" for name, path in INPUTS.items() if not path.is_file()]
    if missing:
        print("ERROR: 缺少公共数据输入文件：", file=sys.stderr)
        print("\n".join(missing), file=sys.stderr)
        raise SystemExit(1)
    for name, path in INPUTS.items():
        try:
            with path.open("rb") as handle:
                handle.read(1)
        except OSError as exc:
            print(f"ERROR: 无法读取 {name}: {path}: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc


def bh_adjust(values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg 校正，保留缺失值。"""
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


def safe_wilcoxon(diff: np.ndarray) -> float:
    """患者配对差值的双侧 Wilcoxon 检验。"""
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) < 3 or np.allclose(diff, 0):
        return np.nan
    return float(wilcoxon(diff, alternative="two-sided", method="auto").pvalue)


def minimum_attainable_two_sided_p(n_pairs: int) -> float:
    """无零差值时双侧精确符号秩检验可达到的理论最小 P 值。"""
    if n_pairs < 1:
        return np.nan
    return float(2 ** (1 - n_pairs))


def count_nonzero_pairs(diff: np.ndarray) -> int:
    """统计 Wilcoxon 实际使用的非零患者配对数。"""
    values = np.asarray(diff, dtype=float)
    values = values[np.isfinite(values)]
    return int(np.count_nonzero(~np.isclose(values, 0.0)))


def matrix_to_dense(matrix) -> np.ndarray:
    """将选定基因的小矩阵转为二维数组。"""
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def aggregate_selected_expression(
    expression: np.ndarray,
    metadata: pd.DataFrame,
    genes: list[str],
) -> pd.DataFrame:
    """按患者、区域和细胞状态汇总平均表达与检出率。"""
    rows: list[dict[str, object]] = []
    group_cols = ["cohort", "dataset", "patient", "region", "state"]
    for keys, index in metadata.groupby(group_cols, observed=True).groups.items():
        loc = metadata.index.get_indexer(index)
        block = expression[loc]
        row: dict[str, object] = dict(zip(group_cols, keys))
        row["n_cells"] = int(len(loc))
        for j, gene in enumerate(genes):
            values = block[:, j]
            row[f"mean_{gene}"] = float(np.mean(values))
            row[f"detect_{gene}"] = float(np.mean(values > 0))
        rows.append(row)
    return pd.DataFrame(rows)


def read_h5ad_cohort(
    path: Path,
    cohort: str,
    dataset_filter: str | None,
    patient_col: str,
    region_col: str,
    dataset_col: str,
    state_map: dict[str, str],
    state_col: str,
    myeloid_mask,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从公共 h5ad 中提取恶性状态和髓系细胞。"""
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs.copy()
    keep_dataset = np.ones(len(obs), dtype=bool)
    if dataset_filter is not None:
        keep_dataset = obs[dataset_col].astype(str).eq(dataset_filter).to_numpy()

    mapped = obs[state_col].astype(str).map(state_map)
    is_malignant = mapped.notna().to_numpy()
    is_myeloid = np.asarray(myeloid_mask(obs), dtype=bool)
    keep = keep_dataset & (is_malignant | is_myeloid)
    obs_index = np.flatnonzero(keep)

    missing_genes = [gene for gene in GENES if gene not in adata.var_names]
    if missing_genes:
        adata.file.close()
        raise ValueError(f"{cohort} 缺少基因: {missing_genes}")
    gene_index = adata.var_names.get_indexer(GENES)
    expression = matrix_to_dense(adata.X[obs_index, :][:, gene_index]).astype(np.float32)

    selected = obs.iloc[obs_index].copy()
    selected["cohort"] = cohort
    selected["dataset"] = selected[dataset_col].astype(str)
    selected["patient"] = selected[patient_col].astype(str)
    selected["region"] = selected[region_col].astype(str)
    selected["state"] = mapped.iloc[obs_index].fillna("Myeloid").to_numpy()
    selected = selected.reset_index(drop=True)
    summary = aggregate_selected_expression(expression, selected, GENES)

    # 组成比例只在四种恶性状态内部计算，避免测序捕获差异混入分母。
    malignant = selected[selected["state"].isin(MALIGNANT_STATES)]
    composition = (
        malignant.groupby(["cohort", "dataset", "patient", "region", "state"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    totals = composition.groupby(["cohort", "dataset", "patient", "region"], observed=True)["n_cells"].transform("sum")
    composition["fraction_of_malignant"] = composition["n_cells"] / totals
    adata.file.close()
    return summary, composition


def read_gse211376() -> tuple[pd.DataFrame, pd.DataFrame]:
    """流式读取 GSE211376 原始计数，避免载入完整矩阵。"""
    meta = pd.read_csv(INPUTS["GSE211376_meta"], index_col=0)
    with gzip.open(INPUTS["GSE211376_counts"], "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header[0] == "":
            cells = header[1:]
        else:
            cells = header
        n_cells = len(cells)
        total = np.zeros(n_cells, dtype=np.float64)
        selected = {gene: np.zeros(n_cells, dtype=np.float32) for gene in GENES}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            gene = parts[0]
            values = np.asarray(parts[1:], dtype=np.float32)
            if len(values) != n_cells:
                raise ValueError(f"GSE211376 行长度不一致: {gene}, {len(values)} != {n_cells}")
            total += values
            if gene in selected:
                selected[gene] = values

    missing = [gene for gene, values in selected.items() if not np.any(values)]
    if missing:
        raise ValueError(f"GSE211376 未读到非零基因: {missing}")
    scale = total / 1e4
    scale[scale == 0] = 1.0
    expression = np.column_stack([np.log1p(selected[gene] / scale) for gene in GENES]).astype(np.float32)

    meta = meta.reindex(cells)
    if meta.isna().any(axis=None):
        raise ValueError("GSE211376 计数列与元数据未完全对齐")
    state_map = {
        "AC-like": "AC-like",
        "MES-like": "MES-like",
        "OPC-like": "OPC-like",
        "NPC-like": "NPC-like",
        "TAM-MG": "Myeloid",
        "TAM-BDM": "Myeloid",
    }
    state = meta["predicted.high_hierarchy"].map(state_map)
    keep = state.notna().to_numpy()
    selected_meta = pd.DataFrame(
        {
            "cohort": "GSE211376",
            "dataset": "GSE211376",
            "patient": meta["patient"].astype(str),
            "region": "Tumor",
            "state": state,
        }
    ).iloc[np.flatnonzero(keep)].reset_index(drop=True)
    summary = aggregate_selected_expression(expression[keep], selected_meta, GENES)
    composition = (
        selected_meta[selected_meta["state"].isin(MALIGNANT_STATES)]
        .groupby(["cohort", "dataset", "patient", "region", "state"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    totals = composition.groupby(["cohort", "dataset", "patient", "region"], observed=True)["n_cells"].transform("sum")
    composition["fraction_of_malignant"] = composition["n_cells"] / totals
    return summary, composition


def ac_vs_other_contrasts(expression: pd.DataFrame) -> pd.DataFrame:
    """比较患者内 AC-like 与其余恶性状态，禁止使用细胞级重复。"""
    records: list[dict[str, object]] = []
    malignant = expression[expression["state"].isin(MALIGNANT_STATES)].copy()
    for (cohort, region), frame in malignant.groupby(["cohort", "region"], observed=True):
        # 主比较限定肿瘤核心；GSE274987 使用 Core，其余使用 Tumor。
        target_region = "Core" if cohort == "GSE274987" else "Tumor"
        if region != target_region:
            continue
        for gene in LIGANDS:
            diffs: list[float] = []
            patient_rows: list[dict[str, object]] = []
            for patient, patient_frame in frame.groupby("patient", observed=True):
                ac = patient_frame[(patient_frame["state"] == "AC-like") & (patient_frame["n_cells"] >= MIN_CELLS)]
                other = patient_frame[(patient_frame["state"] != "AC-like") & (patient_frame["n_cells"] >= MIN_CELLS)]
                if ac.empty or other.empty:
                    continue
                ac_mean = float(ac.iloc[0][f"mean_{gene}"])
                weights = other["n_cells"].to_numpy(float)
                other_mean = float(np.average(other[f"mean_{gene}"].to_numpy(float), weights=weights))
                diff = ac_mean - other_mean
                diffs.append(diff)
                patient_rows.append({"patient": patient, "ac_mean": ac_mean, "other_mean": other_mean, "difference": diff})
            records.append(
                {
                    "cohort": cohort,
                    "region": region,
                    "gene": gene,
                    "n_patients": len(diffs),
                    "median_difference": float(np.median(diffs)) if diffs else np.nan,
                    "mean_difference": float(np.mean(diffs)) if diffs else np.nan,
                    "p_value": safe_wilcoxon(np.asarray(diffs)),
                    "patient_values": json.dumps(patient_rows, ensure_ascii=False),
                }
            )
    results = pd.DataFrame(records)
    results["q_value"] = results.groupby("cohort", observed=True)["p_value"].transform(bh_adjust)
    return results


def communication_scores(expression: pd.DataFrame) -> pd.DataFrame:
    """构建患者-区域级配体受体共可用指数，仅作关联量而非因果通讯。"""
    records: list[dict[str, object]] = []
    keys = ["cohort", "dataset", "patient", "region"]
    for group_keys, frame in expression.groupby(keys, observed=True):
        ac = frame[(frame["state"] == "AC-like") & (frame["n_cells"] >= MIN_CELLS)]
        myeloid = frame[(frame["state"] == "Myeloid") & (frame["n_cells"] >= MIN_CELLS)]
        if ac.empty or myeloid.empty:
            continue
        for ligand, receptor in PAIRS:
            ligand_mean = float(ac.iloc[0][f"mean_{ligand}"])
            receptor_mean = float(myeloid.iloc[0][f"mean_{receptor}"])
            records.append(
                {
                    **dict(zip(keys, group_keys)),
                    "pair": f"{ligand}-{receptor}",
                    "ligand": ligand,
                    "receptor": receptor,
                    "n_ac": int(ac.iloc[0]["n_cells"]),
                    "n_myeloid": int(myeloid.iloc[0]["n_cells"]),
                    "ligand_mean": ligand_mean,
                    "receptor_mean": receptor_mean,
                    "coavailability": float(np.sqrt(max(ligand_mean, 0) * max(receptor_mean, 0))),
                }
            )
    return pd.DataFrame(records)


def meta_ligand_results(contrasts: pd.DataFrame) -> pd.DataFrame:
    """用带方向的 Stouffer 方法合并独立公共队列，不合并细胞。"""
    def combine(frame: pd.DataFrame) -> tuple[float, float]:
        p = np.clip(frame["p_value"].to_numpy(float), 1e-15, 1.0)
        signs = np.sign(frame["median_difference"].to_numpy(float))
        z = norm.isf(p / 2) * signs
        weights = np.sqrt(frame["n_patients"].to_numpy(float))
        meta_z = float(np.sum(weights * z) / np.sqrt(np.sum(weights**2)))
        return meta_z, float(2 * norm.sf(abs(meta_z)))

    rows: list[dict[str, object]] = []
    for gene, frame in contrasts.groupby("gene", observed=True):
        valid = frame.dropna(subset=["p_value", "median_difference"]).copy()
        valid = valid[valid["n_patients"] >= 3]
        if valid.empty:
            continue
        meta_z, meta_p = combine(valid)
        loo = []
        if len(valid) >= 3:
            for omitted in valid["cohort"]:
                loo_frame = valid[valid["cohort"] != omitted]
                loo_z, loo_p = combine(loo_frame)
                loo.append({"omitted": omitted, "meta_z": loo_z, "p_value": loo_p})
        rows.append(
            {
                "gene": gene,
                "n_cohorts": int(len(valid)),
                "n_patients_total": int(valid["n_patients"].sum()),
                "positive_cohorts": int((valid["median_difference"] > 0).sum()),
                "negative_cohorts": int((valid["median_difference"] < 0).sum()),
                "meta_z": meta_z,
                "p_value": meta_p,
                "loo_max_p_value": float(max(item["p_value"] for item in loo)) if loo else np.nan,
                "loo_results": json.dumps(loo, ensure_ascii=False),
                "cohort_effects": json.dumps(
                    valid[["cohort", "n_patients", "median_difference", "p_value"]].to_dict(orient="records"),
                    ensure_ascii=False,
                ),
            }
        )
    results = pd.DataFrame(rows)
    results["q_value"] = bh_adjust(results["p_value"])
    results["loo_q_value"] = bh_adjust(results["loo_max_p_value"])
    return results.sort_values("q_value").reset_index(drop=True)


def regional_pair_tests(scores: pd.DataFrame) -> pd.DataFrame:
    """按患者配对比较区域，不将细胞数当作样本量。"""
    comparisons = {
        "GSE273274": ("Tumor", "Peritumoral"),
        "GSE276177": ("Tumor", "Peritumoral"),
        "GSE286419": ("Tumor", "Peritumoral"),
        "GSE274987_core_peripheral": ("Core", "Peripheral"),
        "GSE274987_core_edge": ("Core", "Edge"),
    }
    records: list[dict[str, object]] = []
    for label, (region_a, region_b) in comparisons.items():
        cohort = "GSE274987" if label.startswith("GSE274987") else label
        frame = scores[scores["cohort"] == cohort]
        for pair, pair_frame in frame.groupby("pair", observed=True):
            wide = pair_frame.pivot_table(index="patient", columns="region", values="coavailability", aggfunc="mean")
            if region_a not in wide or region_b not in wide:
                diff = np.array([], dtype=float)
            else:
                paired = wide[[region_a, region_b]].dropna()
                diff = paired[region_a].to_numpy(float) - paired[region_b].to_numpy(float)
            n_nonzero = count_nonzero_pairs(diff)
            minimum_p = minimum_attainable_two_sided_p(n_nonzero)
            records.append(
                {
                    "comparison": label,
                    "cohort": cohort,
                    "region_a": region_a,
                    "region_b": region_b,
                    "pair": pair,
                    "n_patients": len(diff),
                    "n_nonzero_pairs": n_nonzero,
                    "minimum_attainable_p": minimum_p,
                    "can_attain_p_below_0_05": bool(minimum_p < 0.05) if n_nonzero else False,
                    "median_difference": float(np.median(diff)) if len(diff) else np.nan,
                    "mean_difference": float(np.mean(diff)) if len(diff) else np.nan,
                    "p_value": safe_wilcoxon(diff),
                    "patient_differences": json.dumps(diff.tolist()),
                }
            )
    results = pd.DataFrame(records)
    results["q_value"] = results.groupby("comparison", observed=True)["p_value"].transform(bh_adjust)
    return results


def composition_tests(composition: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """输出 AC-like 比例并做患者内区域配对比较。"""
    ac = composition[composition["state"] == "AC-like"].copy()
    comparisons = {
        "GSE273274": ("Tumor", "Peritumoral"),
        "GSE276177": ("Tumor", "Peritumoral"),
        "GSE286419": ("Tumor", "Peritumoral"),
        "GSE274987_core_peripheral": ("Core", "Peripheral"),
        "GSE274987_core_edge": ("Core", "Edge"),
    }
    rows: list[dict[str, object]] = []
    for label, (region_a, region_b) in comparisons.items():
        cohort = "GSE274987" if label.startswith("GSE274987") else label
        frame = ac[ac["cohort"] == cohort]
        wide = frame.pivot_table(index="patient", columns="region", values="fraction_of_malignant", aggfunc="mean")
        if region_a in wide and region_b in wide:
            paired = wide[[region_a, region_b]].dropna()
            diff = paired[region_a].to_numpy(float) - paired[region_b].to_numpy(float)
        else:
            diff = np.array([], dtype=float)
        n_nonzero = count_nonzero_pairs(diff)
        minimum_p = minimum_attainable_two_sided_p(n_nonzero)
        rows.append(
            {
                "comparison": label,
                "cohort": cohort,
                "region_a": region_a,
                "region_b": region_b,
                "n_patients": len(diff),
                "n_nonzero_pairs": n_nonzero,
                "minimum_attainable_p": minimum_p,
                "can_attain_p_below_0_05": bool(minimum_p < 0.05) if n_nonzero else False,
                "median_difference": float(np.median(diff)) if len(diff) else np.nan,
                "p_value": safe_wilcoxon(diff),
                "patient_differences": json.dumps(diff.tolist()),
            }
        )
    results = pd.DataFrame(rows)
    results["q_value"] = bh_adjust(results["p_value"])
    return ac, results


def build_manifest(expression: pd.DataFrame) -> pd.DataFrame:
    """从实际读入的数据生成队列清单。"""
    accession = {
        "GSE211376": "GSE211376",
        "GSE273274": "GSE273274",
        "GSE276177": "GSE276177",
        "GSE286419": "GSE286419",
        "GSE274987": "GSE274986; GSE274987",
    }
    scope_notes = {
        "GSE211376": "11 tumor patients analyzed",
        "GSE273274": "3 matched tumor-peritumoral patients analyzed",
        "GSE274987": (
            "Joint Wu et al. regional cohort: 6 GSE274986 Multiome patients and "
            "4 GSE274987 ParseBio patients analyzed together; the legacy "
            "internal cohort label is GSE274987"
        ),
        "GSE276177": "2 matched patients analyzed; the source study also reported one tumor-only patient",
        "GSE286419": "4 matched patients from the public confident-sample subset analyzed; the source study reported 6 patients",
    }
    rows = []
    for cohort, frame in expression.groupby("cohort", observed=True):
        rows.append(
            {
                "cohort": cohort,
                "accessions": accession[cohort],
                "patients": int(frame["patient"].nunique()),
                "regions": "; ".join(sorted(frame["region"].unique())),
                "included_cells": int(frame.groupby(["patient", "region", "state"], observed=True)["n_cells"].first().sum()),
                "analysis_unit": "patient-region",
                "analysis_scope_note": scope_notes[cohort],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    stop_if_missing()
    OUT.mkdir(parents=True, exist_ok=True)

    expression_parts: list[pd.DataFrame] = []
    composition_parts: list[pd.DataFrame] = []

    exp, comp = read_gse211376()
    expression_parts.append(exp)
    composition_parts.append(comp)

    exp, comp = read_h5ad_cohort(
        INPUTS["scGBM"],
        cohort="GSE273274",
        dataset_filter="GSE273274",
        patient_col="patient_id",
        region_col="tissue_type",
        dataset_col="dataset",
        state_col="Minor_cluster",
        state_map={"AC": "AC-like", "MES1": "MES-like", "OPC": "OPC-like", "NPC1": "NPC-like", "NPC2": "NPC-like"},
        myeloid_mask=lambda obs: obs["Major_category"].astype(str).isin(["TAM/Macrophage", "Microglia", "Monocyte"]),
    )
    expression_parts.append(exp)
    composition_parts.append(comp)

    for accession in ["GSE276177", "GSE286419"]:
        exp, comp = read_h5ad_cohort(
            INPUTS["snGBM"],
            cohort=accession,
            dataset_filter=accession,
            patient_col="patient",
            region_col="tissue",
            dataset_col="dataset",
            state_col="celltype",
            state_map={state: state for state in MALIGNANT_STATES},
            myeloid_mask=lambda obs: obs["celltype"].astype(str).isin(["Macrophage", "Microglia", "Monocyte"]),
        )
        expression_parts.append(exp)
        composition_parts.append(comp)

    exp, comp = read_h5ad_cohort(
        INPUTS["GSE274987"],
        cohort="GSE274987",
        dataset_filter="GSE274987",
        patient_col="patient_id",
        region_col="tissue_type",
        dataset_col="dataset",
        state_col="celltype",
        state_map={state: state for state in MALIGNANT_STATES},
        myeloid_mask=lambda obs: obs["celltype"].astype(str).isin(["Macrophage", "Microglia", "Monocyte"]),
    )
    expression_parts.append(exp)
    composition_parts.append(comp)

    expression = pd.concat(expression_parts, ignore_index=True)
    composition = pd.concat(composition_parts, ignore_index=True)
    contrasts = ac_vs_other_contrasts(expression)
    meta_ligands = meta_ligand_results(contrasts)
    scores = communication_scores(expression)
    regional = regional_pair_tests(scores)
    ac_composition, composition_results = composition_tests(composition)
    manifest = build_manifest(expression)

    expression.to_csv(OUT / "patient_state_expression.csv", index=False)
    composition.to_csv(OUT / "patient_state_composition.csv", index=False)
    ac_composition.to_csv(OUT / "patient_ac_fraction.csv", index=False)
    contrasts.to_csv(OUT / "ac_vs_other_ligand_contrasts.csv", index=False)
    meta_ligands.to_csv(OUT / "meta_ligand_results.csv", index=False)
    scores.to_csv(OUT / "patient_communication_scores.csv", index=False)
    regional.to_csv(OUT / "regional_communication_tests.csv", index=False)
    composition_results.to_csv(OUT / "regional_ac_fraction_tests.csv", index=False)
    manifest.to_csv(OUT / "cohort_manifest.csv", index=False)

    provenance_rows = []
    for name, path in INPUTS.items():
        accessions, input_type, repository_target = INPUT_DESCRIPTORS[name]
        provenance_rows.append(
            {
                "input": name,
                "accessions": accessions,
                "input_type": input_type,
                "source_file_name": path.name,
                "repository_target_path": repository_target,
                "bytes": path.stat().st_size,
            }
        )
    provenance = pd.DataFrame(provenance_rows)
    provenance.to_csv(OUT / "input_provenance.csv", index=False)

    excluded = pd.DataFrame(
        [
            {
                "dataset": "GSE162631",
                "reason": "Processed public object did not provide a harmonized four-state malignant annotation; AC-like counts were below the predefined 20-cell threshold in several patient-regions.",
                "status": "excluded from primary label-based analysis; included in uniform-marker sensitivity analysis",
            },
            {
                "dataset": "GSE174554",
                "reason": "The prior intermediate clustering object was unavailable, so the earlier cluster-specific results could not be regenerated from the recorded workflow.",
                "status": "excluded; no legacy summary values reused",
            },
        ]
    )
    excluded.to_csv(OUT / "excluded_datasets.csv", index=False)

    summary = {
        "public_only": True,
        "minimum_cells_per_patient_region_state": MIN_CELLS,
        "analysis_unit": "patient-region",
        "cohorts": manifest.to_dict(orient="records"),
        "significant_ac_ligand_contrasts_q_lt_0_05": contrasts.loc[contrasts["q_value"] < 0.05, ["cohort", "gene", "median_difference", "q_value"]].to_dict(orient="records"),
        "meta_ligands_q_lt_0_05": meta_ligands.loc[meta_ligands["q_value"] < 0.05, ["gene", "n_cohorts", "n_patients_total", "positive_cohorts", "meta_z", "q_value"]].to_dict(orient="records"),
        "significant_regional_communication_q_lt_0_05": regional.loc[regional["q_value"] < 0.05, ["comparison", "pair", "median_difference", "q_value"]].to_dict(orient="records"),
        "regional_communication_tests_capable_of_raw_p_lt_0_05": int(regional["can_attain_p_below_0_05"].sum()),
        "interpretation_boundary": "Expression co-availability is an association index and is not evidence of physical signaling or causality.",
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
