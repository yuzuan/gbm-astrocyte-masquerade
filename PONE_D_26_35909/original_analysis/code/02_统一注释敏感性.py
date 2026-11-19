#!/usr/bin/env python3
# 目的：用同一组 marker 在各公共队列内重新判定四种恶性状态，检查患者 TGFB2 差值对注释方式的敏感性。
# 输入：GSE211376_raw_counts.tsv.gz、GSE211376_metadata.csv.gz（patient、predicted.high_hierarchy）；
# scGBM_GSE162631_GSE273274.h5ad、snGBM_GSE276177_GSE286419.h5ad、regional_GSE274987_GSE241037.h5ad
# 的 X、dataset、患者、区域和原恶性细胞注释，按 PUBLIC_INPUT_ROOT 下 raw_inputs/、processed_inputs/ 读取。
# 输出：重算包 data/uniform_annotation_patient_contrasts.csv、data/uniform_annotation_cohort_results.csv、
# data/uniform_annotation_summary.json。独立发布包的重算目录为 01_公共队列重算/。
# 运行方式：在 code 目录运行 python 02_统一注释敏感性.py。保留 Python 以复用患者分析和 statsmodels HC3 模型。

from __future__ import annotations

from importlib import import_module
from itertools import chain
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

primary = import_module("01_公共队列患者表达分析")
paths = import_module("00a_项目路径")


def main() -> None:
    """从同一批恶性细胞生成患者表、队列表和谱系评分调整结果。无返回值。"""
    primary.stop_if_missing()
    cell_table_total = load_uniform_cells()
    patient_table = patient_contrasts(cell_table_total)
    cohort_table = cohort_results(patient_table)
    summary = {
        "uniform_marker_assignment": True,
        "tgfb2_not_used_for_state_definition": all("TGFB2" not in genes for genes in markers.values()),
        "minimum_score_difference": minimum_score_difference,
        "minimum_cells_per_group": primary.minimum_cells,
        "marker_gene_standardization": False,
        "meta": meta_result(cohort_table),
        "lineage_adjustment": lineage_adjustment(patient_table),
        "cohorts": cohort_table.to_dict("records"),
    }

    # 每名合格患者应贡献且只贡献一个核心区域差值，核对人数后写出两种粒度的表。
    if int(cohort_table["n_patients"].sum()) != len(patient_table):
        raise ValueError("队列表的患者人数与患者差值表不一致")
    output = paths.analysis_output_root() / "data"
    paths.write_table(patient_table, output / "uniform_annotation_patient_contrasts.csv", ["cohort", "patient"])
    paths.write_table(cohort_table, output / "uniform_annotation_cohort_results.csv", ["cohort"])
    paths.write_json(summary, output / "uniform_annotation_summary.json")
    print(f"获得 {len(patient_table)} 个患者差值，{summary['meta']['n_cohorts']} 个队列可合并。")


# 基准方案用最高与次高 marker 均值相差至少 0.05 判定状态。03 单独改变此阈值。
minimum_score_difference = 0.05


def load_uniform_cells(standardize: bool = False, score_threshold: float = minimum_score_difference) -> pd.DataFrame:
    """各队列独立打分，按原顺序合并细胞。scGBM 的敏感性分析同时纳入 GSE162631 和 GSE273274。"""
    parts = [read_gse211376(standardize, score_threshold)]
    files = paths.primary_input_files()
    cohorts = [
        ("GSE162631", "scGBM", "patient_id", "tissue_type", sc_malignant),
        ("GSE273274", "scGBM", "patient_id", "tissue_type", sc_malignant),
        ("GSE276177", "snGBM", "patient", "tissue", celltype_malignant),
        ("GSE286419", "snGBM", "patient", "tissue", celltype_malignant),
        ("GSE274987", "GSE274987", "patient_id", "tissue_type", celltype_malignant),
    ]
    for cohort, source, patient_column, region_column, select_cells in cohorts:
        cell_table = read_h5ad_malignant(
            files[source],
            cohort,
            patient_column,
            region_column,
            "dataset",
            cohort,
            select_cells,
            standardize=standardize,
            score_threshold=score_threshold,
        )
        parts.append(cell_table)
    cell_table_total = pd.concat(parts, ignore_index=True)
    return cell_table_total


def sc_malignant(obs_table: pd.DataFrame) -> pd.Series:
    """scGBM 统一注释的候选细胞取 Major_category 原注释为 Tumor 的行。"""
    selected = obs_table["Major_category"].astype(str).eq("Tumor")
    return selected


def celltype_malignant(obs_table: pd.DataFrame) -> pd.Series:
    """单核和区域对象只在原四种恶性状态内重新打分。"""
    selected = obs_table["celltype"].astype(str).isin(primary.malignant_states)
    return selected


def read_gse211376(standardize: bool = False, score_threshold: float = minimum_score_difference) -> pd.DataFrame:
    """读 GEO 原计数，先选原始四种恶性状态，再在这些细胞内计算 marker 分数。"""
    metadata_table, matrix, _ = primary.read_count_expression(selected_genes)
    keep = metadata_table["predicted.high_hierarchy"].isin(primary.malignant_states).to_numpy()
    selected_table = metadata_table.loc[keep, ["patient"]].copy()
    selected_table["region"] = "Tumor"
    cell_table = scored_cells(
        matrix[keep], selected_table, "GSE211376", "patient", "region", standardize, score_threshold
    )
    return cell_table


def read_h5ad_malignant(
    path: Path,
    cohort: str,
    patient_col: str,
    region_col: str,
    dataset_col: str,
    dataset_filter: str,
    malignant_mask,
    *,
    standardize: bool = False,
    score_threshold: float = minimum_score_difference,
) -> pd.DataFrame:
    """按给定 dataset 和恶性标签从 X 读目标基因；输出保留原患者、区域和 TGFB2 表达。"""
    source = ad.read_h5ad(path, backed="r")
    try:
        obs_table = source.obs
        keep = obs_table[dataset_col].astype(str).eq(dataset_filter).to_numpy()
        positions = np.flatnonzero(keep & np.asarray(malignant_mask(obs_table), dtype=bool))
        matrix = primary.selected_expression(source, positions, selected_genes)
        selected_table = obs_table.iloc[positions]
    finally:
        source.file.close()
    cell_table = scored_cells(matrix, selected_table, cohort, patient_col, region_col, standardize, score_threshold)
    return cell_table


def scored_cells(
    matrix: np.ndarray,
    obs_table: pd.DataFrame,
    cohort: str,
    patient_column: str,
    region_column: str,
    standardize: bool,
    score_threshold: float,
) -> pd.DataFrame:
    """一行一个恶性细胞。原尺度 TGFB2 为结局，标准化选项只作用于 marker 打分。"""
    labels, confidence, ac_scores = assign_uniform_states(matrix, selected_genes, standardize, score_threshold)
    cell_table = pd.DataFrame(
        {
            "cohort": cohort,
            "patient": obs_table[patient_column].astype(str).to_numpy(),
            "region": obs_table[region_column].astype(str).to_numpy(),
            "uniform_state": labels,
            "confidence": confidence,
            "ac_marker_score": ac_scores,
            "TGFB2": matrix[:, selected_genes.index("TGFB2")],
        }
    )
    return cell_table


# 四组 marker 和状态顺序均沿用发表方案。TGFB2 只供读出，CD44 同时属于 MES marker 和主分析受体。
markers = {
    "AC-like": ["GFAP", "S100B", "HOPX", "SLC1A3", "MLC1"],
    "MES-like": ["VIM", "CD44", "CHI3L1", "HILPDA", "DDIT3", "ENO2"],
    "OPC-like": ["PLP1", "OLIG1", "OMG", "TNR", "ALCAM"],
    "NPC-like": ["SOX4", "SOX11", "DCX", "CD24", "STMN1", "STMN2"],
}
selected_genes = list(dict.fromkeys(["TGFB2", *chain.from_iterable(markers.values())]))


def assign_uniform_states(
    expression: np.ndarray,
    genes: list[str],
    standardize: bool = False,
    score_threshold: float = minimum_score_difference,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回最高分状态、最高与次高分的差、AC 分数；小于阈值的状态记为 Ambiguous。"""
    score_expression = np.asarray(expression)
    if standardize:
        score_expression = score_expression.astype(float, copy=False)
        deviations = score_expression.std(axis=0)
        deviations[deviations == 0] = 1.0
        score_expression = (score_expression - score_expression.mean(axis=0)) / deviations

    # 每列对应一种状态，每行对应一个细胞。沿用 argsort 取末两位，维持并列分数的历史处理。
    positions = {gene: i for i, gene in enumerate(genes)}
    program_scores = []
    for program in markers.values():
        columns = [positions[gene] for gene in program]
        program_scores.append(score_expression[:, columns].mean(axis=1))
    score_matrix = np.column_stack(program_scores)
    order = np.argsort(score_matrix, axis=1)
    rows = np.arange(len(score_matrix))
    confidence = score_matrix[rows, order[:, -1]] - score_matrix[rows, order[:, -2]]
    states = list(markers)
    labels = np.asarray(states, dtype=object)[order[:, -1]]
    labels[confidence < score_threshold] = "Ambiguous"
    result = labels, confidence, score_matrix[:, states.index("AC-like")]
    return result


def patient_contrasts(cell_table: pd.DataFrame, minimum_cells: int = primary.minimum_cells) -> pd.DataFrame:
    """每队列每位患者一行；敏感性比较将三个其他状态合并后才使用细胞数阈值。"""
    rows = []
    eligible = cell_table["uniform_state"] != "Ambiguous"
    core = cell_table["cohort"].eq("GSE274987")
    target = (core & cell_table["region"].eq("Core")) | (~core & cell_table["region"].eq("Tumor"))
    eligible_table = cell_table.loc[eligible & target]
    for (cohort, patient), patient_table in eligible_table.groupby(["cohort", "patient"], observed=True):
        ac_table = patient_table.loc[patient_table["uniform_state"] == "AC-like"]
        other_table = patient_table.loc[patient_table["uniform_state"].isin(["MES-like", "OPC-like", "NPC-like"])]
        if len(ac_table) < minimum_cells or len(other_table) < minimum_cells:
            continue

        # 先在原 float32 精度下求差，再转 Python float 写表；标准化模式的 AC 分数可为 float64。
        ac_mean = ac_table["TGFB2"].mean()
        other_mean = other_table["TGFB2"].mean()
        score_difference = ac_table["ac_marker_score"].mean() - other_table["ac_marker_score"].mean()
        rows.append(
            {
                "cohort": cohort,
                "patient": patient,
                "region": "Core" if cohort == "GSE274987" else "Tumor",
                "n_ac": len(ac_table),
                "n_other": len(other_table),
                "tgfb2_ac": float(ac_mean),
                "tgfb2_other": float(other_mean),
                "tgfb2_difference": float(ac_mean - other_mean),
                "ac_score_difference": float(score_difference),
            }
        )
    patient_table = pd.DataFrame.from_records(
        rows,
        columns=[
            "cohort",
            "patient",
            "region",
            "n_ac",
            "n_other",
            "tgfb2_ac",
            "tgfb2_other",
            "tgfb2_difference",
            "ac_score_difference",
        ],
    )
    return patient_table


def cohort_results(patient_table: pd.DataFrame) -> pd.DataFrame:
    """按队列汇总患者差值，保留不足三人的描述行，对可估计的队列 P 值作 BH 校正。"""
    rows = []
    for cohort, cohort_table in patient_table.groupby("cohort", observed=True):
        differences = cohort_table["tgfb2_difference"].to_numpy(float)
        rows.append(
            {
                "cohort": cohort,
                "n_patients": len(cohort_table),
                "median_difference": float(np.median(differences)),
                "positive_patients": int((differences > 0).sum()),
                "p_value": primary.safe_wilcoxon(differences),
            }
        )
    cohort_table_total = pd.DataFrame.from_records(
        rows, columns=["cohort", "n_patients", "median_difference", "positive_patients", "p_value"]
    )
    cohort_table_total["q_value"] = primary.bh_adjust(cohort_table_total["p_value"])
    return cohort_table_total


def meta_result(cohort_table: pd.DataFrame) -> dict:
    """合并至少三位患者且有 P 值的队列；每轮删除一个完整队列并记录其名称。"""
    valid_table = cohort_table.loc[cohort_table["p_value"].notna() & cohort_table["n_patients"].ge(3)]
    meta_z, meta_p = primary.combine_meta(valid_table)
    leave_one_out = []
    for omitted in valid_table["cohort"]:
        z_value, p_value = primary.combine_meta(valid_table.loc[valid_table["cohort"] != omitted])
        leave_one_out.append({"omitted": omitted, "z": z_value, "p_value": p_value})
    result = {
        "n_cohorts": len(valid_table),
        "n_patients": int(valid_table["n_patients"].sum()),
        "positive_cohorts": int((valid_table["median_difference"] > 0).sum()),
        "meta_z": meta_z,
        "p_value": meta_p,
        "loo_max_p_value": max((row["p_value"] for row in leave_one_out), default=np.nan),
        "loo_results": leave_one_out,
    }
    return result


def lineage_adjustment(patient_table: pd.DataFrame) -> dict:
    """在有至少三位患者的队列中拟合调整 AC 分数的敏感性模型，输出 HC3 区间和秩相关。"""
    import statsmodels.formula.api as smf

    inferential = patient_table.groupby("cohort")["patient"].transform("count") >= 3
    model_table = patient_table.loc[inferential].copy()
    model_table["ac_score_centered"] = model_table["ac_score_difference"] - model_table["ac_score_difference"].mean()

    # Sum 编码使截距对应各队列等权的水平；自变量在纳入患者均值处中心化。
    model = smf.ols("tgfb2_difference ~ ac_score_centered + C(cohort, Sum)", data=model_table)
    fit = model.fit(cov_type="HC3")
    rho, rho_p = spearmanr(model_table["tgfb2_difference"], model_table["ac_score_difference"])
    intervals_table = fit.conf_int()
    lower, upper = intervals_table.loc["Intercept"]
    result = {
        "n_patients": len(model_table),
        "grand_mean_at_average_ac_score": float(fit.params["Intercept"]),
        "hc3_p_value": float(fit.pvalues["Intercept"]),
        "hc3_ci_low": float(lower),
        "hc3_ci_high": float(upper),
        "ac_score_slope": float(fit.params["ac_score_centered"]),
        "ac_score_slope_p": float(fit.pvalues["ac_score_centered"]),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "interpretation": "Sensitivity model only; uniform states are intentionally defined by the AC marker program.",
    }
    return result


if __name__ == "__main__":
    main()
