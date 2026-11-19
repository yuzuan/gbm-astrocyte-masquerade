#!/usr/bin/env python3
# 目的：在每位患者内比较 AC-like 与其他恶性状态的配体表达，计算区域组成和配体/受体共可用指数。
# 输入：raw_inputs/GSE211376_raw_counts.tsv.gz、GSE211376_metadata.csv.gz（patient、predicted.high_hierarchy）；
# processed_inputs 下 scGBM_GSE162631_GSE273274.h5ad、snGBM_GSE276177_GSE286419.h5ad、
# regional_GSE274987_GSE241037.h5ad 的 X、dataset、患者、区域及原注释列。原研究范围说明引自 TableS1_public_cohort_manifest.csv。
# 输出：重算包 data/ 下 patient_state_expression.csv、patient_state_composition.csv、patient_ac_fraction.csv、
# ac_vs_other_ligand_contrasts.csv、meta_ligand_results.csv、patient_communication_scores.csv、
# regional_communication_tests.csv、regional_ac_fraction_tests.csv、cohort_manifest.csv、
# input_provenance.csv、excluded_datasets.csv、analysis_summary.json。沿用发表文件名供下游读取。
# 运行方式：在 code 目录运行 python 01_公共队列患者表达分析.py；PUBLIC_INPUT_ROOT 指向原始公共输入。
# 独立发布包的重算文件放在 01_公共队列重算/，14 号脚本可直接复用下列患者级函数。

from __future__ import annotations

import gzip
import json
from importlib import import_module
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm, wilcoxon

paths = import_module("00a_项目路径")


def main() -> None:
    """从原始公共输入生成患者级结果和来源清单，无返回值。"""
    stop_if_missing()
    expression_table_total, composition_table_total = read_primary_cohorts()
    tables = primary_tables(expression_table_total, composition_table_total)
    output = paths.analysis_output_root() / "data"

    # 这八张表也由精简输入复现。行标识同时用于写表回读和按患者、区域、基因对照。
    for name, result_table in tables.items():
        paths.write_table(result_table, output / f"{name}.csv", primary_row_columns[name])

    # 清单依据实际进入患者分析的细胞。来源文件的大小取自本轮输入。
    manifest_table = build_manifest(expression_table_total)
    paths.write_table(manifest_table, output / "cohort_manifest.csv", ["cohort"])
    provenance_table = input_provenance()
    paths.write_table(provenance_table, output / "input_provenance.csv", ["input"])

    # 两个排除理由是发表时的队列决定，保留原字段，供数据来源表和引用它的图表使用。
    excluded_table = pd.DataFrame(
        [
            {
                "dataset": "GSE162631",
                "reason": (
                    "Processed public object did not provide a harmonized four-state malignant annotation; "
                    "AC-like counts were below the predefined 20-cell threshold in several patient-regions."
                ),
                "status": "excluded from primary label-based analysis; included in uniform-marker sensitivity analysis",
            },
            {
                "dataset": "GSE174554",
                "reason": (
                    "The prior intermediate clustering object was unavailable, so the earlier cluster-specific "
                    "results could not be regenerated from the recorded workflow."
                ),
                "status": "excluded; no legacy summary values reused",
            },
        ]
    )
    paths.write_table(excluded_table, output / "excluded_datasets.csv", ["dataset"])

    # 用未取位 q 值判断显著性；区域检验还要区分在现有配对数下是否有机会达到名义阈值。
    contrast_table = tables["ac_vs_other_ligand_contrasts"]
    meta_table = tables["meta_ligand_results"]
    regional_table = tables["regional_communication_tests"]
    significant_contrast_table = contrast_table.loc[
        contrast_table["q_value"] < 0.05, ["cohort", "gene", "median_difference", "q_value"]
    ]
    significant_meta_table = meta_table.loc[
        meta_table["q_value"] < 0.05, ["gene", "n_cohorts", "n_patients_total", "positive_cohorts", "meta_z", "q_value"]
    ]
    significant_region_table = regional_table.loc[
        regional_table["q_value"] < 0.05, ["comparison", "pair", "median_difference", "q_value"]
    ]
    summary = {
        "public_only": True,
        "minimum_cells_per_patient_region_state": minimum_cells,
        "analysis_unit": "patient-region",
        "cohorts": manifest_table.to_dict("records"),
        "significant_ac_ligand_contrasts_q_lt_0_05": significant_contrast_table.to_dict("records"),
        "meta_ligands_q_lt_0_05": significant_meta_table.to_dict("records"),
        "significant_regional_communication_q_lt_0_05": significant_region_table.to_dict("records"),
        "regional_communication_tests_capable_of_raw_p_lt_0_05": int(regional_table["can_attain_p_below_0_05"].sum()),
        "interpretation_boundary": (
            "Expression co-availability is an association index and is not evidence of physical signaling or causality."
        ),
    }
    paths.write_json(summary, output / "analysis_summary.json")
    print(f"纳入 {len(manifest_table)} 个队列，得到 {len(contrast_table)} 个队列配体比较。")


def stop_if_missing() -> None:
    """在读取首个大矩阵前列出所有缺失输入，无返回值。"""
    missing = [str(path) for path in paths.primary_input_files().values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少公共分析输入：\n" + "\n".join(missing))


def read_primary_cohorts(minimal_directory: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按原顺序读五个队列；传入精简目录时直接使用已归一化的 X。"""
    if minimal_directory is None:
        files = paths.primary_input_files()
        expression_table, composition_table = read_gse211376()
    else:
        files = {
            "scGBM": minimal_directory / "scGBM_minimal.h5ad",
            "snGBM": minimal_directory / "snGBM_minimal.h5ad",
            "GSE274987": minimal_directory / "regional_GSE274987_minimal.h5ad",
        }
        expression_table, composition_table = read_h5ad_cohort(
            minimal_directory / "GSE211376_minimal.h5ad",
            "GSE211376",
            "GSE211376",
            "patient",
            "region",
            "dataset",
            state_labels,
            "predicted.high_hierarchy",
            tam_myeloid,
        )
    expression_parts = [expression_table]
    composition_parts = [composition_table]

    # 单细胞主比较仅取 GSE273274。区域对象按历史 dataset 值选人，两个测序平台一起作为一个队列。
    cohorts = [
        ("GSE273274", "scGBM", "patient_id", "tissue_type", "Minor_cluster", sc_state_labels, sc_myeloid),
        ("GSE276177", "snGBM", "patient", "tissue", "celltype", state_labels, celltype_myeloid),
        ("GSE286419", "snGBM", "patient", "tissue", "celltype", state_labels, celltype_myeloid),
        ("GSE274987", "GSE274987", "patient_id", "tissue_type", "celltype", state_labels, celltype_myeloid),
    ]
    for cohort, source, patient, region, state, labels, myeloid_mask in cohorts:
        expression_table, composition_table = read_h5ad_cohort(
            files[source], cohort, cohort, patient, region, "dataset", labels, state, myeloid_mask
        )
        expression_parts.append(expression_table)
        composition_parts.append(composition_table)

    expression_table_total = pd.concat(expression_parts, ignore_index=True)
    composition_table_total = pd.concat(composition_parts, ignore_index=True)
    result = expression_table_total, composition_table_total
    return result


# 标签次序也用于统一 marker 分数并列时的排序；scGBM 的两个 NPC 标签归入同一状态。
malignant_states = ["AC-like", "MES-like", "OPC-like", "NPC-like"]
state_labels = {state: state for state in malignant_states}
sc_state_labels = {"AC": "AC-like", "MES1": "MES-like", "OPC": "OPC-like", "NPC1": "NPC-like", "NPC2": "NPC-like"}


def tam_myeloid(obs_table: pd.DataFrame) -> pd.Series:
    """从 GSE211376 发表标签标记两类 TAM，返回与细胞同行的布尔向量。"""
    selected = obs_table["predicted.high_hierarchy"].astype(str).isin(["TAM-MG", "TAM-BDM"])
    return selected


def sc_myeloid(obs_table: pd.DataFrame) -> pd.Series:
    """按 scGBM 的大类标记髓系细胞，返回布尔向量。"""
    selected = obs_table["Major_category"].astype(str).isin(["TAM/Macrophage", "Microglia", "Monocyte"])
    return selected


def celltype_myeloid(obs_table: pd.DataFrame) -> pd.Series:
    """按 snGBM 和区域对象的 celltype 标记髓系细胞，返回布尔向量。"""
    selected = obs_table["celltype"].astype(str).isin(["Macrophage", "Microglia", "Monocyte"])
    return selected


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
    """按给定列、状态映射和髓系筛选取 X；缺基因时停止，返回患者状态表达与恶性细胞组成。"""
    source = ad.read_h5ad(path, backed="r")
    try:
        obs_table = source.obs
        states = obs_table[state_col].astype(str).map(state_map)
        selected = states.notna().to_numpy() | np.asarray(myeloid_mask(obs_table), dtype=bool)
        if dataset_filter is not None:
            selected &= obs_table[dataset_col].astype(str).eq(dataset_filter).to_numpy()

        # 若原标注同时符合两个条件，沿用已知恶性状态；仅有髓系标签的细胞归为 Myeloid。
        positions = np.flatnonzero(selected)
        matrix = selected_expression(source, positions, primary_genes)
        selected_table = obs_table.iloc[positions]
        cell_table = pd.DataFrame(
            {
                "cohort": cohort,
                "dataset": selected_table[dataset_col].astype(str).to_numpy(),
                "patient": selected_table[patient_col].astype(str).to_numpy(),
                "region": selected_table[region_col].astype(str).to_numpy(),
                "state": states.iloc[positions].fillna("Myeloid").to_numpy(),
            }
        )
    finally:
        source.file.close()

    result = summarize_states(matrix, cell_table)
    return result


def selected_expression(source, positions: np.ndarray, genes: list[str]) -> np.ndarray:
    """从已打开的 AnnData 按传入的行序、基因序取小矩阵；任何目标基因缺失即停止。"""
    columns = source.var_names.get_indexer(genes)
    missing = [gene for gene, column in zip(genes, columns) if column < 0]
    if missing:
        raise ValueError(f"表达矩阵缺少基因 {missing}")
    matrix = source.X[positions, :][:, columns]
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    expression = np.asarray(matrix, dtype=np.float32)
    return expression


# 六个配体及七个预先指定的配对。去重保持初次出现顺序，保证矩阵列与发表表头相同。
ligands = ["TGFB2", "TGFB1", "ANXA1", "SPP1", "CSF1", "CCL2"]
ligand_receptor_pairs = [
    ("TGFB2", "TGFBR1"),
    ("TGFB2", "TGFBR2"),
    ("TGFB1", "TGFBR1"),
    ("ANXA1", "FPR1"),
    ("SPP1", "CD44"),
    ("CSF1", "CSF1R"),
    ("CCL2", "CCR2"),
]
primary_genes = list(dict.fromkeys(ligands + [receptor for _, receptor in ligand_receptor_pairs]))


def read_gse211376() -> tuple[pd.DataFrame, pd.DataFrame]:
    """原始计数归一化后取原注释状态；主分析沿用目标基因须有非零计数的条件。"""
    metadata_table, matrix, _ = read_count_expression(primary_genes)
    if metadata_table.isna().any(axis=None):
        raise ValueError("GSE211376 的计数列与完整元数据未对齐")
    if not np.all(np.any(matrix != 0, axis=0)):
        raise ValueError("GSE211376 主分析目标基因含全零列")

    labels = {**state_labels, "TAM-MG": "Myeloid", "TAM-BDM": "Myeloid"}
    states = metadata_table["predicted.high_hierarchy"].map(labels)
    keep = states.notna().to_numpy()
    cell_table = pd.DataFrame(
        {
            "cohort": "GSE211376",
            "dataset": "GSE211376",
            "patient": metadata_table.loc[keep, "patient"].astype(str),
            "region": "Tumor",
            "state": states.loc[keep],
        }
    )
    result = summarize_states(matrix[keep], cell_table.reset_index(drop=True))
    return result


def read_count_expression(genes: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """流式读 GSE211376；返回按计数列排序的元数据、目标基因 log1p 表达及全基因总计数。"""
    files = paths.primary_input_files()
    metadata_table = pd.read_csv(files["GSE211376_meta"], index_col=0)
    with gzip.open(files["GSE211376_counts"], "rt") as stream:
        header = stream.readline().rstrip("\n").split("\t")
        cells = header[1:] if header[0] == "" else header
        totals = np.zeros(len(cells), dtype=np.float64)
        gene_counts = {}
        for line in stream:
            symbol, *fields = line.rstrip("\n").split("\t")
            values = np.asarray(fields, dtype=np.float32)
            if len(values) != len(cells):
                raise ValueError(f"GSE211376 {symbol} 的计数列数与表头不符")
            totals += values
            if symbol in genes:
                gene_counts[symbol] = values

    # 分母来自全部基因；只把指定基因留在内存中。总计数为零的细胞保留零表达。
    missing = sorted(set(genes) - gene_counts.keys())
    if missing:
        raise ValueError(f"GSE211376 缺少目标基因 {missing}")
    scale = totals / 1e4
    scale[scale == 0] = 1.0
    matrix = np.column_stack([np.log1p(gene_counts[gene] / scale) for gene in genes]).astype(np.float32)

    metadata_table = metadata_table.reindex(cells)
    if metadata_table[["predicted.high_hierarchy", "patient"]].isna().any(axis=None):
        raise ValueError("GSE211376 的计数列缺少对应患者或细胞注释")
    result = metadata_table, matrix, totals
    return result


def summarize_states(matrix: np.ndarray, cell_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """由同一批细胞给出两张表；组成分母限定为该患者该区域所有已注释恶性状态。"""
    expression_table = aggregate_selected_expression(matrix, cell_table, primary_genes)
    malignant = expression_table["state"].isin(malignant_states)
    composition_table = expression_table.loc[malignant, [*patient_region, "state", "n_cells"]].copy()
    denominators = composition_table.groupby(patient_region, observed=True)["n_cells"].transform("sum")
    composition_table["fraction_of_malignant"] = composition_table["n_cells"] / denominators
    result = expression_table, composition_table.reset_index(drop=True)
    return result


patient_region = ["cohort", "dataset", "patient", "region"]


def aggregate_selected_expression(
    expression: np.ndarray, metadata_table: pd.DataFrame, genes: list[str]
) -> pd.DataFrame:
    """矩阵一行为一个细胞。按患者、区域、状态计算每个基因的平均值和非零细胞比例。"""
    if expression.shape != (len(metadata_table), len(genes)):
        raise ValueError("细胞元数据、基因列表和表达矩阵维度不符")
    group_columns = [*patient_region, "state"]
    rows = []
    for group, positions in metadata_table.groupby(group_columns, observed=True).indices.items():
        block = expression[positions]
        row = dict(zip(group_columns, group))
        row["n_cells"] = len(positions)
        for column, gene in enumerate(genes):
            values = block[:, column]
            row[f"mean_{gene}"] = float(np.mean(values))
            row[f"detect_{gene}"] = float(np.mean(values > 0))
        rows.append(row)
    expression_table = pd.DataFrame(rows)
    return expression_table


def primary_tables(expression_table: pd.DataFrame, composition_table: pd.DataFrame) -> dict:
    """由两张患者状态表生成八个发表结果；检查组成分母和状态细胞数。"""
    totals = composition_table.groupby(patient_region, observed=True)["fraction_of_malignant"].sum()
    if totals.empty or not np.allclose(totals, 1) or (expression_table["n_cells"] <= 0).any():
        raise ValueError("恶性状态的区域组成之和应为 1，每个状态应有细胞")

    contrast_table = ac_vs_other_contrasts(expression_table)
    score_table = communication_scores(expression_table)
    ac_table, fraction_test_table = composition_tests(composition_table)
    tables = {
        "patient_state_expression": expression_table,
        "patient_state_composition": composition_table,
        "ac_vs_other_ligand_contrasts": contrast_table,
        "meta_ligand_results": meta_ligand_results(contrast_table),
        "patient_communication_scores": score_table,
        "regional_communication_tests": regional_pair_tests(score_table),
        "patient_ac_fraction": ac_table,
        "regional_ac_fraction_tests": fraction_test_table,
    }
    return tables


# 同一患者的不同区域分别计数。配体比较一行是一队列、一区域、一个基因。
primary_row_columns = {
    "patient_state_expression": ["cohort", "patient", "region", "state"],
    "patient_state_composition": ["cohort", "patient", "region", "state"],
    "ac_vs_other_ligand_contrasts": ["cohort", "region", "gene"],
    "meta_ligand_results": ["gene"],
    "patient_communication_scores": ["cohort", "patient", "region", "pair"],
    "regional_communication_tests": ["cohort", "comparison", "pair"],
    "patient_ac_fraction": ["cohort", "patient", "region", "state"],
    "regional_ac_fraction_tests": ["cohort", "comparison"],
}


def ac_vs_other_contrasts(expression_table: pd.DataFrame) -> pd.DataFrame:
    """以患者为配对单位，先按各个状态的细胞数筛选，再对其余合格状态按细胞数加权。"""
    records = []
    malignant_table = expression_table.loc[expression_table["state"].isin(malignant_states)]
    for (cohort, region), region_table in malignant_table.groupby(["cohort", "region"], observed=True):
        if region != ("Core" if cohort == "GSE274987" else "Tumor"):
            continue
        eligible_table = region_table.loc[region_table["n_cells"] >= minimum_cells]
        for gene in ligands:
            values = patient_ligand_differences(eligible_table, gene)
            differences = [row["difference"] for row in values]
            records.append(
                {
                    "cohort": cohort,
                    "region": region,
                    "gene": gene,
                    "n_patients": len(differences),
                    "median_difference": float(np.median(differences)) if differences else np.nan,
                    "mean_difference": float(np.mean(differences)) if differences else np.nan,
                    "p_value": safe_wilcoxon(differences),
                    "patient_values": json.dumps(values, ensure_ascii=False),
                }
            )
    contrast_table = pd.DataFrame(records)
    contrast_table["q_value"] = contrast_table.groupby("cohort", observed=True)["p_value"].transform(bh_adjust)
    return contrast_table


# 发表方案要求同一患者区域至少有这么多 AC 细胞；主比较对每个其他状态也逐一应用此阈值。
minimum_cells = 20


def patient_ligand_differences(eligible_table: pd.DataFrame, gene: str) -> list[dict]:
    """每位患者只保留一个 AC 均值和一个按细胞数加权的其他状态均值。"""
    records = []
    for patient, patient_table in eligible_table.groupby("patient", observed=True):
        ac_table = patient_table.loc[patient_table["state"] == "AC-like"]
        other_table = patient_table.loc[patient_table["state"] != "AC-like"]
        if ac_table.empty or other_table.empty:
            continue
        ac_mean = float(ac_table[f"mean_{gene}"].item())
        weights = other_table["n_cells"].to_numpy(float)
        other_mean = float(np.average(other_table[f"mean_{gene}"].to_numpy(float), weights=weights))
        records.append(
            {"patient": patient, "ac_mean": ac_mean, "other_mean": other_mean, "difference": ac_mean - other_mean}
        )
    return records


def safe_wilcoxon(diff: np.ndarray) -> float:
    """双侧配对 Wilcoxon；不足三名有限差值或差值全接近零时保留缺失。"""
    values = np.asarray(diff, dtype=float)
    values = values[np.isfinite(values)]
    p_value = np.nan
    if len(values) >= 3 and not np.allclose(values, 0):
        p_value = float(wilcoxon(values, alternative="two-sided", method="auto").pvalue)
    return p_value


def bh_adjust(values: pd.Series) -> pd.Series:
    """在本次传入的检验族内作 BH 校正，缺失 P 值不参加排序并保留原位置。"""
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().to_numpy(float)
    if not len(valid):
        return adjusted
    order = np.argsort(valid)
    ranked = valid[order] * len(valid) / np.arange(1, len(valid) + 1)
    corrected = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty_like(corrected)
    restored[order] = np.clip(corrected, 0, 1)
    adjusted.loc[values.notna()] = restored
    return adjusted


def communication_scores(expression_table: pd.DataFrame) -> pd.DataFrame:
    """每位患者每个区域同时有足量 AC 和髓系细胞时，计算预设配对两端表达的几何均值。"""
    records = []
    eligible_table = expression_table.loc[expression_table["n_cells"] >= minimum_cells]
    for group, region_table in eligible_table.groupby(patient_region, observed=True):
        states_table = region_table.set_index("state", verify_integrity=True)
        if "AC-like" not in states_table.index or "Myeloid" not in states_table.index:
            continue
        for ligand, receptor in ligand_receptor_pairs:
            ligand_mean = float(states_table.at["AC-like", f"mean_{ligand}"])
            receptor_mean = float(states_table.at["Myeloid", f"mean_{receptor}"])
            records.append(
                {
                    **dict(zip(patient_region, group)),
                    "pair": f"{ligand}-{receptor}",
                    "ligand": ligand,
                    "receptor": receptor,
                    "n_ac": int(states_table.at["AC-like", "n_cells"]),
                    "n_myeloid": int(states_table.at["Myeloid", "n_cells"]),
                    "ligand_mean": ligand_mean,
                    "receptor_mean": receptor_mean,
                    "coavailability": float(np.sqrt(max(ligand_mean, 0) * max(receptor_mean, 0))),
                }
            )
    score_table = pd.DataFrame(records)
    return score_table


def meta_ligand_results(contrast_table: pd.DataFrame) -> pd.DataFrame:
    """每个基因合并有患者级 P 值的队列；至少三个可估计队列时计算逐一删队列结果。"""
    records = []
    for gene, gene_table in contrast_table.groupby("gene", observed=True):
        eligible_table = gene_table.dropna(subset=["p_value", "median_difference"])
        eligible_table = eligible_table.loc[eligible_table["n_patients"] >= 3]
        if eligible_table.empty:
            continue
        meta_z, p_value = combine_meta(eligible_table)

        # 删队列时重新计算分母。原分析分别校正六个主 P 值和六个最不利的删一 P 值。
        omitted_results = []
        if len(eligible_table) >= 3:
            for omitted in eligible_table["cohort"]:
                z_omitted, p_omitted = combine_meta(eligible_table.loc[eligible_table["cohort"] != omitted])
                omitted_results.append({"omitted": omitted, "meta_z": z_omitted, "p_value": p_omitted})
        effects = eligible_table[["cohort", "n_patients", "median_difference", "p_value"]].to_dict("records")
        records.append(
            {
                "gene": gene,
                "n_cohorts": len(eligible_table),
                "n_patients_total": int(eligible_table["n_patients"].sum()),
                "positive_cohorts": int((eligible_table["median_difference"] > 0).sum()),
                "negative_cohorts": int((eligible_table["median_difference"] < 0).sum()),
                "meta_z": meta_z,
                "p_value": p_value,
                "loo_max_p_value": max((row["p_value"] for row in omitted_results), default=np.nan),
                "loo_results": json.dumps(omitted_results, ensure_ascii=False),
                "cohort_effects": json.dumps(effects, ensure_ascii=False),
            }
        )
    meta_table = pd.DataFrame(records)
    meta_table["q_value"] = bh_adjust(meta_table["p_value"])
    meta_table["loo_q_value"] = bh_adjust(meta_table["loo_max_p_value"])
    meta_table = meta_table.sort_values("q_value").reset_index(drop=True)
    return meta_table


def combine_meta(cohort_table: pd.DataFrame) -> tuple[float, float]:
    """以队列中位差定方向、配对患者数平方根为权重合并双侧 P 值。空集返回缺失。"""
    if cohort_table.empty:
        result = np.nan, np.nan
        return result
    p_values = np.clip(cohort_table["p_value"].to_numpy(float), 1e-15, 1.0)
    signs = np.sign(cohort_table["median_difference"].to_numpy(float))
    z_scores = norm.isf(p_values / 2) * signs
    weights = np.sqrt(cohort_table["n_patients"].to_numpy(float))
    z_combined = float(np.sum(weights * z_scores) / np.sqrt(np.sum(weights**2)))
    result = z_combined, float(2 * norm.sf(abs(z_combined)))
    return result


def regional_pair_tests(score_table: pd.DataFrame) -> pd.DataFrame:
    """每一行比较同一配体受体对在两个区域的患者差值；在每个区域比较内校正 P 值。"""
    rows = []
    for label, cohort, region_a, region_b in regional_comparisons:
        cohort_table = score_table.loc[score_table["cohort"] == cohort]
        for pair, pair_table in cohort_table.groupby("pair", observed=True):
            statistics = paired_region_statistics(pair_table, "coavailability", region_a, region_b)
            rows.append(
                {
                    "comparison": label,
                    "cohort": cohort,
                    "region_a": region_a,
                    "region_b": region_b,
                    "pair": pair,
                    **statistics,
                }
            )
    test_table = pd.DataFrame(rows)
    test_table["q_value"] = test_table.groupby("comparison", observed=True)["p_value"].transform(bh_adjust)
    return test_table


# 方向固定为第一个区域减第二个。两个区域比较均来自同一联合队列。
regional_comparisons = [
    ("GSE273274", "GSE273274", "Tumor", "Peritumoral"),
    ("GSE276177", "GSE276177", "Tumor", "Peritumoral"),
    ("GSE286419", "GSE286419", "Tumor", "Peritumoral"),
    ("GSE274987_core_peripheral", "GSE274987", "Core", "Peripheral"),
    ("GSE274987_core_edge", "GSE274987", "Core", "Edge"),
]


def composition_tests(composition_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在有 AC 记录的患者区域间配对，不把缺席的 AC 状态补成零；所有组成检验用同一 BH 族。"""
    ac_table = composition_table.loc[composition_table["state"] == "AC-like"].copy()
    rows = []
    for label, cohort, region_a, region_b in regional_comparisons:
        cohort_table = ac_table.loc[ac_table["cohort"] == cohort]
        statistics = paired_region_statistics(cohort_table, "fraction_of_malignant", region_a, region_b)
        del statistics["mean_difference"]
        rows.append({"comparison": label, "cohort": cohort, "region_a": region_a, "region_b": region_b, **statistics})
    test_table = pd.DataFrame(rows)
    test_table["q_value"] = bh_adjust(test_table["p_value"])
    result = ac_table, test_table
    return result


def paired_region_statistics(region_table: pd.DataFrame, column: str, region_a: str, region_b: str) -> dict:
    """按患者对齐两个区域；无配对时报告零对和空统计量，保留发表的无编号差值数组。"""
    wide_table = region_table.pivot_table(index="patient", columns="region", values=column, aggfunc="mean")
    differences = np.array([], dtype=float)
    if region_a in wide_table and region_b in wide_table:
        paired_table = wide_table[[region_a, region_b]].dropna()
        differences = paired_table[region_a].to_numpy(float) - paired_table[region_b].to_numpy(float)

    # 理论下限以近似非零差值数计算；Wilcoxon 检验本身仍接收完整配对差值。
    nonzero = count_nonzero_pairs(differences)
    minimum_p = minimum_attainable_two_sided_p(nonzero)
    statistics = {
        "n_patients": len(differences),
        "n_nonzero_pairs": nonzero,
        "minimum_attainable_p": minimum_p,
        "can_attain_p_below_0_05": bool(minimum_p < 0.05),
        "median_difference": float(np.median(differences)) if len(differences) else np.nan,
        "mean_difference": float(np.mean(differences)) if len(differences) else np.nan,
        "p_value": safe_wilcoxon(differences),
        "patient_differences": json.dumps(differences.tolist()),
    }
    return statistics


def count_nonzero_pairs(diff: np.ndarray) -> int:
    """按原方案用 isclose 识别近零值，仅统计有限的患者配对。"""
    values = np.asarray(diff, dtype=float)
    finite = values[np.isfinite(values)]
    count = int(np.count_nonzero(~np.isclose(finite, 0.0)))
    return count


def minimum_attainable_two_sided_p(n_pairs: int) -> float:
    """由非零配对数得双侧符号秩检验的理论最小 P 值；无配对时留空。"""
    minimum = float(2 ** (1 - n_pairs)) if n_pairs > 0 else np.nan
    return minimum


def build_manifest(expression_table: pd.DataFrame) -> pd.DataFrame:
    """每队列一行，人数和细胞数取纳入记录；联合队列的来源人数取原对象中的测序平台。"""
    reference_path = Path(__file__).resolve().parents[1] / "source_data" / "TableS1_public_cohort_manifest.csv"
    reference_table = pd.read_csv(reference_path)
    source_notes = reference_table.set_index("cohort", verify_integrity=True)["analysis_scope_note"]
    rows = []
    for cohort, cohort_table in expression_table.groupby("cohort", observed=True):
        patient_count = cohort_table["patient"].nunique()
        accessions = cohort
        if cohort == "GSE274987":
            accessions = "GSE274986; GSE274987"
            platform_counts = regional_platform_counts(cohort_table["patient"])
            scope = (
                f"Joint Wu et al. regional cohort: {platform_counts['Multiome']} GSE274986 Multiome patients and "
                f"{platform_counts['ParseBio']} GSE274987 ParseBio patients analyzed together; the legacy "
                "internal cohort label is GSE274987"
            )
        elif cohort == "GSE211376":
            scope = f"{patient_count} tumor patients analyzed"
        else:
            tumor = cohort_table.loc[cohort_table["region"] == "Tumor", "patient"]
            peritumoral = cohort_table.loc[cohort_table["region"] == "Peritumoral", "patient"]
            matched = len(set(tumor) & set(peritumoral))
            scope = f"{matched} matched tumor-peritumoral patients analyzed"

            # 分号后是外部研究的来源描述。本轮配对人数来自两区域取交集。
            if cohort == "GSE276177":
                source_note = source_notes[cohort].split("; ", maxsplit=1)[1]
                scope = f"{matched} matched patients analyzed; {source_note}"
            elif cohort == "GSE286419":
                source_note = source_notes[cohort].split("; ", maxsplit=1)[1]
                scope = f"{matched} matched patients from the public confident-sample subset analyzed; {source_note}"

        counts = cohort_table.groupby(["patient", "region", "state"], observed=True)["n_cells"].first()
        rows.append(
            {
                "cohort": cohort,
                "accessions": accessions,
                "patients": int(patient_count),
                "regions": "; ".join(sorted(cohort_table["region"].unique())),
                "included_cells": int(counts.sum()),
                "analysis_unit": "patient-region",
                "analysis_scope_note": scope,
            }
        )
    manifest_table = pd.DataFrame(rows)
    return manifest_table


def regional_platform_counts(patients: pd.Series) -> pd.Series:
    """从联合对象中按平台统计本次入组患者；平台映射须唯一且覆盖这些患者。"""
    source = ad.read_h5ad(paths.primary_input_files()["GSE274987"], backed="r")
    try:
        obs_table = source.obs
        selected = obs_table["dataset"].astype(str).eq("GSE274987") & obs_table["patient_id"].astype(str).isin(patients)
        member_table = obs_table.loc[selected, ["patient_id", "Platform"]].astype(str).drop_duplicates()
    finally:
        source.file.close()
    if member_table["patient_id"].duplicated().any() or set(member_table["patient_id"]) != set(patients):
        raise ValueError("联合队列的患者与平台未一一对应")
    if set(member_table["Platform"]) != {"Multiome", "ParseBio"}:
        raise ValueError("联合队列平台无法对应两个发表来源")
    counts = member_table["Platform"].value_counts()
    return counts


def input_provenance() -> pd.DataFrame:
    """从本次读入的五个文件记录实际名称、相对位置和字节数，保留发表输入名。"""
    descriptions = {
        "GSE211376_counts": ("GSE211376", "GEO raw count matrix"),
        "GSE211376_meta": ("GSE211376", "GEO cell metadata"),
        "scGBM": ("GSE162631; GSE273274", "project-generated processed AnnData"),
        "snGBM": ("GSE276177; GSE286419", "project-generated processed AnnData"),
        "GSE274987": (
            "GSE274986; GSE274987; GSE241037",
            "project-generated processed AnnData; joint GSE274986/GSE274987 "
            "cohort selected through legacy dataset value GSE274987; GSE241037 excluded",
        ),
    }
    input_root = paths.public_input_root(paths.project_root())
    rows = []
    for name, path in paths.primary_input_files().items():
        accessions, description = descriptions[name]
        rows.append(
            {
                "input": name,
                "accessions": accessions,
                "input_type": description,
                "source_file_name": path.name,
                "repository_target_path": path.relative_to(input_root).as_posix(),
                "bytes": path.stat().st_size,
            }
        )
    provenance_table = pd.DataFrame(rows)
    return provenance_table


if __name__ == "__main__":
    main()
