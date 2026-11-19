#!/usr/bin/env python3
# 目的：在 GSE253080 原始 Visium 计数中比较 AC 样评分、六种配体与邻近髓系受体信号。
# 输入：--input-dir 下的 pd.csv（geo_accession、description、subject id:ch1）及每张切片的矩阵、
# features、barcodes、tissue_positions_list 四份压缩文件。Python 使用 SciPy 稀疏矩阵与原随机序列。
# 输出：默认在 external_validation/09_GSE253080空间信号 写 01_切片配体指标.csv、02_患者配体指标.csv、
# 03_切片空间指标.csv、04_患者空间指标.csv、05_配体汇总.csv、06_代表切片点位.csv、07_分析摘要.json、
# 08_FigureS7_GSE253080空间信号.pdf 和 08_FigureS7_GSE253080空间信号.png。
# 运行方式：python 09_GSE253080空间信号.py --input-dir GSE253080目录 [--output-dir 复算目录]

import argparse
import gzip
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import mmread

spatial = import_module("00a_空间分析")
files = import_module("00a_项目路径")


def main():
    """读取每张切片并以患者为单位汇总。样本或坐标不能对应、写出核对失败时停止，无返回值。"""
    external_dir = Path(__file__).resolve().parents[1] / "external_validation"
    parser = argparse.ArgumentParser(description="GSE253080 患者级空间敏感性分析")
    parser.add_argument("--input-dir", type=Path, default=external_dir / "raw/gse253080")
    parser.add_argument("--output-dir", type=Path, default=external_dir / Path(__file__).stem)
    args = parser.parse_args()

    # 一行对应一张切片。GEO 描述首段和 GSM 编号共同组成原始文件前缀。
    metadata_table = pd.read_csv(args.input_dir / "pd.csv", dtype={"geo_accession": str, "subject id:ch1": str})
    metadata_table["sample"] = metadata_table["description"].str.split("_").str[0]
    metadata_table = metadata_table.rename(columns={"subject id:ch1": "patient"})
    identity_table = metadata_table[["geo_accession", "sample", "patient"]]
    if identity_table.isna().any(axis=None) or metadata_table["sample"].duplicated().any() or metadata_table.empty:
        raise ValueError("pd.csv 缺少样本标识、重复登记切片或没有样本")

    # 三组评分各有独立的生物含义；六种配体只作为被检验的表达量。
    marker_sets = {
        "ac": spatial.ac_markers, "myeloid": ["AIF1", "CD163", "CSF1R", "LST1", "TYROBP", "FCER1G"],
        "receptor": spatial.receptors,
    }
    score_genes = sorted(set().union(*marker_sets.values()))
    target_genes = sorted(set(score_genes) | set(spatial.ligands))
    ligand_rows = []
    section_rows = []
    plot_tables = {}
    for row in metadata_table.itertuples(index=False):
        prefix = f"{row.geo_accession}_{row.sample}"
        matrix = mmread(args.input_dir / f"{prefix}_matrix.mtx.gz").tocsr()
        feature_table = pd.read_csv(
            args.input_dir / f"{prefix}_features.tsv.gz", sep="\t", header=None,
            names=["gene_id", "gene_symbol", "feature_type"]
        )
        with gzip.open(args.input_dir / f"{prefix}_barcodes.tsv.gz", "rt") as stream:
            barcodes = [line.rstrip("\n") for line in stream]

        # 原位置文件无表头。先按 barcode 对齐，再用组织内掩码同步筛选计数行与位置行。
        position_table = pd.read_csv(
            args.input_dir / f"{prefix}_tissue_positions_list.csv.gz", header=None,
            names=["barcode", "in_tissue", "array_row", "array_col", "pixel_row", "pixel_col"]
        )
        if matrix.shape == (len(feature_table), len(barcodes)):
            matrix = matrix.transpose().tocsr()
        elif matrix.shape != (len(barcodes), len(feature_table)):
            raise ValueError(f"{row.sample} 的矩阵维度与基因、条码数不一致")
        position_table = position_table.set_index("barcode", verify_integrity=True).reindex(barcodes)
        in_tissue = position_table["in_tissue"].fillna(0).to_numpy() == 1
        matrix = matrix[in_tissue]
        position_table = position_table.loc[in_tissue]

        # 重复基因符号先合计原始计数，全特征数归一到一万后取 log1p。零文库按原方案以 1 作分母。
        symbols = feature_table["gene_symbol"].astype(str).str.upper().to_numpy()
        counts = {}
        for gene in target_genes:
            columns = np.flatnonzero(symbols == gene)
            if len(columns):
                counts[gene] = np.asarray(matrix[:, columns].sum(axis=1)).ravel()
        library_sizes = np.maximum(np.asarray(matrix.sum(axis=1)).ravel(), 1)
        expression_table = pd.DataFrame(counts, index=position_table.index)
        expression_table = np.log1p(expression_table.div(library_sizes, axis=0) * 10_000)

        # 沿点位逐基因标准化。每组至少两个有变异的标志物才取均值，其余点位分数留空。
        standardized = {}
        for gene in score_genes:
            if gene in expression_table and expression_table[gene].std(ddof=0) > 0:
                values = expression_table[gene].to_numpy()
                standardized[gene] = (values - np.nanmean(values)) / np.nanstd(values)
        scores = {}
        for name, markers in marker_sets.items():
            available = [standardized[gene] for gene in markers if gene in standardized]
            scores[name] = np.full(len(position_table), np.nan)
            if len(available) >= 2:
                scores[name] = np.nanmean(np.column_stack(available), axis=1)

        # 髓系和受体两个模块再次在切片内标准化，对可估计分数取均值。
        components = []
        for name in ["myeloid", "receptor"]:
            values = scores[name]
            deviation = np.nanstd(values)
            component = np.full(len(values), np.nan)
            if np.isfinite(deviation) and deviation > 0:
                component = (values - np.nanmean(values)) / deviation
            components.append(component)
        receiver = np.nanmean(np.column_stack(components), axis=1)

        # Visium 六个固定偏移按行、列坐标找邻点，不含自身。没有有效邻点时留空。
        array_rows = position_table["array_row"].to_numpy()
        array_cols = position_table["array_col"].to_numpy()
        coordinates = pd.MultiIndex.from_arrays([array_rows, array_cols])
        lookup = pd.Series(receiver, index=coordinates)
        neighbors = []
        for row_offset, col_offset in [(0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor_coordinates = pd.MultiIndex.from_arrays([array_rows + row_offset, array_cols + col_offset])
            neighbors.append(lookup.reindex(neighbor_coordinates).to_numpy())
        neighbor_matrix = np.column_stack(neighbors)
        present = np.isfinite(neighbor_matrix).sum(axis=1)
        neighbor_receiver = np.full(len(position_table), np.nan)
        np.divide(np.nansum(neighbor_matrix, axis=1), present, out=neighbor_receiver, where=present > 0)

        # 秩回归控制 log1p 文库大小与髓系评分。本队列沿用未标准化的协变量秩。
        controls = np.column_stack([np.log1p(library_sizes), scores["myeloid"]])
        ac_score = scores["ac"]
        ac_residual = spatial.rank_residual(ac_score, controls, standardize_controls=False)
        for ligand in spatial.ligands:
            values = expression_table[ligand].to_numpy()
            residual = spatial.rank_residual(values, controls, standardize_controls=False)
            positive = values > 0
            high = positive & (values >= np.median(values[positive])) if positive.any() else positive
            zero = values == 0
            delta = np.nan
            if min(high.sum(), zero.sum()) >= 10:
                delta = np.nanmean(ac_score[high]) - np.nanmean(ac_score[zero])
            ligand_rows.append({
                "dataset": "GSE253080", "patient": row.patient, "sample": row.sample, "ligand": ligand,
                "n_spots": len(values), "detection_fraction": np.mean(positive),
                "rho_ac_score": spatial.spot_spearman(values, ac_score),
                "rho_ac_score_adjusted": spatial.spot_spearman(residual, ac_residual),
                "high_minus_zero_ac_score": delta,
            })

        # AC 评分上四分位内，比较 TGFB2 阳性上半部与零表达点的邻点均值；每组至少 10 个点。
        tgfb2 = expression_table["TGFB2"].to_numpy()
        ac_high = ac_score >= np.nanquantile(ac_score, 0.75)
        positive = ac_high & (tgfb2 > 0) & np.isfinite(neighbor_receiver)
        high = np.zeros(len(tgfb2), dtype=bool)
        if positive.sum() >= 10:
            high = positive & (tgfb2 >= np.nanmedian(tgfb2[positive]))
        zero = ac_high & (tgfb2 == 0) & np.isfinite(neighbor_receiver)
        delta = np.nan
        if min(high.sum(), zero.sum()) >= 10:
            delta = np.nanmean(neighbor_receiver[high]) - np.nanmean(neighbor_receiver[zero])
        section_rows.append({
            "dataset": "GSE253080", "patient": row.patient, "sample": row.sample, "n_spots": len(tgfb2),
            "n_tgfb2_high_ac": int(high.sum()), "n_tgfb2_zero_ac": int(zero.sum()),
            "tgfb2_neighbor_receiver_delta": delta,
        })
        plot_tables[row.sample] = pd.DataFrame({
            "sample": row.sample, "pixel_col": position_table["pixel_col"], "pixel_row": position_table["pixel_row"],
            "tgfb2": tgfb2, "ac_score": ac_score, "receiver_score": receiver,
        })

    # 同一患者的多张切片先取中位数；每种指标在配体间做 BH 校正，邻近检验单独汇总。
    section_ligand_table_total = pd.DataFrame(ligand_rows)
    section_table_total = pd.DataFrame(section_rows)
    metrics = ["rho_ac_score", "rho_ac_score_adjusted", "high_minus_zero_ac_score"]
    patient_ligand_table = section_ligand_table_total.groupby(["patient", "ligand"], as_index=False)[metrics].median()
    patient_table = section_table_total.groupby("patient", as_index=False)[["tgfb2_neighbor_receiver_delta"]].median()
    summary_rows = []
    for ligand in spatial.ligands:
        group_table = patient_ligand_table.loc[patient_ligand_table["ligand"] == ligand]
        record = {"ligand": ligand}
        for metric in metrics:
            measures = spatial.patient_metrics(group_table[metric])
            for name, value in measures.items():
                suffix = "n_patients" if name == "n" else name
                record[f"{metric}_{suffix}"] = value
        summary_rows.append(record)
    summary_table = pd.DataFrame(summary_rows)
    for metric in metrics:
        summary_table[f"{metric}_signflip_q"] = spatial.bh_adjust(summary_table[f"{metric}_signflip_p"])

    # 有限调整相关按大小取中间两个位置；奇数时二者重合。遇到同为中央值的切片，取 pd.csv 中最先出现者。
    tgfb2_table = section_ligand_table_total.loc[section_ligand_table_total["ligand"] == "TGFB2"]
    correlations = tgfb2_table.set_index("sample")["rho_ac_score_adjusted"]
    finite_correlations = correlations[np.isfinite(correlations)]
    if finite_correlations.empty:
        raise ValueError("没有可估计的切片 TGFB2 调整相关，无法选择代表切片")
    ordered = np.sort(finite_correlations.to_numpy())
    n_sections = len(ordered)
    middle_values = [ordered[(n_sections - 1) // 2], ordered[n_sections // 2]]
    sample = finite_correlations.index[finite_correlations.isin(middle_values)][0]
    representative_table = plot_tables[sample]
    spatial.write_spatial_tables(
        args.output_dir, section_ligand_table_total, patient_ligand_table, section_table_total, patient_table,
        summary_table, representative_table
    )

    # 汇总保持原有每指标的患者数字段。公开队列的三人措辞只在患者数确为三时沿用。
    proximal = spatial.patient_metrics(patient_table["tgfb2_neighbor_receiver_delta"])
    proximity = {}
    for name, value in proximal.items():
        suffix = "n_patients" if name == "n" else name
        proximity[f"tgfb2_neighbor_receiver_delta_{suffix}"] = value if pd.notna(value) else None
    tgfb2_result = summary_table.loc[summary_table["ligand"] == "TGFB2"].iloc[0]
    patient_count = len(patient_table)
    patient_text = "three" if patient_count == 3 else str(patient_count)
    summary = {
        "dataset": "GSE253080", "n_sections": len(section_table_total), "n_patients": patient_count,
        "tgfb2": tgfb2_result.astype(object).where(tgfb2_result.notna(), None).to_dict(), "proximity": proximity,
        "inference_limit": f"{patient_text} patients; directional sensitivity analysis only",
    }
    files.write_json(summary, args.output_dir / "07_分析摘要.json")

    # 患者散点按原先面板顺序使用同一随机流，抖动只用于展示。
    rng = np.random.default_rng(20260716)
    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    axis = axes[0, 0]
    values = patient_ligand_table.loc[patient_ligand_table["ligand"] == "TGFB2", "rho_ac_score_adjusted"].to_numpy()
    axis.scatter(rng.normal(1, 0.035, len(values)), values, color="#267D83", s=55)
    axis.hlines(np.median(values), 0.82, 1.18, color="#B33A3A", linewidth=2.5)
    axis.set(xlim=(0.65, 1.35), ylabel="Adjusted within-section Spearman rho")
    axis.set_xticks([1], [f"{np.isfinite(values).sum()} patient medians"])
    axis.set_title("a  TGFB2 vs AC-like score", loc="left", fontweight="bold")

    # 固定面板展示全部配体；邻近面板只绘制能得到有效差值的患者。
    axis = axes[0, 1]
    for i, ligand in enumerate(spatial.ligands):
        values = patient_ligand_table.loc[patient_ligand_table["ligand"] == ligand, "rho_ac_score_adjusted"].dropna()
        axis.scatter(rng.normal(i, 0.045, len(values)), values, color="#6C7A89", alpha=0.75, s=30)
        color = "#B33A3A" if ligand == "TGFB2" else "#222222"
        axis.hlines(values.median(), i - 0.20, i + 0.20, color=color, linewidth=2)
    axis.set_xticks(range(len(spatial.ligands)), spatial.ligands, rotation=45, ha="right")
    axis.set_ylabel("Adjusted within-section Spearman rho")
    axis.set_title("b  Fixed ligand panel", loc="left", fontweight="bold")

    axis = axes[0, 2]
    values = patient_table["tgfb2_neighbor_receiver_delta"].dropna()
    axis.scatter(rng.normal(1, 0.035, len(values)), values, color="#6A4C93", s=55)
    if len(values):
        axis.hlines(values.median(), 0.82, 1.18, color="#B33A3A", linewidth=2.5)
    axis.set(xlim=(0.65, 1.35), ylabel="Neighbor receiver score difference")
    axis.set_xticks([1], ["patient medians"])
    axis.set_title("c  Descriptive proximity", loc="left", fontweight="bold")
    for axis in axes[0]:
        axis.axhline(0, color="#777777", linestyle="--", linewidth=1)
        axis.spines[["top", "right"]].set_visible(False)

    # 三个空间面板共用代表切片的原像素坐标及向下的像素行方向。
    maps = [
        ("ac_score", "d  AC-like score", "viridis"), ("tgfb2", "e  TGFB2 expression", "magma"),
        ("receiver_score", "f  Myeloid/TGFBR score", "cividis"),
    ]
    for axis, (column, title, palette) in zip(axes[1], maps):
        points = axis.scatter(
            representative_table["pixel_col"], -representative_table["pixel_row"], c=representative_table[column],
            s=4, cmap=palette, linewidth=0
        )
        axis.set_aspect("equal")
        axis.axis("off")
        axis.set_title(title, loc="left", fontweight="bold")
        figure.colorbar(points, ax=axis, fraction=0.046, pad=0.02)
    title = f"GSE253080 spatial sensitivity analysis; representative {sample}"
    figure.suptitle(title, x=0.06, ha="left", fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    files.save_figure(figure, args.output_dir / "08_FigureS7_GSE253080空间信号.pdf")
    plt.close(figure)
    print(f"切片 {len(section_table_total)} 张，患者 {patient_count} 名")


if __name__ == "__main__":
    main()
