#!/usr/bin/env python3
# 目的：在 GBM-Space 切片内估计 AC 样恶性状态与六种配体的相关，再以患者为单位检验。
# 输入：--input-dir 下每张切片一个 h5ad，读取 X、var.feature_types、obs.sample_name/in_tissue 和 obsm.spatial。
# 使用 Python 保留 AnnData 的磁盘读取和原 NumPy/SciPy 计算。公开 section_spatial_metrics.csv 给出完整切片数。
# 输出：默认在 external_validation/06_GBMSpace空间信号 写 01_切片配体指标.csv、02_患者配体指标.csv、
# 03_切片空间指标.csv、04_患者空间指标.csv、05_配体汇总.csv、06_代表切片点位.csv、07_分析摘要.json、
# 08_Figure5_GBMSpace空间信号.pdf 和 08_Figure5_GBMSpace空间信号.png。
# 运行方式：python 06_GBMSpace空间信号.py --input-dir 原始h5ad目录 [--output-dir 复算目录]

import argparse
from contextlib import closing
from importlib import import_module
from itertools import product
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

spatial = import_module("00a_空间分析")
files = import_module("00a_项目路径")


def main():
    """读取切片，写患者、切片和代表图数据。输入缺失或产物核对不符时停止，无返回值。"""
    external_dir = Path(__file__).resolve().parents[1] / "external_validation"
    parser = argparse.ArgumentParser(description="GBM-Space 患者级空间验证")
    parser.add_argument("--input-dir", type=Path, default=external_dir / "raw/gbmspace/extracted/anndata")
    parser.add_argument("--output-dir", type=Path, default=external_dir / Path(__file__).stem)
    args = parser.parse_args()

    # 完整队列的规模从冻结表读取。原文件和公开表使用不同编码，此处只核切片数。
    baseline_table = pd.read_csv(external_dir / "results/gbmspace/section_spatial_metrics.csv", usecols=["sample"])
    paths = sorted(args.input_dir.glob("*.h5ad"))
    if not paths or len(paths) != baseline_table["sample"].nunique():
        raise ValueError(f"找到 {len(paths)} 个 h5ad，公开基线含 {baseline_table['sample'].nunique()} 张切片")

    # 每个文件只取目标基因和状态矩阵，全基因表达用于计算各点位文库大小。
    target_genes = spatial.ligands + spatial.receptors + spatial.ac_markers
    ligand_rows = []
    section_rows = []
    plot_tables = {}
    for path in paths:
        section = ad.read_h5ad(path, backed="r")
        with closing(section.file):
            sample_names = section.obs["sample_name"].drop_duplicates()
            if len(sample_names) != 1 or sample_names.isna().any():
                raise ValueError(f"{path.name} 需要唯一的切片名称")
            sample = str(sample_names.iloc[0])
            patient = sample.split("-")[0]
            names = np.asarray(section.var_names, dtype=str)
            types = section.var["feature_types"].astype(str).to_numpy()

            # AnnData 同时存基因和细胞丰度。按类型、名称双重匹配，要求每个目标基因恰有一列。
            target_mask = (types == "Gene Expression") & np.isin(names, target_genes)
            feature_ids = pd.Series(np.flatnonzero(target_mask), index=names[target_mask])
            if feature_ids.index.has_duplicates or set(feature_ids.index) != set(target_genes):
                raise ValueError(f"{sample} 缺少目标基因或目标表达特征重复")
            indices = feature_ids.loc[target_genes].to_numpy()
            selected = section[:, indices].X
            counts = selected.toarray() if sparse.issparse(selected) else np.asarray(selected)
            totals = np.asarray(section[:, types == "Gene Expression"].X.sum(axis=1)).ravel()

            # 空间坐标与组织掩码均沿同一 obs 行顺序读取，关闭文件后继续计算。
            state_indices = np.flatnonzero(types == "Cell state abundances")
            selected = section[:, state_indices].X
            states = selected.toarray() if sparse.issparse(selected) else np.asarray(selected)
            states = states.astype(float)
            coordinates = np.asarray(section.obsm["spatial"], dtype=float)
            in_tissue = section.obs["in_tissue"].astype(float).to_numpy() > 0

        # 作者标签中，广义 AC 额外计入增殖性 AC-OPC。每个点位用恶性总丰度作 AC 分母。
        state_names = pd.Index(names[state_indices])
        ac_prefixes = ("AC progenitor-like", "AC-gliosis-like")
        broad_prefixes = ac_prefixes + ("Proliferative AC-OPC-like",)
        malignant_prefixes = broad_prefixes + (
            "Hypoxic ", "NPC-neuronal-like", "OPC-NPC-like", "OPC-like ", "OPC-neuronal-like",
            "Proliferative NPC-OPC-like", "Proliferative nIPC-like", "Undefined ",
        )
        ac_mask = state_names.str.startswith(ac_prefixes)
        malignant_mask = state_names.str.startswith(malignant_prefixes)
        myeloid_mask = state_names.str.contains("TAMs", regex=False)
        myeloid_mask |= state_names.isin(["Monocytes", "Dendritic cells"])
        if not ac_mask.any() or not malignant_mask.any() or not myeloid_mask.any():
            raise ValueError(f"{sample} 缺少 AC、恶性或髓系状态")

        # 髓系和正常星形胶质状态的分母为全部状态，微小下限沿用原比例计算。
        astro_mask = state_names.str.lower().str.contains("astrocytes", regex=False)
        astro_mask |= state_names.isin(["Gliosis-like", "Neural-support astrocytes"])
        total_abundance = np.maximum(states.sum(axis=1), 1e-12)
        malignant = states[:, malignant_mask].sum(axis=1)
        malignant_fraction = malignant / total_abundance
        ac_fraction = states[:, ac_mask].sum(axis=1) / np.maximum(malignant, 1e-12)
        broad_abundance = states[:, state_names.str.startswith(broad_prefixes)].sum(axis=1)
        broad_fraction = broad_abundance / np.maximum(malignant, 1e-12)
        myeloid_fraction = states[:, myeloid_mask].sum(axis=1) / total_abundance
        astro_fraction = states[:, astro_mask].sum(axis=1) / total_abundance

        # 全基因总数归一到一万后取 log1p。受体分数为两个受体的均值乘髓系比例。
        normalized = np.log1p(counts.astype(float) / np.maximum(totals[:, None], 1.0) * 10_000)
        expression_table = pd.DataFrame(normalized, columns=target_genes)
        marker_score = expression_table[spatial.ac_markers].mean(axis=1).to_numpy()
        receiver = myeloid_fraction * expression_table[spatial.receptors].mean(axis=1).to_numpy()
        tree = cKDTree(coordinates)
        distances, _ = tree.query(coordinates, k=2)
        nearest = distances[:, 1]
        spacing = float(np.median(nearest[np.isfinite(nearest) & (nearest > 0)]))

        # 半径为切片中位最近邻距离的 1.25 倍。排除自身，无有效邻点时留空。
        neighbor_receiver = np.full(len(receiver), np.nan)
        for i, neighbors in enumerate(tree.query_ball_point(coordinates, r=1.25 * spacing)):
            neighbors = np.asarray(neighbors, dtype=int)
            neighbors = neighbors[(neighbors != i) & np.isfinite(receiver[neighbors])]
            if len(neighbors):
                neighbor_receiver[i] = np.mean(receiver[neighbors])

        # 入选点须在组织内、有表达和恶性丰度，且恶性比例至少 20%。秩回归先用整张切片拟合。
        malignant_threshold = 0.20
        valid = in_tissue & (totals > 0) & (malignant > 0) & (malignant_fraction >= malignant_threshold)
        controls = np.column_stack([astro_fraction, malignant_fraction, np.log1p(totals)])
        ac_residual = spatial.rank_residual(ac_fraction, controls, standardize_controls=True)
        broad_residual = spatial.rank_residual(broad_fraction, controls, standardize_controls=True)

        # 各配体分别在有效点内按狭义或广义 AC 的四分位分组。边界保留并列，两组各需 20 点。
        contrasts = {}
        groupings = {"ac": ac_fraction, "broad": broad_fraction}
        for ligand, grouping_name in product(spatial.ligands, groupings):
            values = expression_table[ligand].to_numpy()
            grouping = groupings[grouping_name]
            eligible = valid & np.isfinite(values) & np.isfinite(grouping)
            contrasts[ligand, grouping_name] = (np.nan, 0, 0)
            if eligible.sum() < 40:
                continue
            lower, upper = np.quantile(grouping[eligible], [0.25, 0.75])
            low = eligible & (grouping <= lower)
            high = eligible & (grouping >= upper)
            delta = np.mean(values[high]) - np.mean(values[low]) if min(high.sum(), low.sum()) >= 20 else np.nan
            contrasts[ligand, grouping_name] = (delta, int(high.sum()), int(low.sum()))

        # 一行代表一个切片与一个配体；相关只在入选点上计算，统计单元随后汇总到患者。
        for ligand in spatial.ligands:
            values = expression_table[ligand].to_numpy()
            residual = spatial.rank_residual(values, controls, standardize_controls=True)
            delta, n_high, n_low = contrasts[ligand, "ac"]
            ligand_rows.append({
                "dataset": "GBM-Space", "patient": patient, "sample": sample, "ligand": ligand,
                "n_spots": len(values), "n_eligible_spots": int(valid.sum()),
                "detection_fraction": float(np.mean(values[valid] > 0)) if valid.any() else np.nan,
                "rho_ac_fraction": spatial.spot_spearman(values[valid], ac_fraction[valid]),
                "rho_ac_fraction_adjusted": spatial.spot_spearman(residual[valid], ac_residual[valid]),
                "rho_ac_broad_fraction_adjusted": spatial.spot_spearman(residual[valid], broad_residual[valid]),
                "rho_ac_marker_score": spatial.spot_spearman(values[valid], marker_score[valid]),
                "mean_top_minus_bottom_ac_delta": delta,
                "mean_top_minus_bottom_ac_broad_delta": contrasts[ligand, "broad"][0],
                "n_top_ac": n_high, "n_bottom_ac": n_low,
            })

        # 原方案先要求至少 80 个候选点，再取 AC 比例上半部。优先比较阳性表达的上半部与零表达。
        tgfb2 = expression_table["TGFB2"].to_numpy()
        ac_rich = valid & np.isfinite(tgfb2) & np.isfinite(ac_fraction)
        high = np.zeros(len(tgfb2), dtype=bool)
        low = high.copy()
        if ac_rich.sum() >= 80:
            ac_rich &= ac_fraction >= np.median(ac_fraction[ac_rich])
            positive = ac_rich & (tgfb2 > 0)
            zero = ac_rich & (tgfb2 == 0)
            if min(positive.sum(), zero.sum()) >= 20:
                high = positive & (tgfb2 >= np.median(tgfb2[positive]))
                low = zero
            else:
                lower, upper = np.quantile(tgfb2[ac_rich], [0.25, 0.75])
                high = ac_rich & (tgfb2 >= upper) & (tgfb2 > lower)
                low = ac_rich & (tgfb2 <= lower)
            high &= np.isfinite(neighbor_receiver)
            low &= np.isfinite(neighbor_receiver)

        # 两组各足 20 点才给中位数差；同点与邻点使用完全相同的点位组。
        enough = min(high.sum(), low.sum()) >= 20
        neighbor_delta = np.median(neighbor_receiver[high]) - np.median(neighbor_receiver[low]) if enough else np.nan
        same_spot_delta = np.median(receiver[high]) - np.median(receiver[low]) if enough else np.nan
        section_rows.append({
            "dataset": "GBM-Space", "patient": patient, "sample": sample, "n_spots": len(valid),
            "n_eligible_spots": int(valid.sum()), "n_cell_states": len(state_names), "n_ac_states": int(ac_mask.sum()),
            "n_malignant_states": int(malignant_mask.sum()), "n_myeloid_states": int(myeloid_mask.sum()),
            "median_malignant_fraction": np.median(malignant_fraction[valid]) if valid.any() else np.nan,
            "median_ac_fraction": np.median(ac_fraction[valid]) if valid.any() else np.nan,
            "median_myeloid_fraction": np.median(myeloid_fraction[valid]) if valid.any() else np.nan,
            "neighbor_spacing": spacing,
            "tgfb2_neighbor_receiver_delta": neighbor_delta, "tgfb2_same_spot_receiver_delta": same_spot_delta,
            "n_tgfb2_high_ac_rich": int(high.sum()), "n_tgfb2_low_ac_rich": int(low.sum()),
        })
        plot_tables[sample] = pd.DataFrame({
            "patient": patient, "sample": sample, "x": coordinates[:, 0], "y": coordinates[:, 1], "valid": valid,
            "tgfb2": tgfb2, "ac_fraction": ac_fraction, "normal_astro_fraction": astro_fraction,
            "receiver": receiver, "neighbor_receiver": neighbor_receiver,
        })

    # 一名患者的同一配体先取各切片中位数；各指标单独在六种配体之间校正。
    section_ligand_table_total = pd.DataFrame(ligand_rows)
    section_table_total = pd.DataFrame(section_rows)
    metrics = [
        "rho_ac_fraction", "rho_ac_fraction_adjusted", "rho_ac_broad_fraction_adjusted", "rho_ac_marker_score",
        "mean_top_minus_bottom_ac_delta", "mean_top_minus_bottom_ac_broad_delta",
    ]
    grouped = section_ligand_table_total.groupby(["patient", "ligand"], as_index=False)
    patient_ligand_table = grouped[metrics].median().sort_values(["ligand", "patient"])
    rng = np.random.default_rng(20260716)
    summary_rows = []
    for ligand, group_table in patient_ligand_table.groupby("ligand"):
        record = {"ligand": ligand, "n_patients": group_table["patient"].nunique()}
        for metric in metrics:
            measures = spatial.patient_metrics(group_table[metric], rng)
            for name in ["median", "mean", "n_positive", "signflip_p", "wilcoxon_p"]:
                record[f"{metric}_{name}"] = measures[name]
        summary_rows.append(record)
    summary_table = pd.DataFrame(summary_rows)
    for metric in metrics:
        summary_table[f"{metric}_signflip_q"] = spatial.bh_adjust(summary_table[f"{metric}_signflip_p"])

    # 邻近分数和同点分数也按患者取中位数。代表切片沿用旧排序的并列处理，缺少有效相关时取首个样本名。
    proximity_columns = ["tgfb2_neighbor_receiver_delta", "tgfb2_same_spot_receiver_delta"]
    patient_table = section_table_total.groupby("patient", as_index=False)[proximity_columns].median()
    proximity = {"n_patients": patient_table["patient"].nunique()}
    for metric in proximity_columns:
        proximity[metric] = spatial.patient_metrics(patient_table[metric], rng)
    tgfb2_table = section_ligand_table_total.loc[section_ligand_table_total["ligand"] == "TGFB2"]
    adjusted = tgfb2_table.set_index("sample")["rho_ac_fraction_adjusted"].dropna()
    distances = (adjusted - adjusted.median()).abs()
    sample = distances.sort_values().index[0] if len(distances) else sorted(plot_tables)[0]
    representative_table = plot_tables[sample]

    # 逐份回读 CSV；摘要中的总点位、有效点位与患者数都取自切片表。
    spatial.write_spatial_tables(
        args.output_dir, section_ligand_table_total, patient_ligand_table, section_table_total, patient_table,
        summary_table, representative_table
    )
    tgfb2_result = summary_table.set_index("ligand").loc["TGFB2"]
    summary = {
        "dataset": "GBM-Space", "n_sections": len(section_table_total), "n_patients": len(patient_table),
        "n_spots": int(section_table_total["n_spots"].sum()),
        "n_eligible_spots": int(section_table_total["n_eligible_spots"].sum()),
        "malignant_fraction_threshold": malignant_threshold,
        "tgfb2": {key: float(value) if pd.notna(value) else None for key, value in tgfb2_result.items()},
        "proximity": proximity,
    }
    files.write_json(summary, args.output_dir / "07_分析摘要.json")

    # 前两幅按原顺序消费同一随机序列，仅对展示位置加横向抖动。
    figure = plt.figure(figsize=(13.8, 8.2))
    grid = figure.add_gridspec(2, 4, height_ratios=[1, 1.15], wspace=0.75, hspace=0.50)
    axis = figure.add_subplot(grid[0, 0])
    tgfb2_patient_table = patient_ligand_table.loc[patient_ligand_table["ligand"] == "TGFB2"]
    unadjusted = tgfb2_patient_table["rho_ac_fraction"].to_numpy()
    adjusted = tgfb2_patient_table["rho_ac_fraction_adjusted"].to_numpy()
    for first, second in zip(unadjusted, adjusted):
        if np.isfinite(first) and np.isfinite(second):
            axis.plot([0, 1], [first, second], color="#B7C3C8", lw=0.8)
    for position, values, color in [(0, unadjusted, "#167D8D"), (1, adjusted, "#6C7A89")]:
        axis.scatter(rng.normal(position, 0.035, len(values)), values, s=30, color=color, alpha=0.85, zorder=3)
        axis.hlines(np.nanmedian(values), position - 0.18, position + 0.18, color="#C0392B", lw=2.5)
    axis.set(xlim=(-0.35, 1.35), ylabel="Patient median within-section rho")
    axis.set_xticks([0, 1], ["Raw", "Adjusted"])
    axis.set_title("a  TGFB2-AC association", loc="left", fontsize=10, fontweight="bold")

    # 第二幅显示每个配体的患者分布；每种配体单独取中位数。
    panel_axis = figure.add_subplot(grid[0, 1])
    pivot_table = patient_ligand_table.pivot(index="patient", columns="ligand", values="rho_ac_fraction_adjusted")
    for i, ligand in enumerate(spatial.ligands):
        values = pivot_table[ligand].dropna().to_numpy()
        panel_axis.scatter(i + rng.normal(0, 0.06, len(values)), values, s=18, color="#6C7A89", alpha=0.65)
        if len(values):
            color = "#C0392B" if ligand == "TGFB2" else "#222222"
            panel_axis.hlines(np.median(values), i - 0.22, i + 0.22, color=color, lw=2.5)
    panel_axis.set_xticks(range(len(spatial.ligands)), spatial.ligands, rotation=45, ha="right")
    panel_axis.set_ylabel("Patient median adjusted rho")
    panel_axis.set_title("b  Fixed ligand panel", loc="left", fontsize=10, fontweight="bold")

    # 两个差值面板分别展示 AC 高低组、以及 TGFB2 高低组周边受体信号。
    strip_panels = [
        (tgfb2_patient_table["mean_top_minus_bottom_ac_delta"], "c  AC-rich minus AC-poor", "TGFB2 log-normalized Δ"),
        (patient_table["tgfb2_neighbor_receiver_delta"], "d  Myeloid/TGFBR proximity", "High-minus-low TGFB2 delta"),
    ]
    for i, (series, title, label) in enumerate(strip_panels, start=2):
        strip = figure.add_subplot(grid[0, i])
        values = series.dropna().to_numpy()
        strip.scatter(rng.normal(0, 0.035, len(values)), values, s=34, color="#167D8D", alpha=0.85, zorder=3)
        if len(values):
            strip.hlines(np.median(values), -0.18, 0.18, color="#C0392B", lw=3, zorder=4)
        strip.set(xlim=(-0.35, 0.35), xticks=[], ylabel=label)
        strip.set_title(title, loc="left", fontsize=10, fontweight="bold")
        strip.spines["bottom"].set_visible(False)
    for panel in figure.axes:
        panel.axhline(0, color="#777777", lw=1, ls="--")
        panel.spines[["top", "right"]].set_visible(False)

    # 三张图共用代表切片的入选点位与方向，各自以原量纲着色。
    map_table = representative_table.loc[representative_table["valid"]]
    maps = [
        ("ac_fraction", "e  Malignant AC fraction", "viridis"), ("tgfb2", "f  TGFB2 expression", "magma"),
        ("receiver", "g  Myeloid x TGFBR score", "cividis"),
    ]
    for i, (column, title, palette) in enumerate(maps):
        panel = figure.add_subplot(grid[1, i])
        points = panel.scatter(map_table["x"], -map_table["y"], c=map_table[column], s=4, cmap=palette, linewidths=0)
        panel.set(aspect="equal", xticks=[], yticks=[])
        panel.set_title(title, loc="left", fontsize=10, fontweight="bold")
        panel.spines[["top", "right", "bottom", "left"]].set_visible(False)
        figure.colorbar(points, ax=panel, fraction=0.046, pad=0.02)

    # 图注按所选切片和实际患者数生成。PDF 绘制一次，由同页生成 600 dpi PNG。
    note = figure.add_subplot(grid[1, 3])
    note.axis("off")
    selection = "Closest to the median adjusted rho." if len(distances) else "First section by name."
    note_text = f"Representative section\n{sample}\n\n{selection}\n\nPatient medians: {len(patient_table)}."
    note.text(0, 1, note_text, va="top", fontsize=10, linespacing=1.45)
    figure.suptitle("GBM-Space patient-level spatial validation of AC-like TGFB2", x=0.02, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    files.save_figure(figure, args.output_dir / "08_Figure5_GBMSpace空间信号.pdf")
    plt.close(figure)
    print(f"切片 {len(section_table_total)} 张，患者 {len(patient_table)} 名，有效点位 {summary['n_eligible_spots']} 个")


if __name__ == "__main__":
    main()
