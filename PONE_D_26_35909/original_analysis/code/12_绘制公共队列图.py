#!/usr/bin/env python3
# 目的：用已冻结的患者级结果重绘公共队列主图与补图，沿用早期分析的面板、Arial 字体和紧裁版式。
# 输入：original_analysis/data 下的 cohort_manifest.csv（cohort、patients、regions、included_cells），
# ac_vs_other_ligand_contrasts.csv（patient_values、n_patients）、meta_ligand_results.csv（meta_z、q_value、loo_q_value），
# patient_ac_fraction.csv、regional_ac_fraction_tests.csv、regional_communication_tests.csv（区域配对差值及检验），
# patient_state_expression.csv（状态细胞数和配体均值），uniform_annotation_patient_contrasts.csv、
# uniform_annotation_cohort_results.csv 和 uniform_annotation_summary.json（统一注释、谱系校正）。
# 输出：投稿包 figures/ 下 Figure1 至 Figure4、SupplementaryFigureS1 至 SupplementaryFigureS3，各有 .pdf 和 .png。
# 运行方式：从 original_analysis 运行 MPLBACKEND=Agg python code/12_绘制公共队列图.py。Python 沿用原 Matplotlib 图形对象。
# 输出目录可用 --output 单独指定，便于与同名的投稿图比较。PDF 转换依赖 pdftocairo、pypdf 和 Pillow。
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from importlib import import_module
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

# 两位编号不能写在 Python 的 from 语句中。样式位于两套分析的共同上级，路径模块位于本脚本旁。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
style = import_module("00a_论文图形样式")
project_paths = import_module("00a_项目路径")


def main(output: Path | None = None) -> None:
    """绘制原分析的四张主图和三张补图；缺少冻结表或保存核对失败时停止。无返回值。"""
    package_dir = project_paths.submission_package(project_paths.project_root())
    data_dir = package_dir / "data"
    style.require_cohort_inputs(data_dir, supplementary=True)
    if output is None:
        output = package_dir / "figures"

    # 这一版使用紧裁 PDF、原配色和单独的面板字母；刻度尺寸与投稿版不同。
    save_public = partial(style.save_figure, directory=output)
    open_axes = partial(style.open_axes, tick_width=0.7, tick_length=3)
    sns.set_theme(style="white")
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    # Figure1a、b 从队列清单取地区覆盖、患者数和纳入细胞数，按原分析的顺序排版。
    manifest_table = pd.read_csv(data_dir / "cohort_manifest.csv")
    manifest_table = manifest_table.set_index("cohort", verify_integrity=True).loc[style.cohort_order].reset_index()
    fig = plt.figure(figsize=(7.2, 5.2))
    grid_spec = GridSpec(
        2, 2, figure=fig, height_ratios=[1.05, 1.0], width_ratios=[1.25, 0.9], hspace=0.55, wspace=0.45
    )

    # a：公共队列与区域覆盖矩阵。
    ax = fig.add_subplot(grid_spec[0, 0])
    regions = ["Tumor/Core", "Edge", "Peritumoral/Peripheral"]
    y = np.arange(len(manifest_table))[::-1]
    cohort_colors = {
        "GSE211376": "#2B6F77",
        "GSE273274": "#D96C4F",
        "GSE274987": "#C49A2C",
        "GSE276177": "#6B5B95",
        "GSE286419": "#4C78A8",
    }
    for yi, row in zip(y, manifest_table.itertuples()):
        row_regions = set(str(row.regions).split("; "))
        present = [
            bool(row_regions & {"Tumor", "Core"}),
            "Edge" in row_regions,
            bool(row_regions & {"Peritumoral", "Peripheral"}),
        ]
        xpos = np.flatnonzero(present)
        if len(xpos) > 1:
            ax.plot(xpos, np.repeat(yi, len(xpos)), color="#B8B8B8", lw=1.2, zorder=1)
        for xi, available in enumerate(present):
            size = 38 + row.patients * 4 if available else 18
            color = cohort_colors[row.cohort] if available else "white"
            ax.scatter(xi, yi, s=size, facecolor=color, edgecolor="#A8A8A8", linewidth=0.7, zorder=2)
        ax.text(2.42, yi, f"n={row.patients}", va="center", fontsize=7, color="#444444")

    # 横坐标对应三个取材区类别；右侧标出真实纳入患者数。
    ax.set_xticks(range(len(regions)), regions)
    ax.set_yticks(y, manifest_table["cohort"])
    ax.set_xlim(-0.45, 3.05)
    ax.set_ylim(-0.6, len(y) - 0.4)
    ax.set_title("Public cohorts and spatial sampling", loc="left", pad=8)
    ax.set_xlabel("Tissue region")
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    style.public_panel_label(ax, "a")

    # b：真实纳入细胞与患者数量，不用柱状图。
    ax = fig.add_subplot(grid_spec[0, 1])
    for yi, row in zip(y, manifest_table.itertuples()):
        ax.scatter(
            row.included_cells,
            yi,
            s         = 26 + row.patients * 7,
            color     = cohort_colors[row.cohort],
            edgecolor = "white",
            linewidth = 0.7,
        )
        ax.text(row.included_cells * 1.05, yi, f"{row.included_cells:,}", va="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_yticks(y, manifest_table["cohort"])
    ax.set_xlabel("Included cells/nuclei (log scale)")
    ax.set_title("Analysis set", loc="left", pad=8)
    ax.grid(axis="x", color="#EAEAEA", lw=0.6)
    open_axes(ax)
    style.public_panel_label(ax, "b")

    # c：两个不同证据层级的统计对象。
    ax = fig.add_subplot(grid_spec[1, :])
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # 示意图分两行表示患者内状态比较和区域配对；箭头连接计算步骤。
    state_colors = {"AC-like": "#C84B31", "Other malignant": "#3D6F8E", "Myeloid": "#4D8061"}
    ax.scatter(
        [0.8, 0.8],
        [2.9, 2.0],
        s         = [760, 760],
        color     = [state_colors["AC-like"], state_colors["Other malignant"]],
        edgecolor = "white",
        linewidth = 1,
    )
    ax.text(0.8, 2.9, "AC-like", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
    ax.text(0.8, 2.0, "Other\nstates", ha="center", va="center", color="white", fontsize=6.5, fontweight="bold")
    ax.annotate("", xy=(2.25, 2.45), xytext=(1.45, 2.45), arrowprops=dict(arrowstyle="->", color="#555555", lw=1))
    ax.text(2.75, 2.45, "Patient-level\nexpression difference", ha="center", va="center", fontsize=7.5)
    ax.annotate("", xy=(4.1, 2.45), xytext=(3.55, 2.45), arrowprops=dict(arrowstyle="->", color="#555555", lw=1))
    ax.text(4.85, 2.45, "Cohort-specific\neffect", ha="center", va="center", fontsize=7.5)

    # 第一行从患者内表达差走到队列合并，研究对象仍是患者。
    ax.annotate("", xy=(6.25, 2.45), xytext=(5.65, 2.45), arrowprops=dict(arrowstyle="->", color="#555555", lw=1))
    ax.scatter(7.0, 2.45, s=470, color="#F1C75B", edgecolor="#806000", linewidth=0.8)
    ax.text(7.0, 2.45, "Cross-cohort\nmeta-analysis", ha="center", va="center", fontsize=7)

    # 通讯分支。
    ax.scatter(0.8, 0.72, s=760, color=state_colors["Myeloid"], edgecolor="white", linewidth=1)
    ax.text(0.8, 0.72, "Myeloid", ha="center", va="center", color="white", fontsize=6.8, fontweight="bold")
    ax.annotate("", xy=(2.25, 0.72), xytext=(1.45, 0.72), arrowprops=dict(arrowstyle="->", color="#555555", lw=1))
    ax.text(2.85, 0.72, "Ligand-receptor\nco-availability", ha="center", va="center", fontsize=7.5)
    ax.annotate("", xy=(4.55, 0.72), xytext=(3.65, 0.72), arrowprops=dict(arrowstyle="->", color="#555555", lw=1))
    ax.text(5.35, 0.72, "Within-patient\nregional difference", ha="center", va="center", fontsize=7.5)

    # 第二行只描述表达共可用性，图中保留原推断边界。
    ax.annotate("", xy=(6.8, 0.72), xytext=(6.15, 0.72), arrowprops=dict(arrowstyle="->", color="#555555", lw=1))
    ax.scatter(7.55, 0.72, s=440, facecolor="white", edgecolor="#555555", linewidth=1.0)
    ax.text(7.55, 0.72, "Comparison\nacross regions", ha="center", va="center", fontsize=7)
    ax.plot([8.35, 8.35], [0.25, 3.25], ls="--", color="#888888", lw=0.8)
    ax.text(8.35, 3.38, "Inference boundary", ha="center", fontsize=7, color="#666666")
    ax.text(8.65, 2.38, "Expression contrast", ha="left", fontsize=7, color="#333333")
    ax.text(8.65, 0.62, "No causal signaling claim", ha="left", fontsize=7, color="#333333")
    ax.text(0.0, 3.75, "Patient-level estimands and evidence hierarchy", fontsize=9)
    style.public_panel_label(ax, "c")
    save_public(fig, stem="Figure1", panels="abc")

    # Figure2 的首行固定展示原推断队列。每一行配对数据代表一位患者，对比表由队列和基因唯一定位。
    contrast_table = pd.read_csv(data_dir / "ac_vs_other_ligand_contrasts.csv")
    contrast_index_table = contrast_table.set_index(["cohort", "gene"], verify_integrity=True)
    inferential = ["GSE211376", "GSE273274", "GSE274987", "GSE286419"]
    fig = plt.figure(figsize=(7.2, 6.1))
    grid_spec = GridSpec(
        2, 2, figure=fig, height_ratios=[1.0, 1.2], width_ratios=[1.45, 0.75], hspace=0.55, wspace=0.42
    )

    # a：每个公共队列的患者内 TGFB2 配对值。
    paired_grid = GridSpecFromSubplotSpec(1, len(inferential), subplot_spec=grid_spec[0, :], wspace=0.35)
    for i, cohort in enumerate(inferential):
        ax = fig.add_subplot(paired_grid[0, i])
        contrast_row = contrast_index_table.loc[(cohort, "TGFB2"), :]
        paired_table = style.paired_expression_table(contrast_row)
        for _, item in paired_table.iterrows():
            ax.plot([0, 1], [item["other_mean"], item["ac_mean"]], color="#B8B8B8", lw=0.8, zorder=1)
            ax.scatter(0, item["other_mean"], s=17, color=state_colors["Other malignant"], zorder=2)
            ax.scatter(1, item["ac_mean"], s=17, color=state_colors["AC-like"], zorder=2)

        # 粗线连接两组各自的患者中位数，每条灰线保留同一患者的配对关系。
        ax.plot(
            [0, 1],
            [paired_table["other_mean"].median(), paired_table["ac_mean"].median()],
            color  = "#222222",
            lw     = 2.2,
            zorder = 3,
        )
        ax.set_xticks([0, 1], ["Other", "AC-like"])
        ax.set_title(f"{cohort}\nn={len(paired_table)} patients", loc="center", fontsize=8)
        ax.set_xlim(-0.25, 1.25)
        ax.grid(axis="y", color="#EFEFEF", lw=0.5)
        open_axes(ax)
        if i == 0:
            ax.set_ylabel("Mean log-normalized TGFB2")
            style.public_panel_label(ax, "a")
        else:
            ax.set_ylabel("")

    # b：所有配体在各队列中的效应方向与强度。
    ax = fig.add_subplot(grid_spec[1, 0])
    gene_order = style.contrast_ligands
    cohort_order = style.cohort_order
    gene_positions = {gene: i for i, gene in enumerate(gene_order)}
    cohort_positions = {cohort: i for i, cohort in enumerate(cohort_order[::-1])}
    max_abs = max(abs(contrast_table["median_difference"].min()), abs(contrast_table["median_difference"].max()))
    norm = mpl.colors.Normalize(-max_abs, max_abs)
    for row in contrast_table.itertuples():
        x, y = gene_positions[row.gene], cohort_positions[row.cohort]
        color = mpl.colormaps["RdBu_r"](norm(row.median_difference))
        size = 26 + 7 * row.n_patients
        edge = "black" if pd.notna(row.q_value) and row.q_value < 0.05 else "white"
        ax.scatter(x, y, s=size, color=color, edgecolor=edge, linewidth=1.0 if edge == "black" else 0.6)

    # 点面积编码患者数，边线按未取位的原 q 值判定。
    ax.set_xticks(range(len(gene_order)), gene_order, rotation=35, ha="right")
    ax.set_yticks(range(len(cohort_order)), cohort_order[::-1])
    ax.set_xlabel("Ligand")
    ax.set_ylabel("Public cohort")
    ax.set_title("AC-like minus other malignant states", loc="left", pad=8)
    ax.set_xlim(-0.6, len(gene_order) - 0.4)
    ax.set_ylim(-0.6, len(cohort_order) - 0.4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    color_scale = mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    colorbar = fig.colorbar(color_scale, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Median expression difference", fontsize=7)
    style.public_panel_label(ax, "b")

    # c：合并证据及删一队列敏感性。
    ax = fig.add_subplot(grid_spec[1, 1])
    meta_table = pd.read_csv(data_dir / "meta_ligand_results.csv")
    ordered_meta_table = meta_table.set_index("gene").loc[gene_order].reset_index()
    y_positions = np.arange(len(ordered_meta_table))[::-1]
    for y_position, row in zip(y_positions, ordered_meta_table.itertuples()):
        robust = row.loo_q_value < 0.05
        ax.plot([0, row.meta_z], [y_position, y_position], color="#CFCFCF", lw=1.2)
        ax.scatter(
            row.meta_z, y_position, s=48, color="#B43C52" if robust else "#777777", edgecolor="white", linewidth=0.7
        )
        ax.text(row.meta_z + 0.12, y_position, f"q={row.q_value:.3g}", va="center", fontsize=7)

    # 合并图显示原始有符号 Stouffer z，文字标全模型 q，着色使用删一队列 q。
    ax.axvline(0, color="#555555", lw=0.7)
    ax.set_yticks(y_positions, ordered_meta_table["gene"])
    ax.set_xlabel("Signed Stouffer z")
    ax.set_title("Cross-cohort evidence", loc="left", pad=8)
    ax.text(0.02, -0.22, "Red: leave-one-cohort-out q<0.05", transform=ax.transAxes, fontsize=7, color="#555555")
    open_axes(ax)
    style.public_panel_label(ax, "c")
    save_public(fig, stem="Figure2", panels="abc")

    # Figure3a 的比例分母为患者该区域内所有恶性细胞；双区域轨迹只连接两处都有值的患者。
    fraction_table = pd.read_csv(data_dir / "patient_ac_fraction.csv")
    fig = plt.figure(figsize=(7.2, 5.4))
    grid_spec = GridSpec(2, 3, figure=fig, height_ratios=[1.05, 0.95], hspace=0.55, wspace=0.4)

    # a：三个双区域队列的患者轨迹。
    pair_cohorts = ["GSE273274", "GSE276177", "GSE286419"]
    region_colors = {
        "Tumor": "#A63D40",
        "Core": "#A63D40",
        "Edge": "#D08C38",
        "Peritumoral": "#2B7A78",
        "Peripheral": "#2B7A78",
    }
    for i, cohort in enumerate(pair_cohorts):
        ax = fig.add_subplot(grid_spec[0, i])
        cohort_table = fraction_table[fraction_table["cohort"] == cohort]
        region_table = cohort_table.pivot_table(
            index="patient", columns="region", values="fraction_of_malignant", aggfunc="mean"
        )
        region_table = region_table.reindex(columns=["Peritumoral", "Tumor"]).dropna()
        for _, row in region_table.iterrows():
            ax.plot([0, 1], row[["Peritumoral", "Tumor"]], color="#AFAFAF", lw=0.9)
            ax.scatter(0, row["Peritumoral"], s=19, color=region_colors["Peritumoral"])
            ax.scatter(1, row["Tumor"], s=19, color=region_colors["Tumor"])

        # 每一组粗线端点是完整配对患者在该区域的中位数。
        ax.plot([0, 1], [region_table["Peritumoral"].median(), region_table["Tumor"].median()], color="#222222", lw=2.2)
        ax.set_xticks([0, 1], ["Peritumoral", "Tumor"], rotation=25, ha="right")
        ax.set_ylim(-0.04, 1.04)
        ax.set_title(f"{cohort}\nn={len(region_table)} paired", fontsize=8)
        ax.grid(axis="y", color="#EEEEEE", lw=0.5)
        open_axes(ax)
        if i == 0:
            ax.set_ylabel("AC-like fraction of malignant cells")
            style.public_panel_label(ax, "a")
        else:
            ax.set_ylabel("")

    # b：三位置队列保留患者缺失结构。
    ax = fig.add_subplot(grid_spec[1, :2])
    cohort_table = fraction_table[fraction_table["cohort"] == "GSE274987"]
    region_table = cohort_table.pivot_table(
        index="patient", columns="region", values="fraction_of_malignant", aggfunc="mean"
    )
    region_order = ["Core", "Edge", "Peripheral"]
    for _, row in region_table.iterrows():
        values = row.reindex(region_order).to_numpy(float)
        valid = np.isfinite(values)
        ax.plot(np.arange(len(region_order))[valid], values[valid], color="#B4B4B4", lw=0.8, alpha=0.9)
        for j, value in enumerate(values):
            if np.isfinite(value):
                ax.scatter(j, value, s=19, color=region_colors[region_order[j]], zorder=2)

    # 三位置队列按区域分别取中位数，轨迹保留缺失位置。
    medians = [region_table.get(region, pd.Series(dtype=float)).median() for region in region_order]
    ax.plot(range(len(region_order)), medians, color="#222222", lw=2.3, marker="D", markersize=4)
    ax.set_xticks(range(len(region_order)), region_order)
    ax.set_ylabel("AC-like fraction of malignant cells")
    ax.set_title("GSE274987 regional trajectories", loc="left", pad=8)
    ax.set_ylim(-0.04, 1.04)
    ax.grid(axis="y", color="#EEEEEE", lw=0.5)
    open_axes(ax)
    style.public_panel_label(ax, "b")

    # c：区域配对效应的患者点分布。
    ax = fig.add_subplot(grid_spec[1, 2])
    labels = {
        "GSE273274": "273274 T-P",
        "GSE276177": "276177 T-P",
        "GSE286419": "286419 T-P",
        "GSE274987_core_peripheral": "274987 C-P",
        "GSE274987_core_edge": "274987 C-E",
    }
    test_table = pd.read_csv(data_dir / "regional_ac_fraction_tests.csv")
    comparison_table = test_table.set_index("comparison").loc[list(labels)].reset_index()
    y_positions = np.arange(len(comparison_table))[::-1]

    # 17 只用于图3的纵向点位错开；按冻结比较顺序连续取样。
    rng = np.random.default_rng(17)
    for y_position, row in zip(y_positions, comparison_table.itertuples()):
        diffs = np.asarray(json.loads(row.patient_differences), dtype=float)
        jitter = rng.normal(0, 0.055, size=len(diffs))
        ax.scatter(diffs, np.repeat(y_position, len(diffs)) + jitter, s=18, color=cohort_colors[row.cohort], alpha=0.85)
        ax.scatter(
            row.median_difference, y_position, s=52, marker="D", color="#111111", edgecolor="white", linewidth=0.7
        )
    ax.axvline(0, color="#777777", lw=0.8, ls="--")
    ax.set_yticks(y_positions, [labels[x] for x in comparison_table["comparison"]])
    ax.set_xlabel("Region A minus region B")
    ax.set_title("Paired effects", loc="left", pad=8)
    open_axes(ax)
    style.public_panel_label(ax, "c")
    save_public(fig, stem="Figure3", panels="abc")

    # Figure4a 每个格子对应一个区域比较和配体–受体对，最小可达 P 由非零患者对数决定。
    regional_table = pd.read_csv(data_dir / "regional_communication_tests.csv")
    regional_index_table = regional_table.set_index(["comparison", "pair"], verify_integrity=True)
    fig = plt.figure(figsize=(7.2, 5.7))
    grid_spec = GridSpec(
        2, 2, figure=fig, height_ratios=[1.15, 0.9], width_ratios=[1.25, 0.85], hspace=0.68, wspace=0.45
    )

    comparisons = ["GSE273274", "GSE276177", "GSE286419", "GSE274987_core_peripheral", "GSE274987_core_edge"]
    pairs = ["TGFB2-TGFBR1", "TGFB2-TGFBR2", "TGFB1-TGFBR1", "ANXA1-FPR1", "SPP1-CD44", "CSF1-CSF1R", "CCL2-CCR2"]
    comp_labels = ["273274 T-P", "276177 T-P", "286419 T-P", "274987 C-P", "274987 C-E"]

    # a：所有区域配对通讯效应矩阵。
    ax = fig.add_subplot(grid_spec[0, :])
    max_abs = max(abs(regional_table["median_difference"].min()), abs(regional_table["median_difference"].max()))
    norm = mpl.colors.Normalize(-max_abs, max_abs)
    for i, comparison in enumerate(comparisons):
        for j, pair in enumerate(pairs):
            row = regional_index_table.loc[(comparison, pair), :]
            size = 28 + 180 * abs(row["median_difference"]) / max_abs if max_abs else 28
            color = mpl.colormaps["RdBu_r"](norm(row["median_difference"]))
            significant = pd.notna(row["q_value"]) and row["q_value"] < 0.05
            ax.scatter(
                j,
                len(comparisons) - 1 - i,
                s         = size,
                color     = color,
                edgecolor = "black" if significant else "white",
                linewidth = 1.0 if significant else 0.6,
            )
            ax.text(
                j,
                len(comparisons) - 1 - i - 0.28,
                f"n={int(row['n_patients'])}",
                ha       = "center",
                va       = "top",
                fontsize = 6.5,
                color    = "#555555",
            )
    ax.set_xticks(range(len(pairs)), [p.replace("-", "–") for p in pairs], rotation=30, ha="right")
    ax.set_yticks(range(len(comparisons)), comp_labels[::-1])
    ax.set_xlabel("Ligand–receptor pair", labelpad=1)
    ax.set_ylabel("Within-patient comparison")
    ax.set_title("Regional change in expression co-availability", loc="left", pad=8)
    attainable = regional_table["minimum_attainable_p"].dropna()
    if attainable.empty:
        power_note = "Minimum attainable raw P unavailable"
    elif attainable.lt(0.05).any():
        power_note = f"{attainable.lt(0.05).sum()} of {len(attainable)} evaluable exact tests could attain raw P<0.05"
    elif len(attainable) < len(regional_table):
        power_note = (
            f"No evaluable exact test could attain raw P<0.05 ({len(attainable)}/{len(regional_table)} evaluable)"
        )
    else:
        power_note = "No exact test could attain raw P<0.05 at the available paired sample sizes"
    ax.text(0.0, 1.01, power_note, transform=ax.transAxes, fontsize=7, color="#555555")
    ax.set_xlim(-0.6, len(pairs) - 0.4)
    ax.set_ylim(-0.65, len(comparisons) - 0.35)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Median region A − B", fontsize=7)
    style.public_panel_label(ax, "a")

    # b：TGFB2 两个受体的患者级效应。
    ax = fig.add_subplot(grid_spec[1, 0])
    rows = []
    for comparison in comparisons:
        for pair in ["TGFB2-TGFBR1", "TGFB2-TGFBR2"]:
            rows.append(regional_index_table.loc[(comparison, pair), :])
    y_positions = np.arange(len(rows))[::-1]

    # 23 只用于这一面板的患者散点；同一比较内先 TGFBR1 再 TGFBR2。
    rng = np.random.default_rng(23)
    for y_position, row in zip(y_positions, rows):
        diffs = np.asarray(json.loads(row["patient_differences"]), dtype=float)
        ax.scatter(
            diffs,
            np.repeat(y_position, len(diffs)) + rng.normal(0, 0.045, len(diffs)),
            s     = 15,
            color = cohort_colors[row["cohort"]],
            alpha = 0.85,
        )
        ax.scatter(
            row["median_difference"], y_position, s=42, marker="D", color="#111111", edgecolor="white", linewidth=0.7
        )
    ax.axvline(0, color="#777777", lw=0.8, ls="--")
    row_labels = []
    for comparison in comparisons:
        short = comp_labels[comparisons.index(comparison)]
        row_labels.extend([f"{short} R1", f"{short} R2"])
    ax.set_yticks(y_positions, row_labels)
    ax.set_xlabel("Regional difference in co-availability")
    ax.set_title("TGFB2 receptor-specific effects", loc="left", pad=8)
    open_axes(ax)
    style.public_panel_label(ax, "b")

    # c：内在表达证据与区域通讯证据直接比较。
    ax = fig.add_subplot(grid_spec[1, 1])

    # 每个配体取其区域检验最小 q，再与同配体的内在表达 q 作图。
    ligand_table = regional_table.copy()
    ligand_table["gene"] = ligand_table["pair"].str.split("-").str[0]
    best_q = ligand_table.groupby("gene", observed=True)["q_value"].min()
    for row in meta_table.itertuples():
        if row.gene not in best_q.index:
            continue
        x = -np.log10(max(row.q_value, 1e-12))
        y = -np.log10(max(float(best_q.loc[row.gene]), 1e-12))
        robust = row.loo_q_value < 0.05
        ax.scatter(
            x, y, s=58 if robust else 38, color="#B43C52" if robust else "#777777", edgecolor="white", linewidth=0.7
        )
        offsets = {
            "TGFB2": (0.06, 0.05),
            "CSF1": (0.05, 0.07),
            "ANXA1": (0.05, 0.05),
            "SPP1": (0.05, 0.08),
            "CCL2": (0.05, -0.14),
            "TGFB1": (0.05, 0.14),
        }
        dx, dy = offsets.get(row.gene, (0.05, 0.04))
        ax.text(x + dx, y + dy, row.gene, fontsize=7)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1], 1.4)
    ax.plot([0, lim], [0, lim], color="#B0B0B0", ls="--", lw=0.8)
    ax.axhline(-np.log10(0.05), color="#777777", ls=":", lw=0.8)
    ax.axvline(-np.log10(0.05), color="#777777", ls=":", lw=0.8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Intrinsic AC-like evidence, −log10(q)")
    ax.set_ylabel("Best regional evidence, −log10(q)")
    regional_title = "Intrinsic and regional evidence"
    available_q = regional_table["q_value"].dropna()
    if not available_q.empty and not available_q.lt(0.05).any():
        regional_title = "Regional evidence remains unresolved"
    ax.set_title(regional_title, loc="left", pad=8)
    open_axes(ax)
    style.public_panel_label(ax, "c")
    save_public(fig, stem="Figure4", panels="abc")

    # 补图 S1 每行对应队列、患者及区域。点的颜色和面积分别表示预定细胞数门槛及样本量。
    expression_table = pd.read_csv(data_dir / "patient_state_expression.csv")
    state_order = ["AC-like", "MES-like", "OPC-like", "NPC-like", "Myeloid"]
    expression_table = expression_table[expression_table["state"].isin(state_order)].copy()
    region_short = {"Tumor": "T", "Core": "C", "Edge": "E", "Peritumoral": "P", "Peripheral": "P"}
    samples = []
    for cohort in style.cohort_order:
        cohort_table = expression_table[expression_table["cohort"] == cohort]
        for patient_id, region in cohort_table[["patient", "region"]].drop_duplicates().itertuples(index=False):
            short_patient = str(patient_id).replace("GSE273274_", "").replace("GSE274987_", "")
            samples.append(
                (cohort, patient_id, region, f"{cohort} | {short_patient} | {region_short.get(region, region)}")
            )

    fig, ax = plt.subplots(figsize=(6.5, 10.2))
    row_lookup = {(cohort, patient_id, region): i for i, (cohort, patient_id, region, _) in enumerate(samples)}
    for row in expression_table.itertuples():
        x = state_order.index(row.state)
        y = row_lookup[(row.cohort, row.patient, row.region)]
        below = row.n_cells < 20
        ax.scatter(
            x,
            y,
            s         = 10 + 15 * np.log10(max(row.n_cells, 1)),
            facecolor = "white" if below else cohort_colors[row.cohort],
            edgecolor = cohort_colors[row.cohort],
            linewidth = 0.8,
        )

    # 队列之间划线，保留表达表中首次出现的患者和区域顺序。
    boundaries = []
    last = None
    for i, (cohort, _, _, _) in enumerate(samples):
        if last is not None and cohort != last:
            boundaries.append(i - 0.5)
        last = cohort
    for boundary in boundaries:
        ax.axhline(boundary, color="#D8D8D8", lw=0.7)

    # 空心点保留不足 20 个细胞的记录，供读者核对主分析的纳入条件。
    ax.set_xticks(range(len(state_order)), ["AC", "MES", "OPC", "NPC", "Myeloid"])
    ax.set_yticks(range(len(samples)), [label for _, _, _, label in samples])
    ax.invert_yaxis()
    ax.set_xlabel("Cell state")
    ax.set_ylabel("")
    ax.set_title("Patient-region cell counts and predefined eligibility", loc="left", pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, labelsize=6)
    style.public_panel_label(ax, "a")
    fig.text(
        0.5,
        0.01,
        "Filled: ≥20 cells; open: <20 cells. Point area scales with log10(cell count).",
        ha = "center",
        fontsize = 7,
    )
    save_public(fig, stem="SupplementaryFigureS1", panels="a")

    # 补图 S2 在各队列肿瘤或核心区内汇总合格患者；每一列是配体，每一行是队列和恶性状态。
    z_score_table = style.state_expression_heatmap_table(expression_table)
    z_score_table.index = [f"{cohort} | {state}" for cohort, state in z_score_table.index]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    sns.heatmap(
        z_score_table,
        cmap       = "vlag",
        center     = 0,
        linewidths = 0.35,
        linecolor  = "white",
        cbar_kws   = {"label": "Column z score"},
        ax         = ax,
    )
    ax.set_xlabel("Ligand")
    ax.set_ylabel("")
    ax.set_title("Median patient-level ligand expression by malignant state", loc="left", pad=8)
    style.public_panel_label(ax, "a")
    save_public(fig, stem="SupplementaryFigureS2", panels="a")

    # 补图 S3 以统一注释队列为主保留所有行，GSE162631 的原注释效应按原数据留空。
    patient_table = pd.read_csv(data_dir / "uniform_annotation_patient_contrasts.csv")
    uniform_table = pd.read_csv(data_dir / "uniform_annotation_cohort_results.csv")
    primary_table = contrast_table.loc[contrast_table["gene"].eq("TGFB2"), ["cohort", "median_difference"]]
    primary_table = primary_table.rename(columns={"median_difference": "primary_difference"})
    annotation_table_total = uniform_table.merge(primary_table, on="cohort", how="left", validate="one_to_one")
    summary = json.loads((data_dir / "uniform_annotation_summary.json").read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(7.2, 5.0))
    grid_spec = GridSpec(1, 3, figure=fig, width_ratios=[0.9, 1.0, 1.15], wspace=0.55)

    # a：原标签和统一标签的队列效应对照。
    ax = fig.add_subplot(grid_spec[0, 0])
    order = ["GSE162631", "GSE211376", "GSE273274", "GSE274987", "GSE276177", "GSE286419"]
    annotation_table_total = annotation_table_total.set_index("cohort").loc[order].reset_index()
    y_positions = np.arange(len(annotation_table_total))[::-1]
    for y_position, row in zip(y_positions, annotation_table_total.itertuples()):
        if pd.notna(row.primary_difference):
            ax.plot([row.primary_difference, row.median_difference], [y_position, y_position], color="#B8B8B8", lw=1.0)
            ax.scatter(
                row.primary_difference,
                y_position,
                s         = 32,
                facecolor = "white",
                edgecolor = cohort_colors.get(row.cohort, "#555555"),
                linewidth = 1.0,
            )
        ax.scatter(
            row.median_difference,
            y_position,
            s         = 38,
            color     = cohort_colors.get(row.cohort, "#555555"),
            edgecolor = "white",
            linewidth = 0.6,
        )

    # 空心点表示原注释，实心点表示统一注释；同一队列的两个估计相连。
    ax.axvline(0, color="#777777", lw=0.8, ls="--")
    ax.set_yticks(y_positions, annotation_table_total["cohort"])
    ax.set_xlabel("Median TGFB2 difference")
    ax.set_title("Annotation sensitivity", loc="left", pad=8)
    handles = []
    for face, edge, label in [("white", "#555555", "Primary labels"), ("#555555", "white", "Uniform markers")]:
        handle = Line2D([0], [0], marker="o", color="none", markerfacecolor=face, markeredgecolor=edge, label=label)
        handles.append(handle)
    ax.legend(handles=handles, loc="upper right", fontsize=6.5)
    open_axes(ax)
    style.public_panel_label(ax, "a")

    # b：统一标签下每位患者的差值。
    ax = fig.add_subplot(grid_spec[0, 1])

    # 41 只用于统一注释患者点的位置，队列顺序与图中标签一致。
    rng = np.random.default_rng(41)
    for i, cohort in enumerate(order):
        values = patient_table.loc[patient_table["cohort"] == cohort, "tgfb2_difference"].to_numpy(float)
        ax.scatter(
            np.repeat(i, len(values)) + rng.normal(0, 0.055, len(values)),
            values,
            s     = 19,
            color = cohort_colors.get(cohort, "#555555"),
            alpha = 0.85,
        )
        if len(values):
            ax.scatter(i, np.median(values), s=48, marker="D", color="#111111", edgecolor="white", linewidth=0.7)
    ax.axhline(0, color="#777777", lw=0.8, ls="--")
    ax.set_xticks(range(len(order)), [item.replace("GSE", "") for item in order], rotation=45, ha="right")
    ax.set_ylabel("Uniform AC-like minus other TGFB2")
    ax.set_xlabel("Public cohort")
    ax.set_title("Patient-level effects", loc="left", pad=8)
    open_axes(ax)
    style.public_panel_label(ax, "b")

    # c：TGFB2 与 AC marker 强度的关系和校正后估计。
    ax = fig.add_subplot(grid_spec[0, 2])
    for cohort, cohort_table in patient_table.groupby("cohort", observed=True):
        ax.scatter(
            cohort_table["ac_score_difference"],
            cohort_table["tgfb2_difference"],
            s     = 25,
            color = cohort_colors.get(cohort, "#555555"),
            alpha = 0.85,
            label = cohort,
        )
    x = patient_table["ac_score_difference"].to_numpy(float)

    # 直线仅为描绘患者差值的线性拟合；相关系数和 HC3 结果直接读取上游摘要。
    y = patient_table["tgfb2_difference"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted_x = np.linspace(x.min(), x.max(), 100)
    ax.plot(fitted_x, intercept + slope * fitted_x, color="#333333", lw=1.0)
    lineage = summary["lineage_adjustment"]
    ax.text(
        0.03,
        0.97,
        f"Spearman ρ={lineage['spearman_rho']:.2f}\n"
        f"HC3 adjusted mean={lineage['grand_mean_at_average_ac_score']:.3f}\nP={lineage['hc3_p_value']:.2g}",
        transform = ax.transAxes,
        va        = "top",
        fontsize  = 7,
        bbox      = dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5),
    )
    ax.axhline(0, color="#B0B0B0", lw=0.7, ls="--")
    ax.set_xlabel("AC marker-score difference")
    ax.set_ylabel("TGFB2 expression difference")
    ax.set_title("Lineage-program sensitivity", loc="left", pad=8)
    open_axes(ax)
    style.public_panel_label(ax, "c")
    save_public(fig, stem="SupplementaryFigureS3", panels="abc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制公共队列的原分析图")
    parser.add_argument("--output", type=Path, help="公共队列图的输出目录，默认使用投稿包 figures/")
    main(parser.parse_args().output)
