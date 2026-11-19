#!/usr/bin/env python3
# 目的：考察 marker 原均值与队列内基因 z 分数、置信差阈值、每组细胞数对患者 TGFB2 结果的影响。
# 输入：与 02_统一注释敏感性.py 相同的五个公共输入；用 score_threshold、minimum_cells 和 scoring_mode 区分组合。
# 输出：重算包 data/parameter_sensitivity_grid.csv、data/parameter_sensitivity_cohort_results.csv、
# data/parameter_sensitivity_summary.json 及 figures/SupplementaryFigureS4.pdf、figures/SupplementaryFigureS4.png。
# 运行方式：在 code 目录运行 python 03_参数敏感性.py。单独调用 make_figure() 可从已有两张参数表绘图。
# 保留原 Python 数值库，独立包写入 01_公共队列重算/，PNG 从同一份 PDF 转出。

from __future__ import annotations

from importlib import import_module
from itertools import product

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

uniform = import_module("02_统一注释敏感性")
paths = import_module("00a_项目路径")


def main() -> None:
    """生成参数结果，核实每组的可估计队列和人数，再绘制补图。无返回值。"""
    uniform.primary.stop_if_missing()
    grid_table, cohort_table_total = calculate_grid()
    key_columns = ["scoring_mode", "score_threshold", "minimum_cells"]
    expected = set(product(scoring_modes, score_thresholds, cell_thresholds))
    if set(grid_table[key_columns].itertuples(index=False, name=None)) != expected:
        raise ValueError("敏感性汇总缺少预设组合，或含额外组合")

    # 队列表额外保留不足三人的描述行；汇总中的人数与队列数只来自有双侧 P 值的队列。
    valid_table = cohort_table_total.loc[cohort_table_total["p_value"].notna() & cohort_table_total["n_patients"].ge(3)]
    counts_table = valid_table.groupby(key_columns)["n_patients"].agg(["size", "sum"])
    grid_indexed_table = grid_table.set_index(key_columns)
    counts_table = counts_table.reindex(grid_indexed_table.index, fill_value=0)
    if not counts_table["size"].equals(grid_indexed_table["n_cohorts"]):
        raise ValueError("参数汇总中的可估计队列数与队列明细不符")
    if not counts_table["sum"].equals(grid_indexed_table["n_patients"]):
        raise ValueError("参数汇总中的患者数与队列明细不符")

    output = paths.analysis_output_root() / "data"
    paths.write_table(grid_table, output / "parameter_sensitivity_grid.csv", key_columns)
    paths.write_table(cohort_table_total, output / "parameter_sensitivity_cohort_results.csv", [*key_columns, "cohort"])
    summary = {
        "parameter_combinations": len(grid_table),
        "all_meta_p_below_0_05": bool((grid_table["meta_p_value"] < 0.05).all()),
        "minimum_positive_cohorts": int(grid_table["positive_cohorts"].min()),
        "minimum_meta_z": float(grid_table["meta_z"].min()),
        "maximum_meta_p_value": float(grid_table["meta_p_value"].max()),
        "maximum_leave_one_out_p_value": float(grid_table["loo_max_p_value"].max()),
    }
    paths.write_json(summary, output / "parameter_sensitivity_summary.json")
    make_figure()
    print(f"{len(grid_table)} 个参数组合，{len(cohort_table_total)} 行队列结果。")


# 两种计分各自读一次原始表达。阈值只在细胞已经取得最高分状态之后生效。
scoring_modes = {"Raw marker mean": False, "Within-cohort gene z-score": True}


def calculate_grid() -> tuple[pd.DataFrame, pd.DataFrame]:
    """遍历发表网格；逐次传入参数，其他脚本的默认阈值不随循环改变。"""
    grid_rows = []
    cohort_parts = []
    for mode, standardize in scoring_modes.items():
        cell_table_total = uniform.load_uniform_cells(standardize=standardize, score_threshold=0.0)
        for score_threshold in score_thresholds:
            eligible_table = cell_table_total.loc[~(cell_table_total["confidence"] < score_threshold)]
            for cell_threshold in cell_thresholds:
                patient_table = uniform.patient_contrasts(eligible_table, minimum_cells=cell_threshold)
                cohort_table = uniform.cohort_results(patient_table)
                meta = uniform.meta_result(cohort_table)

                # 一份是每个组合的合并结果，另一份保留该组合下各队列（包括不可检验行）。
                parameters = {"scoring_mode": mode, "score_threshold": score_threshold, "minimum_cells": cell_threshold}
                grid_rows.append(
                    {
                        **parameters,
                        "n_cohorts": meta["n_cohorts"],
                        "n_patients": meta["n_patients"],
                        "positive_cohorts": meta["positive_cohorts"],
                        "meta_z": meta["meta_z"],
                        "meta_p_value": meta["p_value"],
                        "loo_max_p_value": meta["loo_max_p_value"],
                    }
                )
                cohort_table.insert(0, "minimum_cells", cell_threshold)
                cohort_table.insert(0, "score_threshold", score_threshold)
                cohort_table.insert(0, "scoring_mode", mode)
                cohort_parts.append(cohort_table)

    grid_table = pd.DataFrame(grid_rows)
    cohort_table_total = pd.concat(cohort_parts, ignore_index=True)
    result = grid_table, cohort_table_total
    return result


# 原方案的两条阈值轴按发表顺序排列，基准点与 02 的默认值相同。
score_thresholds = [0.0, 0.025, 0.05, 0.10]
cell_thresholds = [10, 20, 30, 50]


def make_figure() -> None:
    """回读两个来源相同的 CSV，绘制 S4 的两个热图和同队列的配对点。无返回值。"""
    directory = paths.result_input_directory(
        "parameter_sensitivity_grid.csv", "parameter_sensitivity_cohort_results.csv"
    )
    grid_table = pd.read_csv(directory / "parameter_sensitivity_grid.csv")
    cohort_table = pd.read_csv(directory / "parameter_sensitivity_cohort_results.csv")

    # 只在本图使用 seaborn 的白底样式和原字体，返回时恢复调用者的 Matplotlib 设置。
    with mpl.rc_context():
        sns.set_theme(style="white")
        mpl.rcParams.update(
            {
                "font.family": "Arial",
                "font.size": 8,
                "axes.labelsize": 8,
                "axes.titlesize": 9,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "pdf.fonttype": 42,
                "figure.facecolor": "white",
            }
        )
        figure = plt.figure(figsize=(7.3, 6.4))
        try:
            layout = figure.add_gridspec(2, 2, height_ratios=[1.0, 0.82], hspace=0.52, wspace=0.48)
            for index, mode in enumerate(scoring_modes):
                axis = figure.add_subplot(layout[0, index])
                draw_heatmap(axis, grid_table.loc[grid_table["scoring_mode"] == mode], mode)
                axis.text(-0.13, 1.08, "ab"[index], transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")

            axis = figure.add_subplot(layout[1, :])
            draw_baseline(axis, cohort_table)
            figure.suptitle(
                "Uniform-marker TGFB2 results across analysis parameters",
                x=0.07,
                ha="left",
                fontsize=11,
                fontweight="bold",
            )
            paths.save_figure(figure, paths.analysis_output_root() / "figures" / "SupplementaryFigureS4.pdf")
        finally:
            plt.close(figure)


def draw_heatmap(axis: plt.Axes, grid_table: pd.DataFrame, title: str) -> None:
    """每格使用同一行的合并 z、正向队列数和可估计队列数；缺组合则停止。无返回值。"""
    indexed_table = grid_table.set_index(["minimum_cells", "score_threshold"], verify_integrity=True)
    if set(indexed_table.index) != set(product(cell_thresholds, score_thresholds)):
        raise ValueError(f"{title} 的热图格子与预设网格不符")
    values_table = indexed_table["meta_z"].unstack().loc[cell_thresholds, score_thresholds]
    labels = []
    for row in indexed_table.itertuples():
        labels.append(f"z={row.meta_z:.2f}\n{row.positive_cohorts}/{row.n_cohorts} +")
    indexed_table["label"] = labels
    labels_table = indexed_table["label"].unstack().loc[cell_thresholds, score_thresholds]
    sns.heatmap(
        values_table,
        ax=axis,
        annot=labels_table,
        fmt="",
        cmap="viridis",
        vmin=3.0,
        vmax=4.2,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Signed Stouffer z", "shrink": 0.78},
        annot_kws={"fontsize": 6.7},
    )
    axis.set_xlabel("Top-versus-second score threshold")
    axis.set_ylabel("Minimum cells per group")
    axis.set_xticklabels([f"{value:g}" for value in values_table.columns])
    axis.set_yticklabels([str(value) for value in values_table.index], rotation=0)
    axis.set_title(title, loc="left", pad=8)


def draw_baseline(axis: plt.Axes, cohort_table: pd.DataFrame) -> None:
    """S4c 沿用发表的五个队列。GSE276177 的不足三人结果仍保留在完整队列表中。无返回值。"""
    baseline = np.isclose(cohort_table["score_threshold"], uniform.minimum_score_difference)
    baseline &= cohort_table["minimum_cells"].eq(uniform.primary.minimum_cells)
    baseline_table = cohort_table.loc[baseline]
    order = ["GSE162631", "GSE211376", "GSE273274", "GSE274987", "GSE286419"]
    wide_table = baseline_table.pivot(index="cohort", columns="scoring_mode", values="median_difference")
    wide_table = wide_table.loc[order, list(scoring_modes)]
    if wide_table.isna().any(axis=None):
        raise ValueError("S4c 有队列缺少一种计分方式的基准差值")

    positions = np.arange(len(order))[::-1]
    raw_means = wide_table["Raw marker mean"].to_numpy(float)
    standardized = wide_table["Within-cohort gene z-score"].to_numpy(float)
    for y, left, right in zip(positions, raw_means, standardized):
        axis.plot([left, right], [y, y], color="#B7B7B7", lw=1.0, zorder=1)
    axis.scatter(
        raw_means, positions, color="#C84B31", s=38, edgecolor="white", linewidth=0.7, label="Raw marker mean", zorder=2
    )
    axis.scatter(
        standardized, positions, color="#2B7A78", s=38, edgecolor="white", linewidth=0.7, label="Gene z-score", zorder=2
    )
    axis.axvline(0, color="#777777", lw=0.8, ls="--")
    axis.set_yticks(positions, order)
    axis.set_xlabel("Cohort median TGFB2 difference (AC-like - other states)")
    axis.set_title("Baseline parameters: cohort direction under two scoring scales", loc="left", pad=8)
    axis.legend(frameon=False, loc="lower right")
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(-0.13, 1.08, "c", transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")


if __name__ == "__main__":
    main()
