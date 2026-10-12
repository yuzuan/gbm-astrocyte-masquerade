#!/usr/bin/env python3
"""检验统一 marker 结果对置信阈值、细胞数阈值和计分尺度的敏感性。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import stage48_uniform_annotation_sensitivity as base
from stage_public_paths import project_root, submission_package


ROOT = project_root()
PACKAGE = submission_package(ROOT)
DATA = PACKAGE / "data"
FIGURES = PACKAGE / "figures"

SCORE_THRESHOLDS = [0.0, 0.025, 0.05, 0.10]
CELL_THRESHOLDS = [10, 20, 30, 50]
SCORING_MODES = {
    "Raw marker mean": False,
    "Within-cohort gene z-score": True,
}
BASELINE_SCORE_THRESHOLD = 0.05
BASELINE_CELL_THRESHOLD = 20


def load_uniform_cells() -> pd.DataFrame:
    """按当前全局计分设置读取全部公共恶性细胞并统一标注。"""
    parts = [base.read_gse211376()]
    for accession in ["GSE162631", "GSE273274"]:
        parts.append(
            base.read_h5ad_malignant(
                base.SCGBM,
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
            base.read_h5ad_malignant(
                base.SNGBM,
                cohort=accession,
                patient_col="patient",
                region_col="tissue",
                dataset_col="dataset",
                dataset_filter=accession,
                malignant_mask=lambda obs: obs["celltype"].astype(str).isin(base.STATES),
            )
        )
    parts.append(
        base.read_h5ad_malignant(
            base.GSE274,
            cohort="GSE274987",
            patient_col="patient_id",
            region_col="tissue_type",
            dataset_col="dataset",
            dataset_filter="GSE274987",
            malignant_mask=lambda obs: obs["celltype"].astype(str).isin(base.STATES),
        )
    )
    return pd.concat(parts, ignore_index=True)


def calculate_grid() -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐一重算参数组合，保留总体和队列级结果。"""
    grid_rows: list[dict[str, object]] = []
    cohort_rows: list[dict[str, object]] = []
    original_settings = (base.MIN_SCORE_DIFF, base.MIN_CELLS, base.STANDARDIZE_MARKER_GENES)
    try:
        for scoring_mode, standardize in SCORING_MODES.items():
            base.STANDARDIZE_MARKER_GENES = standardize
            for score_threshold in SCORE_THRESHOLDS:
                base.MIN_SCORE_DIFF = score_threshold
                cells = load_uniform_cells()
                for cell_threshold in CELL_THRESHOLDS:
                    base.MIN_CELLS = cell_threshold
                    patients = base.patient_contrasts(cells)
                    cohorts = base.cohort_results(patients)
                    meta = base.meta_result(cohorts)
                    grid_rows.append(
                        {
                            "scoring_mode": scoring_mode,
                            "score_threshold": score_threshold,
                            "minimum_cells": cell_threshold,
                            "n_cohorts": meta["n_cohorts"],
                            "n_patients": meta["n_patients"],
                            "positive_cohorts": meta["positive_cohorts"],
                            "meta_z": meta["meta_z"],
                            "meta_p_value": meta["p_value"],
                            "loo_max_p_value": meta["loo_max_p_value"],
                        }
                    )
                    for row in cohorts.itertuples(index=False):
                        cohort_rows.append(
                            {
                                "scoring_mode": scoring_mode,
                                "score_threshold": score_threshold,
                                "minimum_cells": cell_threshold,
                                "cohort": row.cohort,
                                "n_patients": row.n_patients,
                                "median_difference": row.median_difference,
                                "positive_patients": row.positive_patients,
                                "p_value": row.p_value,
                                "q_value": row.q_value,
                            }
                        )
    finally:
        base.MIN_SCORE_DIFF, base.MIN_CELLS, base.STANDARDIZE_MARKER_GENES = original_settings
    return pd.DataFrame(grid_rows), pd.DataFrame(cohort_rows)


def configure_style() -> None:
    """统一补图字体和矢量输出设置。"""
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
            "savefig.dpi": 600,
            "figure.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    """添加统一面板标签。"""
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def draw_heatmap(ax: plt.Axes, frame: pd.DataFrame, title: str) -> None:
    """绘制 meta z 热图，并标出阳性队列数和总队列数。"""
    values = frame.pivot(index="minimum_cells", columns="score_threshold", values="meta_z").loc[CELL_THRESHOLDS]
    labels = frame.copy()
    labels["label"] = labels.apply(
        lambda row: f"z={row.meta_z:.2f}\n{int(row.positive_cohorts)}/{int(row.n_cohorts)} +",
        axis=1,
    )
    annotations = labels.pivot(index="minimum_cells", columns="score_threshold", values="label").loc[CELL_THRESHOLDS]
    sns.heatmap(
        values,
        ax=ax,
        annot=annotations,
        fmt="",
        cmap="viridis",
        vmin=3.0,
        vmax=4.2,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Signed Stouffer z", "shrink": 0.78},
        annot_kws={"fontsize": 6.7},
    )
    ax.set_xlabel("Top-versus-second score threshold")
    ax.set_ylabel("Minimum cells per group")
    ax.set_xticklabels([f"{value:g}" for value in values.columns])
    ax.set_yticklabels([str(value) for value in values.index], rotation=0)
    ax.set_title(title, loc="left", pad=8)


def make_figure() -> None:
    """所有图中数值均从已写出的 CSV 回读。"""
    grid_path = DATA / "parameter_sensitivity_grid.csv"
    cohort_path = DATA / "parameter_sensitivity_cohort_results.csv"
    if not grid_path.is_file() or not cohort_path.is_file():
        raise FileNotFoundError("缺少参数敏感性结果表")
    grid = pd.read_csv(grid_path)
    cohort = pd.read_csv(cohort_path)
    configure_style()

    fig = plt.figure(figsize=(7.3, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.82], hspace=0.52, wspace=0.48)
    for index, mode in enumerate(SCORING_MODES):
        ax = fig.add_subplot(gs[0, index])
        draw_heatmap(ax, grid[grid["scoring_mode"] == mode], mode)
        panel_label(ax, chr(ord("a") + index))

    # 基准参数下直接比较两种 marker 计分尺度的队列中位数。
    ax = fig.add_subplot(gs[1, :])
    baseline = cohort[
        np.isclose(cohort["score_threshold"], BASELINE_SCORE_THRESHOLD)
        & cohort["minimum_cells"].eq(BASELINE_CELL_THRESHOLD)
    ]
    order = ["GSE162631", "GSE211376", "GSE273274", "GSE274987", "GSE286419"]
    wide = baseline.pivot(index="cohort", columns="scoring_mode", values="median_difference").reindex(order)
    y = np.arange(len(order))[::-1]
    raw = wide["Raw marker mean"].to_numpy(float)
    standardized = wide["Within-cohort gene z-score"].to_numpy(float)
    for yi, left, right in zip(y, raw, standardized):
        ax.plot([left, right], [yi, yi], color="#B7B7B7", lw=1.0, zorder=1)
    ax.scatter(raw, y, color="#C84B31", s=38, edgecolor="white", linewidth=0.7, label="Raw marker mean", zorder=2)
    ax.scatter(standardized, y, color="#2B7A78", s=38, edgecolor="white", linewidth=0.7, label="Gene z-score", zorder=2)
    ax.axvline(0, color="#777777", lw=0.8, ls="--")
    ax.set_yticks(y, order)
    ax.set_xlabel("Cohort median TGFB2 difference (AC-like - other states)")
    ax.set_title("Baseline parameters: cohort direction under two scoring scales", loc="left", pad=8)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    panel_label(ax, "c")

    fig.suptitle("Aggregate uniform-marker TGFB2 result is stable across analysis parameters", x=0.07, ha="left", fontsize=11, fontweight="bold")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "SupplementaryFigureS4.png", dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(FIGURES / "SupplementaryFigureS4.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """执行参数网格、写表并生成补图。"""
    required = [base.COUNTS_211, base.META_211, base.SCGBM, base.SNGBM, base.GSE274]
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("缺少公共输入：\n" + "\n".join(missing))
    DATA.mkdir(parents=True, exist_ok=True)
    grid, cohorts = calculate_grid()
    grid.to_csv(DATA / "parameter_sensitivity_grid.csv", index=False)
    cohorts.to_csv(DATA / "parameter_sensitivity_cohort_results.csv", index=False)
    summary = {
        "parameter_combinations": int(len(grid)),
        "all_meta_p_below_0_05": bool((grid["meta_p_value"] < 0.05).all()),
        "minimum_positive_cohorts": int(grid["positive_cohorts"].min()),
        "minimum_meta_z": float(grid["meta_z"].min()),
        "maximum_meta_p_value": float(grid["meta_p_value"].max()),
        "maximum_leave_one_out_p_value": float(grid["loo_max_p_value"].max()),
    }
    (DATA / "parameter_sensitivity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_figure()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
