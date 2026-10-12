#!/usr/bin/env python3
"""根据 stage45 的公共数据结果生成论文主图和补图。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from stage_public_paths import project_root, submission_package


ROOT = project_root()
PACKAGE = submission_package(ROOT)
DATA = PACKAGE / "data"
OUT = PACKAGE / "figures"

COHORT_ORDER = ["GSE211376", "GSE273274", "GSE274987", "GSE276177", "GSE286419"]
COHORT_COLORS = {
    "GSE211376": "#2B6F77",
    "GSE273274": "#D96C4F",
    "GSE274987": "#C49A2C",
    "GSE276177": "#6B5B95",
    "GSE286419": "#4C78A8",
}
STATE_COLORS = {"AC-like": "#C84B31", "Other malignant": "#3D6F8E", "Myeloid": "#4D8061"}
REGION_COLORS = {"Tumor": "#A63D40", "Core": "#A63D40", "Edge": "#D08C38", "Peritumoral": "#2B7A78", "Peripheral": "#2B7A78"}
POS = "#B43C52"
NEG = "#2D6F9F"
NEUTRAL = "#D9D9D9"


def configure_style() -> None:
    """统一期刊图形样式。"""
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


def require_files(names: list[str]) -> None:
    """绘图只读取重算表，缺表立即停止。"""
    missing = [str(DATA / name) for name in names if not (DATA / name).is_file()]
    if missing:
        raise FileNotFoundError("缺少绘图输入：\n" + "\n".join(missing))


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def open_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=3)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=600, facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def figure1() -> None:
    manifest = pd.read_csv(DATA / "cohort_manifest.csv").set_index("cohort").loc[COHORT_ORDER].reset_index()
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.05, 1.0], width_ratios=[1.25, 0.9], hspace=0.55, wspace=0.45)

    # a：公共队列与区域覆盖矩阵。
    ax = fig.add_subplot(gs[0, 0])
    regions = ["Tumor/Core", "Edge", "Peritumoral/Peripheral"]
    y = np.arange(len(manifest))[::-1]
    for yi, row in zip(y, manifest.itertuples()):
        row_regions = set(str(row.regions).split("; "))
        present = [
            bool(row_regions & {"Tumor", "Core"}),
            "Edge" in row_regions,
            bool(row_regions & {"Peritumoral", "Peripheral"}),
        ]
        xpos = np.flatnonzero(present)
        if len(xpos) > 1:
            ax.plot(xpos, np.repeat(yi, len(xpos)), color="#B8B8B8", lw=1.2, zorder=1)
        for xi in range(3):
            ax.scatter(xi, yi, s=18 if not present[xi] else 38 + row.patients * 4, facecolor=COHORT_COLORS[row.cohort] if present[xi] else "white", edgecolor="#A8A8A8", linewidth=0.7, zorder=2)
        ax.text(2.42, yi, f"n={row.patients}", va="center", fontsize=7, color="#444444")
    ax.set_xticks(range(3), regions)
    ax.set_yticks(y, manifest["cohort"])
    ax.set_xlim(-0.45, 3.05)
    ax.set_ylim(-0.6, len(y) - 0.4)
    ax.set_title("Public cohorts and spatial sampling", loc="left", pad=8)
    ax.set_xlabel("Tissue region")
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_label(ax, "a")

    # b：真实纳入细胞与患者数量，不用柱状图。
    ax = fig.add_subplot(gs[0, 1])
    for yi, row in zip(y, manifest.itertuples()):
        ax.scatter(row.included_cells, yi, s=26 + row.patients * 7, color=COHORT_COLORS[row.cohort], edgecolor="white", linewidth=0.7)
        ax.text(row.included_cells * 1.05, yi, f"{row.included_cells:,}", va="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_yticks(y, manifest["cohort"])
    ax.set_xlabel("Included cells/nuclei (log scale)")
    ax.set_title("Analysis set", loc="left", pad=8)
    ax.grid(axis="x", color="#EAEAEA", lw=0.6)
    open_axes(ax)
    panel_label(ax, "b")

    # c：两个不同证据层级的统计对象。
    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 4)
    ax.axis("off")
    # 内在表达分支。
    ax.scatter([0.8, 0.8], [2.9, 2.0], s=[760, 760], color=[STATE_COLORS["AC-like"], STATE_COLORS["Other malignant"]], edgecolor="white", linewidth=1)
    ax.text(0.8, 2.9, "AC-like", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
    ax.text(0.8, 2.0, "Other\nstates", ha="center", va="center", color="white", fontsize=6.5, fontweight="bold")
    ax.annotate("", xy=(2.25, 2.45), xytext=(1.45, 2.45), arrowprops=dict(arrowstyle="->", color="#555555", lw=1))
    ax.text(2.75, 2.45, "Patient-level\nexpression difference", ha="center", va="center", fontsize=7.5)
    ax.annotate("", xy=(4.1, 2.45), xytext=(3.55, 2.45), arrowprops=dict(arrowstyle="->", color="#555555", lw=1))
    ax.text(4.85, 2.45, "Cohort-specific\neffect", ha="center", va="center", fontsize=7.5)
    ax.annotate("", xy=(6.25, 2.45), xytext=(5.65, 2.45), arrowprops=dict(arrowstyle="->", color="#555555", lw=1))
    ax.scatter(7.0, 2.45, s=470, color="#F1C75B", edgecolor="#806000", linewidth=0.8)
    ax.text(7.0, 2.45, "Cross-cohort\nmeta-analysis", ha="center", va="center", fontsize=7)

    # 通讯分支。
    ax.scatter(0.8, 0.72, s=760, color=STATE_COLORS["Myeloid"], edgecolor="white", linewidth=1)
    ax.text(0.8, 0.72, "Myeloid", ha="center", va="center", color="white", fontsize=6.8, fontweight="bold")
    ax.annotate("", xy=(2.25, 0.72), xytext=(1.45, 0.72), arrowprops=dict(arrowstyle="->", color="#555555", lw=1))
    ax.text(2.85, 0.72, "Ligand-receptor\nco-availability", ha="center", va="center", fontsize=7.5)
    ax.annotate("", xy=(4.55, 0.72), xytext=(3.65, 0.72), arrowprops=dict(arrowstyle="->", color="#555555", lw=1))
    ax.text(5.35, 0.72, "Within-patient\nregional difference", ha="center", va="center", fontsize=7.5)
    ax.annotate("", xy=(6.8, 0.72), xytext=(6.15, 0.72), arrowprops=dict(arrowstyle="->", color="#555555", lw=1))
    ax.scatter(7.55, 0.72, s=440, facecolor="white", edgecolor="#555555", linewidth=1.0)
    ax.text(7.55, 0.72, "Replication\nacross regions", ha="center", va="center", fontsize=7)
    ax.plot([8.35, 8.35], [0.25, 3.25], ls="--", color="#888888", lw=0.8)
    ax.text(8.35, 3.38, "Inference boundary", ha="center", fontsize=7, color="#666666")
    ax.text(8.65, 2.38, "Expression enrichment", ha="left", fontsize=7, color="#333333")
    ax.text(8.65, 0.62, "No causal signaling claim", ha="left", fontsize=7, color="#333333")
    ax.text(0.0, 3.75, "Patient-level estimands and evidence hierarchy", fontsize=9)
    panel_label(ax, "c")
    save_figure(fig, "Figure1")


def _paired_values(row: pd.Series) -> pd.DataFrame:
    values = json.loads(row["patient_values"])
    return pd.DataFrame(values)


def figure2() -> None:
    contrasts = pd.read_csv(DATA / "ac_vs_other_ligand_contrasts.csv")
    meta = pd.read_csv(DATA / "meta_ligand_results.csv")
    inferential = ["GSE211376", "GSE273274", "GSE274987", "GSE286419"]
    fig = plt.figure(figsize=(7.2, 6.1))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.2], width_ratios=[1.45, 0.75], hspace=0.55, wspace=0.42)

    # a：每个公共队列的患者内 TGFB2 配对值。
    sub = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0, :], wspace=0.35)
    for i, cohort in enumerate(inferential):
        ax = fig.add_subplot(sub[0, i])
        row = contrasts[(contrasts["cohort"] == cohort) & (contrasts["gene"] == "TGFB2")].iloc[0]
        values = _paired_values(row)
        for _, item in values.iterrows():
            ax.plot([0, 1], [item["other_mean"], item["ac_mean"]], color="#B8B8B8", lw=0.8, zorder=1)
            ax.scatter(0, item["other_mean"], s=17, color=STATE_COLORS["Other malignant"], zorder=2)
            ax.scatter(1, item["ac_mean"], s=17, color=STATE_COLORS["AC-like"], zorder=2)
        ax.plot([0, 1], [values["other_mean"].median(), values["ac_mean"].median()], color="#222222", lw=2.2, zorder=3)
        ax.set_xticks([0, 1], ["Other", "AC-like"])
        ax.set_title(f"{cohort}\nn={len(values)} patients", loc="center", fontsize=8)
        ax.set_xlim(-0.25, 1.25)
        ax.grid(axis="y", color="#EFEFEF", lw=0.5)
        open_axes(ax)
        if i == 0:
            ax.set_ylabel("Mean log-normalized TGFB2")
            panel_label(ax, "a")
        else:
            ax.set_ylabel("")

    # b：所有配体在各队列中的效应方向与强度。
    ax = fig.add_subplot(gs[1, 0])
    gene_order = ["TGFB2", "CSF1", "ANXA1", "CCL2", "TGFB1", "SPP1"]
    cohort_order = ["GSE211376", "GSE273274", "GSE274987", "GSE276177", "GSE286419"]
    xmap = {gene: i for i, gene in enumerate(gene_order)}
    ymap = {cohort: i for i, cohort in enumerate(cohort_order[::-1])}
    max_abs = max(abs(contrasts["median_difference"].min()), abs(contrasts["median_difference"].max()))
    for row in contrasts.itertuples():
        x, y = xmap[row.gene], ymap[row.cohort]
        color = mpl.colormaps["RdBu_r"](mpl.colors.Normalize(-max_abs, max_abs)(row.median_difference))
        size = 26 + 7 * row.n_patients
        edge = "black" if pd.notna(row.q_value) and row.q_value < 0.05 else "white"
        ax.scatter(x, y, s=size, color=color, edgecolor=edge, linewidth=1.0 if edge == "black" else 0.6)
    ax.set_xticks(range(len(gene_order)), gene_order, rotation=35, ha="right")
    ax.set_yticks(range(len(cohort_order)), cohort_order[::-1])
    ax.set_xlabel("Ligand")
    ax.set_ylabel("Public cohort")
    ax.set_title("AC-like minus other malignant states", loc="left", pad=8)
    ax.set_xlim(-0.6, len(gene_order) - 0.4)
    ax.set_ylim(-0.6, len(cohort_order) - 0.4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    norm = mpl.colors.Normalize(-max_abs, max_abs)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Median expression difference", fontsize=7)
    panel_label(ax, "b")

    # c：合并证据及删一队列敏感性。
    ax = fig.add_subplot(gs[1, 1])
    meta = meta.set_index("gene").loc[gene_order].reset_index()
    yy = np.arange(len(meta))[::-1]
    for yv, row in zip(yy, meta.itertuples()):
        robust = row.loo_q_value < 0.05
        ax.plot([0, row.meta_z], [yv, yv], color="#CFCFCF", lw=1.2)
        ax.scatter(row.meta_z, yv, s=48, color=POS if robust else "#777777", edgecolor="white", linewidth=0.7)
        ax.text(row.meta_z + 0.12, yv, f"q={row.q_value:.3g}", va="center", fontsize=7)
    ax.axvline(0, color="#555555", lw=0.7)
    ax.set_yticks(yy, meta["gene"])
    ax.set_xlabel("Signed Stouffer z")
    ax.set_title("Cross-cohort evidence", loc="left", pad=8)
    ax.text(0.02, -0.22, "Red: leave-one-cohort-out q<0.05", transform=ax.transAxes, fontsize=7, color="#555555")
    open_axes(ax)
    panel_label(ax, "c")
    save_figure(fig, "Figure2")


def figure3() -> None:
    ac = pd.read_csv(DATA / "patient_ac_fraction.csv")
    tests = pd.read_csv(DATA / "regional_ac_fraction_tests.csv")
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.05, 0.95], hspace=0.55, wspace=0.4)

    # a：三个双区域队列的患者轨迹。
    pair_cohorts = ["GSE273274", "GSE276177", "GSE286419"]
    for i, cohort in enumerate(pair_cohorts):
        ax = fig.add_subplot(gs[0, i])
        frame = ac[ac["cohort"] == cohort]
        wide = frame.pivot_table(index="patient", columns="region", values="fraction_of_malignant", aggfunc="mean")
        wide = wide[[c for c in ["Peritumoral", "Tumor"] if c in wide]].dropna()
        for _, row in wide.iterrows():
            ax.plot([0, 1], row[["Peritumoral", "Tumor"]], color="#AFAFAF", lw=0.9)
            ax.scatter(0, row["Peritumoral"], s=19, color=REGION_COLORS["Peritumoral"])
            ax.scatter(1, row["Tumor"], s=19, color=REGION_COLORS["Tumor"])
        ax.plot([0, 1], [wide["Peritumoral"].median(), wide["Tumor"].median()], color="#222222", lw=2.2)
        ax.set_xticks([0, 1], ["Peritumoral", "Tumor"], rotation=25, ha="right")
        ax.set_ylim(-0.04, 1.04)
        ax.set_title(f"{cohort}\nn={len(wide)} paired", fontsize=8)
        ax.grid(axis="y", color="#EEEEEE", lw=0.5)
        open_axes(ax)
        if i == 0:
            ax.set_ylabel("AC-like fraction of malignant cells")
            panel_label(ax, "a")
        else:
            ax.set_ylabel("")

    # b：三位置队列保留患者缺失结构。
    ax = fig.add_subplot(gs[1, :2])
    frame = ac[ac["cohort"] == "GSE274987"]
    wide = frame.pivot_table(index="patient", columns="region", values="fraction_of_malignant", aggfunc="mean")
    region_order = ["Core", "Edge", "Peripheral"]
    for _, row in wide.iterrows():
        values = row.reindex(region_order).to_numpy(float)
        valid = np.isfinite(values)
        ax.plot(np.arange(3)[valid], values[valid], color="#B4B4B4", lw=0.8, alpha=0.9)
        for j, value in enumerate(values):
            if np.isfinite(value):
                ax.scatter(j, value, s=19, color=REGION_COLORS[region_order[j]], zorder=2)
    medians = [wide.get(region, pd.Series(dtype=float)).median() for region in region_order]
    ax.plot(range(3), medians, color="#222222", lw=2.3, marker="D", markersize=4)
    ax.set_xticks(range(3), region_order)
    ax.set_ylabel("AC-like fraction of malignant cells")
    ax.set_title("GSE274987 regional trajectories", loc="left", pad=8)
    ax.set_ylim(-0.04, 1.04)
    ax.grid(axis="y", color="#EEEEEE", lw=0.5)
    open_axes(ax)
    panel_label(ax, "b")

    # c：区域配对效应的患者点分布。
    ax = fig.add_subplot(gs[1, 2])
    labels = {
        "GSE273274": "273274 T-P",
        "GSE276177": "276177 T-P",
        "GSE286419": "286419 T-P",
        "GSE274987_core_peripheral": "274987 C-P",
        "GSE274987_core_edge": "274987 C-E",
    }
    rows = tests.set_index("comparison").loc[list(labels)].reset_index()
    yy = np.arange(len(rows))[::-1]
    rng = np.random.default_rng(17)
    for yv, row in zip(yy, rows.itertuples()):
        diffs = np.asarray(json.loads(row.patient_differences), dtype=float)
        jitter = rng.normal(0, 0.055, size=len(diffs))
        ax.scatter(diffs, np.repeat(yv, len(diffs)) + jitter, s=18, color=COHORT_COLORS[row.cohort], alpha=0.85)
        ax.scatter(row.median_difference, yv, s=52, marker="D", color="#111111", edgecolor="white", linewidth=0.7)
    ax.axvline(0, color="#777777", lw=0.8, ls="--")
    ax.set_yticks(yy, [labels[x] for x in rows["comparison"]])
    ax.set_xlabel("Region A minus region B")
    ax.set_title("Paired effects", loc="left", pad=8)
    open_axes(ax)
    panel_label(ax, "c")
    save_figure(fig, "Figure3")


def figure4() -> None:
    regional = pd.read_csv(DATA / "regional_communication_tests.csv")
    meta = pd.read_csv(DATA / "meta_ligand_results.csv")
    fig = plt.figure(figsize=(7.2, 5.7))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.15, 0.9], width_ratios=[1.25, 0.85], hspace=0.68, wspace=0.45)

    comparisons = [
        "GSE273274",
        "GSE276177",
        "GSE286419",
        "GSE274987_core_peripheral",
        "GSE274987_core_edge",
    ]
    pairs = ["TGFB2-TGFBR1", "TGFB2-TGFBR2", "TGFB1-TGFBR1", "ANXA1-FPR1", "SPP1-CD44", "CSF1-CSF1R", "CCL2-CCR2"]
    comp_labels = ["273274 T-P", "276177 T-P", "286419 T-P", "274987 C-P", "274987 C-E"]

    # a：所有区域配对通讯效应矩阵。
    ax = fig.add_subplot(gs[0, :])
    max_abs = max(abs(regional["median_difference"].min()), abs(regional["median_difference"].max()))
    norm = mpl.colors.Normalize(-max_abs, max_abs)
    for i, comparison in enumerate(comparisons):
        for j, pair in enumerate(pairs):
            row = regional[(regional["comparison"] == comparison) & (regional["pair"] == pair)].iloc[0]
            size = 28 + 180 * abs(row["median_difference"]) / max_abs
            color = mpl.colormaps["RdBu_r"](norm(row["median_difference"]))
            significant = pd.notna(row["q_value"]) and row["q_value"] < 0.05
            ax.scatter(j, len(comparisons) - 1 - i, s=size, color=color, edgecolor="black" if significant else "white", linewidth=1.0 if significant else 0.6)
            ax.text(j, len(comparisons) - 1 - i - 0.28, f"n={int(row['n_patients'])}", ha="center", va="top", fontsize=6.5, color="#555555")
    ax.set_xticks(range(len(pairs)), [p.replace("-", "–") for p in pairs], rotation=30, ha="right")
    ax.set_yticks(range(len(comparisons)), comp_labels[::-1])
    ax.set_xlabel("Ligand–receptor pair", labelpad=1)
    ax.set_ylabel("Within-patient comparison")
    ax.set_title("Regional change in expression co-availability", loc="left", pad=8)
    ax.text(0.0, 1.01, "No exact test could attain raw P<0.05 at the available paired sample sizes", transform=ax.transAxes, fontsize=7, color="#555555")
    ax.set_xlim(-0.6, len(pairs) - 0.4)
    ax.set_ylim(-0.65, len(comparisons) - 0.35)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Median region A − B", fontsize=7)
    panel_label(ax, "a")

    # b：TGFB2 两个受体的患者级效应。
    ax = fig.add_subplot(gs[1, 0])
    tgfb = regional[regional["pair"].isin(["TGFB2-TGFBR1", "TGFB2-TGFBR2"])].copy()
    rows = []
    for comparison in comparisons:
        for pair in ["TGFB2-TGFBR1", "TGFB2-TGFBR2"]:
            rows.append(tgfb[(tgfb["comparison"] == comparison) & (tgfb["pair"] == pair)].iloc[0])
    yy = np.arange(len(rows))[::-1]
    rng = np.random.default_rng(23)
    for yv, row in zip(yy, rows):
        diffs = np.asarray(json.loads(row["patient_differences"]), dtype=float)
        ax.scatter(diffs, np.repeat(yv, len(diffs)) + rng.normal(0, 0.045, len(diffs)), s=15, color=COHORT_COLORS[row["cohort"]], alpha=0.85)
        ax.scatter(row["median_difference"], yv, s=42, marker="D", color="#111111", edgecolor="white", linewidth=0.7)
    ax.axvline(0, color="#777777", lw=0.8, ls="--")
    row_labels = []
    for comparison in comparisons:
        short = comp_labels[comparisons.index(comparison)]
        row_labels.extend([f"{short} R1", f"{short} R2"])
    ax.set_yticks(yy, row_labels)
    ax.set_xlabel("Regional difference in co-availability")
    ax.set_title("TGFB2 receptor-specific effects", loc="left", pad=8)
    open_axes(ax)
    panel_label(ax, "b")

    # c：内在表达证据与区域通讯证据直接比较。
    ax = fig.add_subplot(gs[1, 1])
    temp = regional.copy()
    temp["gene"] = temp["pair"].str.split("-").str[0]
    best_q = temp.groupby("gene", observed=True)["q_value"].min()
    for row in meta.itertuples():
        if row.gene not in best_q.index:
            continue
        x = -np.log10(max(row.q_value, 1e-12))
        y = -np.log10(max(float(best_q.loc[row.gene]), 1e-12))
        robust = row.loo_q_value < 0.05
        ax.scatter(x, y, s=58 if robust else 38, color=POS if robust else "#777777", edgecolor="white", linewidth=0.7)
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
    ax.set_title("Regional evidence remains unresolved", loc="left", pad=8)
    open_axes(ax)
    panel_label(ax, "c")
    save_figure(fig, "Figure4")


def supplementary_figure_s1() -> None:
    expression = pd.read_csv(DATA / "patient_state_expression.csv")
    expression = expression[expression["state"].isin(["AC-like", "MES-like", "OPC-like", "NPC-like", "Myeloid"])].copy()
    state_order = ["AC-like", "MES-like", "OPC-like", "NPC-like", "Myeloid"]
    region_short = {"Tumor": "T", "Core": "C", "Edge": "E", "Peritumoral": "P", "Peripheral": "P"}
    samples = []
    for cohort in COHORT_ORDER:
        frame = expression[expression["cohort"] == cohort]
        for patient, region in frame[["patient", "region"]].drop_duplicates().itertuples(index=False):
            short_patient = str(patient).replace("GSE273274_", "").replace("GSE274987_", "")
            samples.append((cohort, patient, region, f"{cohort} | {short_patient} | {region_short.get(region, region)}"))

    fig, ax = plt.subplots(figsize=(6.5, 10.2))
    row_lookup = {(cohort, patient, region): i for i, (cohort, patient, region, _) in enumerate(samples)}
    for row in expression.itertuples():
        x = state_order.index(row.state)
        y = row_lookup[(row.cohort, row.patient, row.region)]
        below = row.n_cells < 20
        ax.scatter(
            x,
            y,
            s=10 + 15 * np.log10(max(row.n_cells, 1)),
            facecolor="white" if below else COHORT_COLORS[row.cohort],
            edgecolor=COHORT_COLORS[row.cohort],
            linewidth=0.8,
        )
    boundaries = []
    last = None
    for i, (cohort, _, _, _) in enumerate(samples):
        if last is not None and cohort != last:
            boundaries.append(i - 0.5)
        last = cohort
    for boundary in boundaries:
        ax.axhline(boundary, color="#D8D8D8", lw=0.7)
    ax.set_xticks(range(len(state_order)), ["AC", "MES", "OPC", "NPC", "Myeloid"])
    ax.set_yticks(range(len(samples)), [item[3] for item in samples])
    ax.invert_yaxis()
    ax.set_xlabel("Cell state")
    ax.set_ylabel("")
    ax.set_title("Patient-region cell counts and predefined eligibility", loc="left", pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, labelsize=6)
    panel_label(ax, "a")
    fig.text(0.5, 0.01, "Filled: ≥20 cells; open: <20 cells. Point area scales with log10(cell count).", ha="center", fontsize=7)
    save_figure(fig, "SupplementaryFigureS1")


def supplementary_figure_s2() -> None:
    expression = pd.read_csv(DATA / "patient_state_expression.csv")
    genes = ["TGFB2", "CSF1", "ANXA1", "CCL2", "TGFB1", "SPP1"]
    rows = []
    for cohort in COHORT_ORDER:
        target = "Core" if cohort == "GSE274987" else "Tumor"
        frame = expression[(expression["cohort"] == cohort) & (expression["region"] == target)]
        for state in ["AC-like", "MES-like", "OPC-like", "NPC-like"]:
            sf = frame[(frame["state"] == state) & (frame["n_cells"] >= 20)]
            row = {"label": f"{cohort} | {state}"}
            for gene in genes:
                row[gene] = sf[f"mean_{gene}"].median() if not sf.empty else np.nan
            rows.append(row)
    matrix = pd.DataFrame(rows).set_index("label")
    matrix = matrix.apply(lambda col: (col - col.mean()) / col.std(ddof=0), axis=0)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    sns.heatmap(matrix, cmap="vlag", center=0, linewidths=0.35, linecolor="white", cbar_kws={"label": "Column z score"}, ax=ax)
    ax.set_xlabel("Ligand")
    ax.set_ylabel("")
    ax.set_title("Median patient-level ligand expression by malignant state", loc="left", pad=8)
    panel_label(ax, "a")
    save_figure(fig, "SupplementaryFigureS2")


def supplementary_figure_s3() -> None:
    """显示统一 marker 重注释、GSE162631 和谱系分数校正结果。"""
    patient = pd.read_csv(DATA / "uniform_annotation_patient_contrasts.csv")
    uniform = pd.read_csv(DATA / "uniform_annotation_cohort_results.csv")
    primary = pd.read_csv(DATA / "ac_vs_other_ligand_contrasts.csv")
    primary = primary[primary["gene"] == "TGFB2"][["cohort", "median_difference"]].rename(columns={"median_difference": "primary_difference"})
    merged = uniform.merge(primary, on="cohort", how="left")
    summary = json.loads((DATA / "uniform_annotation_summary.json").read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.9, 1.0, 1.15], wspace=0.55)

    # a：原标签和统一标签的队列效应对照。
    ax = fig.add_subplot(gs[0, 0])
    order = ["GSE162631", "GSE211376", "GSE273274", "GSE274987", "GSE276177", "GSE286419"]
    merged = merged.set_index("cohort").loc[order].reset_index()
    yy = np.arange(len(merged))[::-1]
    for yv, row in zip(yy, merged.itertuples()):
        if pd.notna(row.primary_difference):
            ax.plot([row.primary_difference, row.median_difference], [yv, yv], color="#B8B8B8", lw=1.0)
            ax.scatter(row.primary_difference, yv, s=32, facecolor="white", edgecolor=COHORT_COLORS.get(row.cohort, "#555555"), linewidth=1.0)
        ax.scatter(row.median_difference, yv, s=38, color=COHORT_COLORS.get(row.cohort, "#555555"), edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="#777777", lw=0.8, ls="--")
    ax.set_yticks(yy, merged["cohort"])
    ax.set_xlabel("Median TGFB2 difference")
    ax.set_title("Annotation sensitivity", loc="left", pad=8)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#555555", label="Primary labels"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", markeredgecolor="white", label="Uniform markers"),
        ],
        loc="upper right",
        fontsize=6.5,
    )
    open_axes(ax)
    panel_label(ax, "a")

    # b：统一标签下每位患者的差值。
    ax = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(41)
    for i, cohort in enumerate(order):
        values = patient.loc[patient["cohort"] == cohort, "tgfb2_difference"].to_numpy(float)
        ax.scatter(np.repeat(i, len(values)) + rng.normal(0, 0.055, len(values)), values, s=19, color=COHORT_COLORS.get(cohort, "#555555"), alpha=0.85)
        if len(values):
            ax.scatter(i, np.median(values), s=48, marker="D", color="#111111", edgecolor="white", linewidth=0.7)
    ax.axhline(0, color="#777777", lw=0.8, ls="--")
    ax.set_xticks(range(len(order)), [item.replace("GSE", "") for item in order], rotation=45, ha="right")
    ax.set_ylabel("Uniform AC-like minus other TGFB2")
    ax.set_xlabel("Public cohort")
    ax.set_title("Patient-level effects", loc="left", pad=8)
    open_axes(ax)
    panel_label(ax, "b")

    # c：TGFB2 与 AC marker 强度的关系和校正后估计。
    ax = fig.add_subplot(gs[0, 2])
    for cohort, frame in patient.groupby("cohort", observed=True):
        ax.scatter(frame["ac_score_difference"], frame["tgfb2_difference"], s=25, color=COHORT_COLORS.get(cohort, "#555555"), alpha=0.85, label=cohort)
    x = patient["ac_score_difference"].to_numpy(float)
    y = patient["tgfb2_difference"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, intercept + slope * xx, color="#333333", lw=1.0)
    lineage = summary["lineage_adjustment"]
    ax.text(
        0.03,
        0.97,
        f"Spearman ρ={lineage['spearman_rho']:.2f}\nHC3 adjusted mean={lineage['grand_mean_at_average_ac_score']:.3f}\nP={lineage['hc3_p_value']:.2g}",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5),
    )
    ax.axhline(0, color="#B0B0B0", lw=0.7, ls="--")
    ax.set_xlabel("AC marker-score difference")
    ax.set_ylabel("TGFB2 expression difference")
    ax.set_title("Lineage-program sensitivity", loc="left", pad=8)
    open_axes(ax)
    panel_label(ax, "c")
    save_figure(fig, "SupplementaryFigureS3")


def main() -> None:
    require_files(
        [
            "cohort_manifest.csv",
            "patient_state_expression.csv",
            "patient_ac_fraction.csv",
            "ac_vs_other_ligand_contrasts.csv",
            "meta_ligand_results.csv",
            "regional_communication_tests.csv",
            "regional_ac_fraction_tests.csv",
            "uniform_annotation_patient_contrasts.csv",
            "uniform_annotation_cohort_results.csv",
            "uniform_annotation_summary.json",
        ]
    )
    configure_style()
    figure1()
    figure2()
    figure3()
    figure4()
    supplementary_figure_s1()
    supplementary_figure_s2()
    supplementary_figure_s3()
    print(f"Saved figures to {OUT}")


if __name__ == "__main__":
    main()
