#!/usr/bin/env python3
"""用患者符号翻转的穷举零分布校准 signed Stouffer 结果。"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon

from stage_public_paths import project_root, submission_package


ROOT = project_root()
PACKAGE = submission_package(ROOT)
DATA = PACKAGE / "data"
FIGURES = PACKAGE / "figures"
COHORT_ORDER = ["GSE211376", "GSE273274", "GSE274987", "GSE286419"]
GENE_ORDER = ["TGFB2", "CSF1", "ANXA1", "CCL2", "TGFB1", "SPP1"]
COLORS = {"TGFB2": "#B43C52", "CSF1": "#C49A2C", "ANXA1": "#4C78A8"}


def bh_adjust(values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg 校正。"""
    values = values.astype(float)
    order = np.argsort(values.to_numpy())
    ranked = values.to_numpy()[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    return pd.Series(restored, index=values.index)


def signed_wilcoxon_z(values: np.ndarray) -> float:
    """将双侧 Wilcoxon P 值和中位数方向转换为 signed z。"""
    nonzero = np.asarray(values, dtype=float)
    nonzero = nonzero[~np.isclose(nonzero, 0.0)]
    if len(nonzero) == 0:
        return 0.0
    p_value = float(wilcoxon(nonzero, alternative="two-sided", method="exact").pvalue)
    return float(norm.isf(np.clip(p_value, 1e-15, 1.0) / 2) * np.sign(np.median(nonzero)))


def signflip_z_distribution(values: np.ndarray) -> np.ndarray:
    """枚举单队列所有患者符号翻转后的 signed z。"""
    output = []
    for signs in itertools.product([-1.0, 1.0], repeat=len(values)):
        output.append(signed_wilcoxon_z(values * np.asarray(signs)))
    return np.asarray(output, dtype=float)


def combine_distribution(
    patient_values: list[np.ndarray],
    cohort_distributions: list[np.ndarray],
    included: list[int],
) -> tuple[float, float, np.ndarray]:
    """组合指定队列并返回观测 z、精确 P 和穷举零分布。"""
    weights = np.asarray([np.sqrt(len(patient_values[index])) for index in included], dtype=float)
    observed_components = np.asarray([signed_wilcoxon_z(patient_values[index]) for index in included])
    observed = float(np.dot(weights, observed_components) / np.sqrt(np.dot(weights, weights)))
    combined = np.asarray([0.0])
    for weight, index in zip(weights, included):
        combined = (combined[:, None] + weight * cohort_distributions[index][None, :]).reshape(-1)
    combined /= np.sqrt(np.dot(weights, weights))
    exact_p = float(np.mean(np.abs(combined) >= abs(observed) - 1e-12))
    return observed, exact_p, combined


def parse_patient_values(contrasts: pd.DataFrame, gene: str) -> list[np.ndarray]:
    """从 stage45 结果表读取每个队列的患者差值。"""
    arrays = []
    for cohort in COHORT_ORDER:
        row = contrasts[(contrasts["gene"] == gene) & (contrasts["cohort"] == cohort)]
        if len(row) != 1:
            raise ValueError(f"{gene}/{cohort} 结果行数量异常: {len(row)}")
        records = json.loads(row.iloc[0]["patient_values"])
        arrays.append(np.asarray([record["difference"] for record in records], dtype=float))
    return arrays


def calculate() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """计算六个配体的完整与删一队列精确校准。"""
    contrast_path = DATA / "ac_vs_other_ligand_contrasts.csv"
    normal_path = DATA / "meta_ligand_results.csv"
    if not contrast_path.is_file() or not normal_path.is_file():
        raise FileNotFoundError("缺少 stage45 meta 输入表")
    contrasts = pd.read_csv(contrast_path)
    normal = pd.read_csv(normal_path).set_index("gene")
    rows: list[dict[str, object]] = []
    tail_rows: list[dict[str, object]] = []
    for gene in GENE_ORDER:
        patient_values = parse_patient_values(contrasts, gene)
        cohort_distributions = [signflip_z_distribution(values) for values in patient_values]
        observed_z, exact_p, full_distribution = combine_distribution(
            patient_values, cohort_distributions, list(range(len(COHORT_ORDER)))
        )
        loo = []
        for omitted_index, omitted in enumerate(COHORT_ORDER):
            included = [index for index in range(len(COHORT_ORDER)) if index != omitted_index]
            loo_z, loo_p, _ = combine_distribution(patient_values, cohort_distributions, included)
            loo.append({"omitted": omitted, "meta_z": loo_z, "exact_p_value": loo_p})
        rows.append(
            {
                "gene": gene,
                "n_cohorts": len(COHORT_ORDER),
                "n_patients": sum(len(values) for values in patient_values),
                "observed_meta_z": observed_z,
                "normal_approximation_p_value": float(normal.loc[gene, "p_value"]),
                "normal_approximation_q_value": float(normal.loc[gene, "q_value"]),
                "exact_signflip_p_value": exact_p,
                "normal_loo_max_p_value": float(normal.loc[gene, "loo_max_p_value"]),
                "normal_loo_q_value": float(normal.loc[gene, "loo_q_value"]),
                "exact_loo_max_p_value": max(item["exact_p_value"] for item in loo),
                "exact_loo_results": json.dumps(loo, ensure_ascii=False),
                "full_signflip_patterns": int(len(full_distribution)),
            }
        )
        if gene in ["TGFB2", "CSF1"]:
            maximum = float(np.max(np.abs(full_distribution)))
            for threshold in np.linspace(0, maximum, 121):
                tail_rows.append(
                    {
                        "gene": gene,
                        "absolute_z_threshold": threshold,
                        "exact_tail_probability": float(np.mean(np.abs(full_distribution) >= threshold - 1e-12)),
                        "observed_absolute_z": abs(observed_z),
                        "observed_exact_p_value": exact_p,
                    }
                )
    results = pd.DataFrame(rows)
    results["exact_signflip_q_value"] = bh_adjust(results["exact_signflip_p_value"])
    results["exact_loo_q_value"] = bh_adjust(results["exact_loo_max_p_value"])
    results = results.sort_values("exact_signflip_q_value").reset_index(drop=True)
    summary = {
        "enumeration_is_exhaustive": True,
        "full_signflip_patterns_per_ligand": int(results["full_signflip_patterns"].iloc[0]),
        "tgfb2_exact_p_value": float(results.loc[results["gene"] == "TGFB2", "exact_signflip_p_value"].iloc[0]),
        "tgfb2_exact_loo_q_value": float(results.loc[results["gene"] == "TGFB2", "exact_loo_q_value"].iloc[0]),
        "genes_passing_exact_primary_and_loo_q": results.loc[
            (results["exact_signflip_q_value"] < 0.05) & (results["exact_loo_q_value"] < 0.05), "gene"
        ].tolist(),
        "interpretation": "Sensitivity calibration of the signed Stouffer statistic; primary claims remain based on the primary normal approximation.",
    }
    return results, pd.DataFrame(tail_rows), summary


def configure_style() -> None:
    """设置期刊补图样式。"""
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
            "axes.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    """添加面板标签。"""
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def calibration_slope(ax: plt.Axes, results: pd.DataFrame, normal_col: str, exact_col: str, title: str) -> None:
    """用配对斜率图比较正态近似与精确校准。"""
    ordered = results.set_index("gene").loc[GENE_ORDER].reset_index()
    for row in ordered.itertuples(index=False):
        normal_value = -np.log10(max(float(getattr(row, normal_col)), 1e-15))
        exact_value = -np.log10(max(float(getattr(row, exact_col)), 1e-15))
        color = COLORS.get(row.gene, "#8A8A8A")
        ax.plot([0, 1], [normal_value, exact_value], color=color, lw=1.1, alpha=0.8)
        ax.scatter([0, 1], [normal_value, exact_value], color=color, s=32, edgecolor="white", linewidth=0.6, zorder=2)
        if row.gene in COLORS:
            ax.text(1.04, exact_value, row.gene, va="center", fontsize=6.7, color=color)
    ax.axhline(-np.log10(0.05), color="#777777", ls="--", lw=0.8)
    ax.set_xlim(-0.18, 1.48)
    ax.set_xticks([0, 1], ["Normal\napproximation", "Exact\nsign flip"])
    ax.set_ylabel("Evidence, -log10(q)")
    ax.set_title(title, loc="left", pad=8)
    ax.spines[["top", "right"]].set_visible(False)


def make_figure() -> None:
    """从写出的 CSV 回读并绘制精确校准补图。"""
    result_path = DATA / "exact_meta_calibration.csv"
    tail_path = DATA / "exact_meta_null_tail.csv"
    if not result_path.is_file() or not tail_path.is_file():
        raise FileNotFoundError("缺少精确 meta 校准结果表")
    results = pd.read_csv(result_path)
    tail = pd.read_csv(tail_path)
    configure_style()
    fig = plt.figure(figsize=(7.3, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9], hspace=0.5, wspace=0.5)

    ax = fig.add_subplot(gs[0, 0])
    calibration_slope(ax, results, "normal_approximation_q_value", "exact_signflip_q_value", "Full meta-analysis calibration")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    calibration_slope(ax, results, "normal_loo_q_value", "exact_loo_q_value", "Leave-one-cohort-out calibration")
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, :])
    for gene in ["TGFB2", "CSF1"]:
        frame = tail[tail["gene"] == gene]
        color = COLORS[gene]
        ax.plot(frame["absolute_z_threshold"], frame["exact_tail_probability"], color=color, lw=1.7, label=gene)
        observed_z = float(frame["observed_absolute_z"].iloc[0])
        observed_p = float(frame["observed_exact_p_value"].iloc[0])
        ax.scatter(observed_z, observed_p, color=color, s=42, edgecolor="white", linewidth=0.7, zorder=3)
        ax.vlines(observed_z, 1e-6, observed_p, color=color, ls=":", lw=0.9)
    ax.axhline(0.05, color="#777777", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.1)
    ax.set_xlabel("Absolute signed Stouffer z")
    ax.set_ylabel("Exact sign-flip tail probability")
    ax.set_title("Exhaustive patient-level sign-flip null distribution", loc="left", pad=8)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    panel_label(ax, "c")

    fig.suptitle("Meta-analysis conclusions are calibrated against discrete small-cohort tests", x=0.07, ha="left", fontsize=11, fontweight="bold")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "SupplementaryFigureS5.png", dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(FIGURES / "SupplementaryFigureS5.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """写出精确校准表、零分布曲线和补图。"""
    DATA.mkdir(parents=True, exist_ok=True)
    results, tail, summary = calculate()
    results.to_csv(DATA / "exact_meta_calibration.csv", index=False)
    tail.to_csv(DATA / "exact_meta_null_tail.csv", index=False)
    (DATA / "exact_meta_calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_figure()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
