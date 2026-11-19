#!/usr/bin/env python3
# 目的：比较 CPTAC-GBM 清洁蛋白矩阵中肿瘤与正常脑的配体信号，检验同批次差值和肿瘤内相关。
# 输入：--input-dir 中的 proteome_mssm_per_gene_clean.v4.0.tsv.gz（gene、refseq_id、样本列）
# 与 proteome_tmt_design.v4.0.tsv.gz（case_id、tumor_normal、tmt_plex）。使用原 NumPy/SciPy 随机检验。
# 输出：默认在 external_validation/08_CPTAC蛋白验证 写 01_蛋白覆盖.csv、02_配体肿瘤与正常比较.csv、
# 03_肿瘤蛋白评分.csv、04_TGFB2蛋白相关.csv、05_TGFB2同批次对比.csv、06_TGFB2样本值.csv、07_分析摘要.json、
# 08_Figure6_CPTAC蛋白验证.pdf 和 08_Figure6_CPTAC蛋白验证.png。复算表沿用输入本身的病例编码。
# 运行方式：python 08_CPTAC蛋白验证.py --input-dir Proteome目录 [--output-dir 复算目录]
# 可选 --matrix-url、--design-url 分别接受原表的 HTTPS 直链；本地缺表时才使用 raw/ 内的来源缓存。

import argparse
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, permutation_test, spearmanr

spatial = import_module("00a_空间分析")
files = import_module("00a_项目路径")


def main():
    """核对清洁矩阵与 TMT 设计，写出蛋白比较和图件；样本不能匹配或没有成对 plex 时停止，无返回值。"""
    external_dir = Path(__file__).resolve().parents[1] / "external_validation"
    parser = argparse.ArgumentParser(description="CPTAC-GBM 蛋白组验证")
    parser.add_argument("--input-dir", type=Path, default=external_dir / "raw/cptac_gbm/Proteome")
    parser.add_argument("--output-dir", type=Path, default=external_dir / Path(__file__).stem)
    parser.add_argument("--matrix-url", help="清洁蛋白矩阵的 HTTPS 直链")
    parser.add_argument("--design-url", help="TMT 设计原表的 HTTPS 直链")
    args = parser.parse_args()

    # 基因行转为样本行，refseq_id 是注释列。重复病例或基因会使组间比较重复计数。
    matrix_path = spatial.cptac_input(
        args.input_dir / "proteome_mssm_per_gene_clean.v4.0.tsv.gz", args.matrix_url, args.output_dir / "raw",
        ["gene", "refseq_id"]
    )
    protein_table = pd.read_csv(matrix_path, sep="\t")
    expression_table = protein_table.set_index("gene", verify_integrity=True).drop(columns="refseq_id").transpose()
    expression_table.index.name = "case_id"
    design_path = spatial.cptac_input(
        args.input_dir / "proteome_tmt_design.v4.0.tsv.gz", args.design_url, args.output_dir / "raw",
        ["case_id", "tumor_normal", "tmt_plex"]
    )
    design_table = pd.read_csv(design_path, sep="\t", dtype={"case_id": str})
    design_table = design_table.set_index("case_id", verify_integrity=True)

    # 以清洁矩阵为分析样本。设计有记录而质控后无蛋白的病例只记入摘要，不插补。
    if not expression_table.index.isin(design_table.index).all():
        raise ValueError("清洁蛋白矩阵含 TMT 设计表中没有的病例")
    excluded = design_table.index.difference(expression_table.index).tolist()
    design_table = design_table.loc[expression_table.index]
    if not design_table["tumor_normal"].isin(["tumor", "normal"]).all():
        raise ValueError("tumor_normal 含未识别的组织分组")
    is_tumor = design_table["tumor_normal"] == "tumor"
    is_normal = design_table["tumor_normal"] == "normal"

    # 覆盖表包括六种配体、受体、SMAD 和评分标志物；缺测基因仍保留一行。
    marker_sets = {
        "astrocytic_protein_score": ["GFAP", "AQP4", "S100B", "SLC1A3", "ALDH1L1", "GLUL", "HOPX"],
        "myeloid_protein_score": ["AIF1", "CD163", "CD14", "CSF1R", "TYROBP", "FCER1G", "LST1"],
        "tgfbr_protein_score": spatial.receptors,
    }
    coverage_genes = set(spatial.ligands + spatial.receptors + ["SMAD2", "SMAD3", "SMAD4"]).union(*marker_sets.values())
    coverage_rows = []
    for gene in sorted(coverage_genes):
        present = gene in expression_table
        values = expression_table[gene] if present else pd.Series(np.nan, index=expression_table.index)
        coverage_rows.append({
            "gene": gene, "present": present, "n_quantified": int(values.notna().sum()),
            "fraction_quantified": values.notna().mean(), "n_tumor_quantified": int(values[is_tumor].notna().sum()),
            "n_normal_quantified": int(values[is_normal].notna().sum()),
        })
    coverage_table = pd.DataFrame(coverage_rows)

    # 固定配体顺序与随机流：每种配体先置换，再按肿瘤、正常次序 bootstrap，图点最后消费随机数。
    rng = np.random.default_rng(20260716)
    comparison_rows = []

    # SciPy 要求统计函数接受 axis，以便一次处理一批置换；每次返回同批肿瘤减正常的中位数差。
    def median_difference(tumor, normal, axis):
        difference = np.median(tumor, axis=axis) - np.median(normal, axis=axis)
        return difference

    for gene in spatial.ligands:
        values = expression_table[gene] if gene in expression_table else pd.Series(np.nan, index=expression_table.index)
        tumor = values[is_tumor].dropna().to_numpy()
        normal = values[is_normal].dropna().to_numpy()
        record = {"gene": gene, "n_tumor": len(tumor), "n_normal": len(normal)}
        comparison_rows.append(record)
        if min(len(tumor), len(normal)) < 3:
            continue
        test = mannwhitneyu(tumor, normal, alternative="two-sided", method="asymptotic")
        permutation = permutation_test(
            (tumor, normal), median_difference, permutation_type="independent", alternative="two-sided",
            n_resamples=100_000, vectorized=True, batch=5_000, rng=rng
        )

        # 各组分别有放回抽样 20,000 次，取中位数差经验分布的 2.5% 和 97.5% 分位数。
        tumor_draws = rng.choice(tumor, size=(20_000, len(tumor)), replace=True)
        normal_draws = rng.choice(normal, size=(20_000, len(normal)), replace=True)
        differences = np.median(tumor_draws, axis=1) - np.median(normal_draws, axis=1)
        ci_low, ci_high = np.quantile(differences, [0.025, 0.975])
        record.update({
            "tumor_median": np.median(tumor), "normal_median": np.median(normal),
            "median_difference": np.median(tumor) - np.median(normal), "median_difference_ci_low": ci_low,
            "median_difference_ci_high": ci_high, "tumor_mean": np.mean(tumor), "normal_mean": np.mean(normal),
            "mannwhitney_u": test.statistic, "mannwhitney_p": test.pvalue, "median_permutation_p": permutation.pvalue,
            "rank_biserial": 2 * test.statistic / (len(tumor) * len(normal)) - 1,
        })

    # 样本不足的配体保留样本数和空统计字段；两种检验分别在配体间做 BH 校正。
    comparison_columns = [
        "gene", "n_tumor", "n_normal", "tumor_median", "normal_median", "median_difference", "median_difference_ci_low",
        "median_difference_ci_high", "tumor_mean", "normal_mean", "mannwhitney_u", "mannwhitney_p",
        "median_permutation_p", "rank_biserial",
    ]
    comparison_table = pd.DataFrame(comparison_rows, columns=comparison_columns)
    for test_name in ["mannwhitney", "median_permutation"]:
        comparison_table[f"{test_name}_q"] = spatial.bh_adjust(comparison_table[f"{test_name}_p"])

    # 肿瘤内至少 80% 病例定量的标志物才能入选；排除零方差列，按总体标准差标准化。
    tumor_table = expression_table.loc[is_tumor]
    score_table = pd.DataFrame(index=tumor_table.index)
    association_rows = []
    for name, markers in marker_sets.items():
        available = [gene for gene in markers if gene in tumor_table and tumor_table[gene].notna().mean() >= 0.80]
        marker_table = tumor_table[available]
        means = marker_table.mean(axis=0, skipna=True)
        deviations = marker_table.std(axis=0, skipna=True, ddof=0)
        used = deviations[deviations > 0].index
        standardized_table = (marker_table[used] - means[used]) / deviations[used]

        # 单个病例需覆盖至少 60% 入选标志物。无可用标志物时整列留空；相关至少需 10 对有效样本。
        minimum = max(1, int(np.ceil(0.60 * len(used))))
        score = standardized_table.mean(axis=1).where(standardized_table.notna().sum(axis=1) >= minimum)
        score_table[name] = score
        valid = tumor_table["TGFB2"].notna() & score.notna()
        record = {
            "score": name, "markers": ";".join(used), "n": int(valid.sum()),
            "spearman_rho": np.nan, "spearman_p": np.nan,
        }
        if valid.sum() >= 10:
            test = spearmanr(tumor_table.loc[valid, "TGFB2"], score[valid])
            record.update(spearman_rho=test.statistic, spearman_p=test.pvalue)
        association_rows.append(record)
    association_table = pd.DataFrame(association_rows)
    association_table["spearman_q"] = spatial.bh_adjust(association_table["spearman_p"])

    # 每个 TMT plex 取本批次肿瘤与正常中位数之差。须有两组，精确翻转单位是批次。
    plex_rows = []
    for plex, group_table in design_table.groupby("tmt_plex", sort=True):
        tumor = expression_table.loc[group_table.index[group_table["tumor_normal"] == "tumor"], "TGFB2"].dropna()
        normal = expression_table.loc[group_table.index[group_table["tumor_normal"] == "normal"], "TGFB2"].dropna()
        if tumor.empty or normal.empty:
            continue
        plex_rows.append({
            "tmt_plex": int(plex), "n_tumor": len(tumor), "n_normal": len(normal), "tumor_median": tumor.median(),
            "normal_median": normal.median(), "median_difference": tumor.median() - normal.median(),
        })
    if not plex_rows:
        raise ValueError("没有同时测得肿瘤和正常脑 TGFB2 的 TMT plex")
    plex_table = pd.DataFrame(plex_rows)
    plex_summary = {
        "n_plexes": len(plex_table), "median_difference": float(plex_table["median_difference"].median()),
        "n_positive": int((plex_table["median_difference"] > 0).sum()),
        "exact_signflip_p": spatial.signflip_p(plex_table["median_difference"]),
    }

    # 与 Fig 6a 对应的每个有效定量值都直接来自清洁矩阵；公开文件中的 P/N 代号需另有对应表。
    sample_table = pd.DataFrame({"group": design_table["tumor_normal"], "tgfb2_log2_ratio": expression_table["TGFB2"]})
    sample_table = sample_table.dropna(subset=["tgfb2_log2_ratio"]).reset_index()
    tgfb2_result = comparison_table.set_index("gene", drop=False).loc["TGFB2"]
    for name in ["tumor", "normal"]:
        if (sample_table["group"] == name).sum() != tgfb2_result[f"n_{name}"]:
            raise ValueError(f"TGFB2 的 {name} 散点数与组间比较不一致")
    group_counts = coverage_table["n_tumor_quantified"] + coverage_table["n_normal_quantified"]
    if not coverage_table["n_quantified"].eq(group_counts).all():
        raise ValueError("蛋白总定量数与两组分项不符")

    files.write_table(coverage_table, args.output_dir / "01_蛋白覆盖.csv", ["gene"])
    files.write_table(comparison_table, args.output_dir / "02_配体肿瘤与正常比较.csv", ["gene"])
    files.write_table(score_table.reset_index(), args.output_dir / "03_肿瘤蛋白评分.csv", ["case_id"])
    files.write_table(association_table, args.output_dir / "04_TGFB2蛋白相关.csv", ["score"])
    files.write_table(plex_table, args.output_dir / "05_TGFB2同批次对比.csv", ["tmt_plex"])
    files.write_table(sample_table, args.output_dir / "06_TGFB2样本值.csv", ["case_id"])

    # 摘要保留质控排除病例及已发表字段，无法估计的统计量在 JSON 中写 null。
    json_association_table = association_table.astype(object).where(association_table.notna(), None)
    summary = {
        "dataset": "CPTAC-GBM PDC000204", "matrix": matrix_path.name, "imputation": "none; clean matrix used",
        "n_tumor": int(is_tumor.sum()), "n_normal": int(is_normal.sum()),
        "design_samples_excluded_after_proteome_qc": excluded,
        "tgfb2": tgfb2_result.astype(object).where(tgfb2_result.notna(), None).to_dict(),
        "tgfb2_plex_matched": plex_summary, "associations": json_association_table.to_dict(orient="records"),
    }
    files.write_json(summary, args.output_dir / "07_分析摘要.json")

    # 组织分组、点数和 plex 阳性数取自实际样本表；散点在每组原顺序上取抖动。
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axis = axes[0, 0]
    groups = []
    for name in ["tumor", "normal"]:
        groups.append(sample_table.loc[sample_table["group"] == name, "tgfb2_log2_ratio"].to_numpy())
    colors = ["#B33A3A", "#2A7181"]
    violin = axis.violinplot(groups, positions=[1, 2], showextrema=False)
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.35)
    for i, (values, color) in enumerate(zip(groups, colors), start=1):
        axis.scatter(i + rng.normal(0, 0.045, len(values)), values, s=18, alpha=0.75, color=color, linewidth=0)
        axis.hlines(np.median(values), i - 0.20, i + 0.20, color="black")
    axis.set_xticks([1, 2], [f"GBM tumor\nn={len(groups[0])}", f"normal brain\nn={len(groups[1])}"])
    axis.set_ylabel("TGFB2 protein log2 ratio")
    axis.set_title("a  TGFB2 protein in CPTAC-GBM", loc="left", fontweight="bold")
    axis.text(
        0.04, 0.96, f"{plex_summary['n_positive']}/{len(plex_table)} matched plexes positive; "
        f"p={plex_summary['exact_signflip_p']:.4g}", transform=axis.transAxes, va="top"
    )

    # 两个肿瘤内相关面板标注各自的 Spearman rho 和 BH q；直线仅用于展示。
    score_panels = [(axes[0, 1], "astrocytic_protein_score", "b"), (axes[1, 0], "myeloid_protein_score", "c")]
    for axis, name, letter in score_panels:
        values = tumor_table["TGFB2"]
        score = score_table[name]
        valid = values.notna() & score.notna()
        axis.scatter(values[valid], score[valid], s=24, alpha=0.72, color="#287C78")
        if valid.sum() > 2 and values[valid].nunique() > 1:
            coefficients = np.polyfit(values[valid], score[valid], 1)
            grid = np.linspace(values[valid].min(), values[valid].max(), 100)
            axis.plot(grid, np.polyval(coefficients, grid), color="#B33A3A", linewidth=1.5)
        stats = association_table.set_index("score").loc[name]
        label = f"rho={stats['spearman_rho']:.2f}, q={stats['spearman_q']:.3g}"
        axis.text(0.04, 0.95, label, transform=axis.transAxes, va="top")
        axis.set(xlabel="TGFB2 protein log2 ratio", ylabel=name.replace("_", " "))
        axis.set_title(f"{letter}  Tumor protein association", loc="left", fontweight="bold")

    # 逐配体展示中位差与 bootstrap 区间。缺测配体保留轴标签，点和线自然留空。
    axis = axes[1, 1]
    positions = np.arange(len(comparison_table))
    axis.hlines(
        positions, comparison_table["median_difference_ci_low"], comparison_table["median_difference_ci_high"],
        color="#8D99AE"
    )
    for column in ["median_difference_ci_low", "median_difference_ci_high"]:
        axis.plot(comparison_table[column], positions, marker="|", color="#8D99AE", linestyle="none", markersize=6)
    axis.scatter(comparison_table["median_difference"], positions, color="#30343B", s=24)
    is_tgfb2 = comparison_table["gene"] == "TGFB2"
    axis.scatter(comparison_table.loc[is_tgfb2, "median_difference"], positions[is_tgfb2], color="#B33A3A", s=52)
    axis.axvline(0, color="#777777", linestyle="--", linewidth=1)
    axis.set_yticks(positions, comparison_table["gene"])
    axis.set_xlabel("Tumor minus normal median log2 ratio (95% bootstrap CI)")
    axis.set_title("d  Fixed ligand panel", loc="left", fontweight="bold")

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Protein-level validation in CPTAC-GBM", x=0.07, ha="left", fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    files.save_figure(figure, args.output_dir / "08_Figure6_CPTAC蛋白验证.pdf")
    plt.close(figure)
    print(f"肿瘤 {int(is_tumor.sum())} 例，正常参照 {int(is_normal.sum())} 例，同批次比较 {len(plex_table)} 个")


if __name__ == "__main__":
    main()
