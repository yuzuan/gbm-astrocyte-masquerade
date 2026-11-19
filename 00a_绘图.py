# ======================================================================
# 00a_绘图.py —— GBM 星形胶质细胞研究公共绘图库
# 适用范围为仓库根目录早期分析的主图和图形摘要。
#
# 目的：
# 让各图读取同一批队列结果、使用同一分母，并从单页 PDF 生成 300 dpi PNG。
# 修改共享读表或保存方式时，所有调用该函数的图都会采用同一改动。
# 阅读具体绘图脚本时，可以在遇到调用后再来查看相应函数。
# 在编辑器里搜索调用处的函数名，支持跳转的编辑器也可用 Ctrl 或 Cmd 加单击。
# 请按函数名定位定义。

import json
import os
import re
import subprocess
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

# 以共享文件所在目录为根，测试副本也按相同布局寻找输入和输出。
project_dir = Path(__file__).resolve().parent
source_dir = project_dir / "01_图件来源"

# 对应旧图的英文字体。PDF 嵌入可选取的字形，便于在成品中核对各面板文字。
mpl.rcParams.update(
    {"font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
)


def analysis_path(archive_name, required=True):
    """按已发文件名寻找新、旧产物。必需文件缺失时报出两条路径；可选文件缺失时由调用处使用已公开图值。"""
    current_names = {
        "noncircular_cnv_validation.json": "01_snGBM免疫参照CNV验证/01_snGBM免疫参照CNV结果.json",
        "gse211376_marker_noncircular.json": "03_GSE211376标志基因定义星形细胞/01_GSE211376标志基因定义星形细胞CNV结果.json",
        "gse174554_cluster.npz": "04_GSE174554聚类与标志基因/01_GSE174554细胞聚类.npz",
        "gse174554_leiden_cnv.json": "05_GSE174554聚类CNV验证/02_GSE174554聚类CNV结果.json",
        "communication_analysis_results.json": "07_scGBM细胞通讯与混合校正/01_通讯得分与混合校正.json",
        "liana_validation.json": "08_snGBM配体通讯排名/01_LIANA配体排名与方法对照.json",
        "gse211376_communication_decontam.json": "09_GSE211376发送细胞筛选/01_筛选前后通讯得分.json",
        "gse174554_communication_decontam.json": "10_GSE174554发送细胞筛选/01_筛选前后通讯得分.json",
        "fig6_concordance_gse211376_vs_scgbm.csv": "12_GSE211376亚型表达与排名/02_跨队列AC排名对照.csv",
    }

    # 已有的新结果即为读取对象。损坏、权限或字段错误会在读取处报出。
    current_path = project_dir / "03_result" / current_names[archive_name]
    if current_path.exists() or current_path.is_symlink():
        return current_path

    archive_path = project_dir / "data" / "intermediate" / archive_name
    if not (archive_path.exists() or archive_path.is_symlink()):
        if required:
            raise FileNotFoundError(
                f"请先生成 {current_path.relative_to(project_dir)}，或提供本地原归档 {archive_path.relative_to(project_dir)}"
            )
        warnings.warn(f"上游尚无 {archive_name}，使用旧提交中已公开的作图值", stacklevel=2)
        result = None
        return result
    warnings.warn(f"本次使用 data/intermediate 中的归档 {archive_name}", stacklevel=2)
    return archive_path


def read_analysis_json(archive_name, required=True):
    """按新旧路径读取一个上游 JSON，返回保留原字段名的字典；仅可选产物缺失时返回空值。"""
    path = analysis_path(archive_name, required)
    if path is None:
        return path
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    return result


def read_cnv_table(require_references=False):
    """返回按 cohort 排列的各群 CNV 百分比、队列规模及平台。已对接的上游字段优先，缺失群不补零。"""
    cohort_table = pd.read_csv(source_dir / "01_队列规模与星形细胞群CNV阳性率.csv", index_col="cohort")
    reference_table = pd.read_csv(source_dir / "02_原图两队列参照细胞CNV阳性率.csv", index_col="cohort")
    cnv_table_total = cohort_table.join(reference_table, validate="one_to_one")

    # 03 的阈值与目标细胞群由标志基因定义。02 的作者标签分析采用不同阈值，不可混用。
    discovery = read_analysis_json("gse211376_marker_noncircular.json", required=require_references)
    if discovery is not None:
        cnv_table_total.at["GSE211376", "cells"] = discovery["n_cells"]
        cnv_table_total.at["GSE211376", "astro_pct"] = discovery["marker_defined_astrocyte"]["cnv_pos_pct"]
        cnv_table_total.at["GSE211376", "myeloid_pct"] = discovery["reference_FPR"]["myeloid"]
        cnv_table_total.at["GSE211376", "oligo_pct"] = discovery["reference_FPR"]["oligodendrocyte"]
        cnv_table_total.at["GSE211376", "author_ac_pct"] = discovery["author_AClike"]["cnv_pos_pct"]
        cnv_table_total.at["GSE211376", "author_astro_pct"] = discovery["author_astrocyte_only"]["cnv_pos_pct"]

    # 01 提供 snGBM 严格阈值下的总体比例和两个参照群。scGBM 无同口径的根模块，保留原图冻结值。
    nuclear = read_analysis_json("noncircular_cnv_validation.json", required=False)
    if nuclear is not None:
        independent = nuclear["snGBM_independent"]
        cnv_table_total.at["GSE276177+286419", "astro_pct"] = independent["strict_CNVpos_pct"]
        cnv_table_total.at["GSE276177+286419", "myeloid_pct"] = independent["reference_FPR"]["Macrophage"]
        cnv_table_total.at["GSE276177+286419", "oligo_pct"] = independent["reference_FPR"]["Oligodendrocyte"]

    # 05 只取 Leiden 簇及其参照结果。04 的每个 leiden 元素对应一个纳入分析的细胞核。
    leiden = read_analysis_json("gse174554_leiden_cnv.json", required=require_references)
    if leiden is not None:
        cnv_table_total.at["GSE174554", "astro_pct"] = leiden["astrocyte_cluster"]["cnv_pos_pct"]
        cnv_table_total.at["GSE174554", "myeloid_pct"] = leiden["reference_FPR"]["myeloid"]
        cnv_table_total.at["GSE174554", "oligo_pct"] = leiden["reference_FPR"]["oligodendrocyte"]
    cluster_path = analysis_path("gse174554_cluster.npz", required=False)
    if cluster_path is not None:
        with np.load(cluster_path, allow_pickle=False) as clustering:
            cnv_table_total.at["GSE174554", "cells"] = len(clustering["leiden"])

    # 总体百分比都有公开旧图值。Fig2 的参照组需要 03、05 的实际统计；原图的 scGBM 少突一项缺报。
    if not cnv_table_total["astro_pct"].between(0, 100).all():
        raise ValueError("目标群 CNV 比例超界或缺报")
    if require_references:
        for column in reference_table.columns:
            values = cnv_table_total[column]
            missing = (values.index == "GSE162631+273274") & (column == "oligo_pct")
            if not np.array_equal(values.isna(), missing) or not values.dropna().between(0, 100).all():
                raise ValueError(f"参照列 {column} 的缺失分组或百分比有误")
    if not cnv_table_total[["cells", "patients"]].gt(0).all().all():
        raise ValueError("队列细胞数与病人数必须为正")
    return cnv_table_total


def draw_cnv_rates(axis, cnv_table_total, decimals):
    """按队列顺序绘制目标群的 CNV 阳性率，返回实际柱顶标注供 PDF 核对。"""
    positions = np.arange(len(cnv_table_total))
    rates = cnv_table_total["astro_pct"]
    colors = ["#27AE60" if cohort == "GSE211376" else "#C0392B" for cohort in cnv_table_total.index]
    bars = axis.bar(positions, rates, color=colors, width=0.66, zorder=2)
    labels = [f"{rate:.{decimals}f}%" for rate in rates]
    axis.bar_label(bars, labels=labels, padding=2, fontsize=7, fontweight="bold")

    # 虚线仅表示第 95 百分位阈值的名义上尾。各类参照细胞的实际阳性率见 Fig2c。
    nominal_tail = 100 - 95
    axis.axhline(nominal_tail, color="#697780", lw=0.8, ls="--", zorder=3)
    axis.text(
        0.98, 0.10, f"Nominal tail {nominal_tail}%", transform=axis.transAxes, ha="right",
        fontsize=6, bbox={"fc": "white", "ec": "none"}
    )
    axis.set_xticks(positions, [name.replace("+", "\n+") for name in cnv_table_total.index], fontsize=6)
    axis.set_ylim(0, 107)
    axis.set_ylabel("Astrocyte-group CNV+ (%)", fontsize=7.5)
    return labels


def read_decontamination_table(for_abstract=False):
    """返回所需配受体的表。Fig5 取五对的上游变化率；摘要只取 TGFB2–TGFBR2 的筛选前后绝对评分。"""
    pairs = ["TGFB2->TGFBR2"]
    if not for_abstract:
        pairs.extend(["TGFB2->TGFBR1", "CSF1->CSF1R", "TGFB1->TGFBR1", "ANXA1->FPR1"])
    score_rows = []
    for cohort in ["GSE211376", "GSE174554"]:
        result = read_analysis_json(f"{cohort.lower()}_communication_decontam.json")
        if result["dataset"] != cohort:
            raise ValueError(f"{cohort} 发送细胞筛选结果中的队列名不同")

        # 按现有上游或明确提供的旧归档取值。要复画历史分区，请接入冻结分区所生成的 10。
        # 每个面板只读自己展示的栏目。Fig5 的变化率由上游未取整评分算出。
        for pair in pairs:
            scores = result["pairs"][pair]
            row = {"cohort": cohort, "pair": pair}
            if not for_abstract:
                row["pct_change"] = scores["pct_change"]
            if pair == "TGFB2->TGFBR2":
                row.update(before=scores["before"], after=scores["after"])
            score_rows.append(row)
    scores_table_total = pd.DataFrame(score_rows).set_index(["cohort", "pair"], verify_integrity=True)
    if not for_abstract and not np.isfinite(scores_table_total["pct_change"]).all():
        raise ValueError("图示配体受体对缺少可读的变化率")
    return scores_table_total


def draw_tgfb2_scores(axis, scores_table_total):
    """绘制各队列 TGFB2–TGFBR2 筛选前后评分，按该图传入的百分比标注。源评分缺失时停止。"""
    pair_table = scores_table_total.xs("TGFB2->TGFBR2", level="pair")
    pair_table = pair_table.loc[["GSE211376", "GSE174554"], ["before", "after", "pct_change"]]
    if not np.isfinite(pair_table).all().all() or (pair_table[["before", "after"]] < 0).any().any():
        raise ValueError("TGFB2–TGFBR2 的评分缺失或取值无效")

    # 灰柱用完整发送群，紫柱用同一发送群中的 CNV 阴性细胞。柱顶百分比来自调用方传入的当前图口径。
    positions = np.arange(len(pair_table))
    axis.bar(positions - 0.17, pair_table["before"], width=0.34, color="#95A5A6", label="Full sender group")
    bars = axis.bar(positions + 0.17, pair_table["after"], width=0.34, color="#884EA0", label="CNV-negative senders")
    labels = []
    for row in pair_table.itertuples():
        label = f"{row.pct_change:+.0f}%"
        labels.append(label)
    axis.bar_label(bars, labels=labels, padding=2, fontsize=7, color="#693781", fontweight="bold")
    axis.set_xticks(positions, ["GSE211376\n(discovery)", "GSE174554\n(validation)"], fontsize=7)
    axis.set_ylabel("TGFB2–TGFBR2\nexpression-product score", fontsize=7.5)
    axis.set_ylim(bottom=0, top=max(pair_table["before"].max(), pair_table["after"].max(), 0.01) * 1.32)
    axis.legend(fontsize=6, frameon=False, loc="upper right")
    return labels


def label_panel(axis, letter):
    """为一个面板加上图内索引，位置使用轴坐标，无返回值。"""
    axis.text(-0.12, 1.09, letter, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")


def save_figure(figure, module_name, required_text):
    """写单页 PDF，以 Poppler 生成 300 dpi PNG。内容、页数、尺寸或像素核对失败就停止，无返回值。"""
    output_dir = project_dir / "03_result" / module_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 只从本次绘制的 PDF 生成位图。先在同一目录的临时位置核对，避免把旧图误当成本次结果。
    with TemporaryDirectory(dir=output_dir) as temporary:
        pdf_path = Path(temporary) / f"{module_name}.pdf"
        FigureCanvasAgg(figure).draw()
        figure.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight", facecolor="white")

        # Poppler 用 point 报告页面宽高，1 英寸为 72 point。固定其输出语言以识别元数据字段。
        info = subprocess.run(
            ["pdfinfo", pdf_path], check=True, capture_output=True, text=True, env={**os.environ, "LC_ALL": "C"}
        )
        if not re.search(r"(?m)^Pages:\s+1\s*$", info.stdout):
            raise ValueError("结果 PDF 应为单页")
        size = re.search(r"(?m)^Page size:\s+([\d.]+) x ([\d.]+) pts", info.stdout)
        if size is None:
            raise ValueError("未能读出 PDF 页面的物理尺寸")

        # 从落盘页面提取文字，逐图核对不同面板的标签和数据。行间换行不影响比对。
        content = subprocess.run(["pdftotext", "-raw", "-enc", "UTF-8", pdf_path, "-"], check=True, capture_output=True)
        page_text = "".join(content.stdout.decode("utf-8").split())
        for label in required_text:
            if "".join(label.split()) not in page_text:
                raise ValueError(f"结果 PDF 缺少应出现的图内文字 {label!r}")

        # PNG 由这一份 PDF 渲染。允许 Poppler 把不足一像素的页边取整，并检查图片确实包含图形。
        prefix = pdf_path.with_suffix("")
        subprocess.run(["pdftoppm", "-r", "300", "-singlefile", "-png", pdf_path, prefix], check=True)
        png_path = pdf_path.with_suffix(".png")
        with Image.open(png_path) as raster:
            raster.load()
            pixel_size = np.asarray(raster.size)
            expected_size = np.asarray(size.groups(), dtype=float) * 300 / 72
            if raster.format != "PNG" or np.any(np.abs(pixel_size - expected_size) > 1.1):
                raise ValueError("PNG 像素尺寸与本页 PDF 的 300 dpi 尺寸不符")
            low, high = raster.convert("L").getextrema()
            if low == high:
                raise ValueError("PNG 全页为同一颜色")

        # 两种格式均通过核对后，发布这次生成的文件。
        pdf_path.replace(output_dir / pdf_path.name)
        png_path.replace(output_dir / png_path.name)
