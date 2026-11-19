# ======================================================================
# 00a_论文图形样式.py —— PONE-D-26-35909 论文图形公共函数库
# 项目简称 PONE-D-26-35909，供原分析和返修分析共同调用。
#
# 目的：
# 统一配色、面板标题和 PDF 导出，保留投稿图及早期公共队列图各自的版式。
# 只读取调用者传入的绘图对象和表；返回值、输出文件及停止条件写在对应函数旁。
# 使用时可从分析脚本的调用位置查阅本文件。
# 在编辑器中用“转到定义”（Mac 使用 Cmd + 点击）或搜索函数名定位。
# 各函数按调用处查找，无需单独运行本文件。
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.text import Text
from matplotlib.transforms import ScaledTranslation
from PIL import Image, ImageChops
from pypdf import PdfReader

# 投稿及返修图以同一种颜色标识同一细胞状态或队列；保留冻结色值和色阶顺序。
ink = "#252A2E"
muted = "#687177"
grid = "#E3E7E9"
pale = "#F2F5F6"
white = "#FFFFFF"

teal = "#2D7F82"
teal_light = "#8FC7C5"
coral = "#D86F55"
coral_light = "#F0B39F"
blue = "#4B78A8"
blue_light = "#AFC7DD"
gold = "#C99A3D"
purple = "#7B6A9D"
green = "#5C8D72"
gray = "#AAB1B5"
gray_dark = "#70787D"

cohort_colors = {
    "GSE162631": purple,
    "GSE211376": teal,
    "GSE273274": coral,
    "GSE274987": gold,
    "GSE276177": blue,
    "GSE286419": green,
}

state_colors = {"AC-like": coral, "Other malignant": blue, "Myeloid": green}

region_colors = {"Tumor": coral, "Core": coral, "Edge": gold, "Peritumoral": teal, "Peripheral": teal}

diverging = LinearSegmentedColormap.from_list("plottie_diverging", [blue, blue_light, white, coral_light, coral])
teal_seq = LinearSegmentedColormap.from_list("plottie_teal", ["#F0F5F5", teal_light, teal])
coral_seq = LinearSegmentedColormap.from_list("plottie_coral", ["#F7F2EF", coral_light, coral])
purple_seq = LinearSegmentedColormap.from_list("plottie_purple", ["#F3F1F6", "#BFB5D2", purple])

# 两套历史图共用同一纳入顺序；GSE162631 仅在统一注释等指定敏感性面板中加入。
cohort_order = ["GSE211376", "GSE273274", "GSE274987", "GSE276177", "GSE286419"]
contrast_ligands = ["TGFB2", "CSF1", "ANXA1", "CCL2", "TGFB1", "SPP1"]


def configure_plottie_style() -> None:
    """设置投稿和返修图的字体优先序、线宽、图例及固定画布。无返回值。"""
    mpl.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "font.weight": "normal",
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.5,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.edgecolor": ink,
            "axes.labelcolor": ink,
            "axes.linewidth": 0.65,
            "axes.grid": False,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "xtick.color": ink,
            "ytick.color": ink,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "legend.fontsize": 6.8,
            "legend.frameon": False,
            "lines.linewidth": 1.0,
            "lines.markersize": 4.2,
            "patch.linewidth": 0.65,
            "figure.facecolor": white,
            "axes.facecolor": white,
            "savefig.facecolor": white,
            "savefig.dpi": 600,
            "savefig.bbox": None,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    """把字母和原有左标题放在同一行，沿用标题实际字号和字重。无返回值。"""
    title = ax.get_title(loc="left")
    elements = [TextArea(label, textprops={"fontsize": 10.0, "fontweight": "bold", "color": ink})]
    if title:

        # 通过公开的 Artist 接口读取标题，避免把轴内图注或默认字号用作面板字号。
        for artist in ax.get_children():
            if not isinstance(artist, Text) or artist in ax.texts:
                continue
            if artist.get_ha() == "left" and artist.get_text() == title:
                props = {"fontsize": artist.get_fontsize(), "fontweight": artist.get_fontweight()}
                elements.append(TextArea(title, textprops={**props, "color": ink, "linespacing": 1.0}))
                break
        else:
            raise ValueError(f"找不到面板 {label} 的左标题对象")
    ax.set_title("", loc="left")

    # 字母与标题顶部对齐，整体放在轴左上方 5 磅处。
    header = HPacker(children=elements, align="top", pad=0, sep=5)
    transform = ax.transAxes + ScaledTranslation(0, 5 / 72, ax.figure.dpi_scale_trans)
    anchored = AnchoredOffsetbox(
        loc            = "lower left",
        child          = header,
        frameon        = False,
        bbox_to_anchor = (0, 1),
        bbox_transform = transform,
        borderpad      = 0,
        pad            = 0,
    )
    ax.add_artist(anchored)


def public_panel_label(ax: plt.Axes, label: str) -> None:
    """在早期公共队列图原有的轴坐标写面板字母，保留标题位置。无返回值。"""
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def open_axes(ax: plt.Axes, grid_axis: str | None = None, *, tick_width=0.65, tick_length=2.6) -> None:
    """隐藏上、右边框，可选一个方向的网格。两套图传入各自的刻度尺寸。无返回值。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=tick_width, length=tick_length)
    if grid_axis:
        ax.grid(axis=grid_axis, color=grid, linewidth=0.55, zorder=0)


def clean_axes(ax: plt.Axes) -> None:
    """去掉边框及刻度线，保留刻度文字。无返回值。"""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def median_bar(ax, center, value, width=0.34, color=ink, linewidth=2.0) -> None:
    """在给定组的中位数上画一条短横线。无返回值。"""
    ax.plot(
        [center - width / 2, center + width / 2],
        [value, value],
        color     = color,
        linewidth = linewidth,
        solid_capstyle = "round",
        zorder    = 4,
    )


def add_spatial_panel(fig, ax, x, y, values, *, cmap, title, point_size=3.8):
    """按传入坐标画空间值并返回散点对象。沿用原公共 API 的方向和色限规则。"""
    valid = mpl.cbook.safe_masked_invalid(values).compressed()
    vmin, vmax = 0.0, 1.0
    if len(valid):
        vmin, vmax = float(valid.min()), float(valid.max())

    # 原公共函数仅在超过十个有效点时截取 1% 和 99% 分位；常数图沿用 0–1 色域。
    if len(valid) > 10:
        vmin, vmax = np.quantile(valid, [0.01, 0.99])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0
    scatter = ax.scatter(x, y, c=values, s=point_size, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)

    ax.set_aspect("equal")
    ax.set_title(title, loc="left", pad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    clean_axes(ax)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.040, pad=0.018)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=5.8, width=0.5, length=2)
    return scatter


def paired_expression_table(contrast_row: pd.Series) -> pd.DataFrame:
    """展开冻结对比表一行的患者配对值。患者标识重复或人数与原行不符时停止。"""
    patient_table = pd.DataFrame(json.loads(contrast_row["patient_values"]))
    if patient_table["patient"].isna().any() or patient_table["patient"].duplicated().any():
        raise ValueError("TGFB2 配对值缺少患者标识或同一患者重复")
    if len(patient_table) != contrast_row["n_patients"]:
        raise ValueError("图中 TGFB2 配对患者数与冻结对比表不符")
    return patient_table


def state_expression_heatmap_table(expression_table: pd.DataFrame) -> pd.DataFrame:
    """返回队列×恶性状态的配体列 z 分数；无合格患者的组合保留为空。"""
    states = ["AC-like", "MES-like", "OPC-like", "NPC-like"]
    target_region = np.where(expression_table["cohort"].eq("GSE274987"), "Core", "Tumor")

    # 一行对应患者、区域和状态；原方案只在指定区域且该状态达到 20 个细胞时取患者中位数。
    eligible = expression_table["region"].eq(target_region) & expression_table["n_cells"].ge(20)
    eligible_table = expression_table.loc[eligible & expression_table["state"].isin(states)]
    mean_columns = [f"mean_{gene}" for gene in contrast_ligands]
    median_table = eligible_table.groupby(["cohort", "state"])[mean_columns].median()
    ordered_index = pd.MultiIndex.from_product([cohort_order, states], names=["cohort", "state"])
    median_table = median_table.reindex(ordered_index)
    median_table.columns = contrast_ligands

    # 每个配体独立在所有队列和状态之间标准化；总体标准差 ddof=0 与两张原补图相同。
    z_score_table = (median_table - median_table.mean(axis=0)) / median_table.std(axis=0, ddof=0)
    return z_score_table


def outside_canvas_text(fig: plt.Figure) -> list:
    """返回实际绘制且超出固定画布的文字与像素边界，供投稿图的公开版面报告使用。"""
    renderer = fig.canvas.get_renderer()
    hidden_ticks = set()
    axes = []
    for ax in fig.axes:
        axes.extend((ax.xaxis, ax.yaxis))
    for axis in axes:
        low, high = sorted(axis.get_view_interval())
        for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
            if not low <= tick.get_loc() <= high:
                hidden_ticks.update((tick.label1, tick.label2))

    # 对数轴等会缓存显示范围外的刻度；只度量实际显示的文字，沿用一屏幕像素容差。
    overflow = []
    for artist in fig.findobj(Text):
        if artist in hidden_ticks or not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer)
        if box.width <= 0 or box.height <= 0:
            continue
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
            overflow.append({"text": artist.get_text(), "bbox": list(box.bounds)})
    return overflow


def write_layout_report(fig, path: Path) -> None:
    """写出公开的文字边界报告；若文字超出固定画布，在异常中列出具体对象。无返回值。"""
    overflow = outside_canvas_text(fig)
    path.write_text(json.dumps({"outside_canvas": overflow}, indent=2), encoding="utf-8")
    if overflow:
        raise ValueError(f"{path.stem} 有文字越出画布：{overflow}")


def save_figure(fig, directory, stem, *, panels="", tiff=False, layout_report=False) -> None:
    """导出 PDF、PNG 和所选主图 TIFF。页面、面板或位图不符时停止。无返回值。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        fig.canvas.draw()

        # 同目录临时写出并回读本次图件，核对完成才用公开文件名替换旧图。
        with TemporaryDirectory(dir=directory) as staging:
            work = Path(staging)
            pdf_path = work / f"{stem}.pdf"
            fig.savefig(pdf_path, format="pdf", dpi=600, facecolor=white)
            page_size = check_figure_pdf(pdf_path, fig, panels)
            write_raster_copies(pdf_path, page_size, tiff)

            # 投稿说明明确列出版面报告。保留原 JSON 字段，使读者可定位超出画布的具体文字。
            if layout_report:
                write_layout_report(fig, work / f"{stem}_layout.json")

            for artifact in work.iterdir():
                artifact.replace(directory / artifact.name)
    finally:
        plt.close(fig)


def check_figure_pdf(pdf_path: Path, fig: plt.Figure, panels: str) -> tuple:
    """回读 PDF 唯一页面，核对文字、面板字母及固定画布，返回实测页宽高（磅）。"""
    reader = PdfReader(pdf_path)
    if len(reader.pages) != 1:
        raise ValueError(f"{pdf_path.name} 应为单页")
    page = reader.pages[0]
    panel_labels = set()

    # 提取 PDF 中单独绘制的粗体片段。按整页空格分词会把紧邻标题的面板字母连到标题上。
    def collect_panel_label(text, _page_matrix, _text_matrix, font, _size) -> None:
        """按 pypdf 提供的字体与原文字片段收集面板字母。无返回值。"""
        label = text.strip()
        if len(label) == 1 and font and "bold" in str(font.get("/BaseFont", "")).lower():
            panel_labels.add(label)

    page_text = page.extract_text(visitor_text=collect_panel_label)
    if not page_text.strip() or any(letter not in panel_labels for letter in panels):
        raise ValueError(f"{pdf_path.name} 缺少文字或预期面板 {panels}")

    # 投稿样式指定 42 型字体，回读实际嵌入的 TrueType 字形；各系统按已声明的字体优先序选择。
    if mpl.rcParams["pdf.fonttype"] == 42:
        for reference in page["/Resources"]["/Font"].get_object().values():
            font = reference.get_object()
            descendants = font.get("/DescendantFonts", [])
            embedded = any("/FontFile2" in entry.get_object()["/FontDescriptor"] for entry in descendants)
            if not embedded:
                raise ValueError(f"{pdf_path.name} 未嵌入预期字体 {font.get('/BaseFont')}")

    # 公共队列图按内容紧裁；投稿及返修图按整张画布对尺寸。所有位图按实测 PDF 页面派生。
    page_size = (float(page.mediabox.width), float(page.mediabox.height))
    if min(page_size) <= 0:
        raise ValueError(f"{pdf_path.name} 页面为空")
    same_size = np.allclose(page_size, fig.get_size_inches() * 72, rtol=0, atol=1e-5)
    if mpl.rcParams["savefig.bbox"] is None and not same_size:
        raise ValueError(f"{pdf_path.name} 页面尺寸与绘制画布不一致")
    return page_size


def write_raster_copies(pdf_path: Path, page_size: tuple, tiff: bool) -> None:
    """以 600 dpi 渲染已核对的 PDF，用同一组 RGB 像素写 PNG 和可选 LZW TIFF。无返回值。"""
    width, height = (round(points * 600 / 72) for points in page_size)
    with TemporaryDirectory(dir=pdf_path.parent) as rendered:
        prefix = Path(rendered) / "page"

        # 明确最终整数像素边界，避免浮点取整多出一行白边。仅渲染一次，无缩放、裁切数据区或重绘散点。
        subprocess.run(
            ["pdftocairo", "-png", "-singlefile", "-r", "600", "-W", str(width), "-H", str(height), pdf_path, prefix],
            check=True,
        )
        with Image.open(prefix.with_suffix(".png")) as source:
            rgb_image = source.convert("RGB")

    if rgb_image.size != (width, height) or all(low == high for low, high in rgb_image.getextrema()):
        raise ValueError(f"{pdf_path.name} 位图尺寸不符或整页是单色")
    png_path = pdf_path.with_suffix(".png")
    rgb_image.save(png_path, dpi=(600, 600))
    files = [png_path]

    # Fig1.tif 等是投稿系统使用的名称；补图和返修图默认仅生成 PNG、PDF。
    if tiff:
        number = pdf_path.stem.removeprefix("Figure")
        if not number.isdecimal() or not pdf_path.stem.startswith("Figure"):
            raise ValueError("投稿 TIFF 只适用于带 Figure 编号的主图")
        tiff_path = pdf_path.with_name(f"Fig{number}.tif")
        rgb_image.save(tiff_path, compression="tiff_lzw", dpi=(600, 600))
        files.append(tiff_path)

    # 用实际文件核对格式、像素、分辨率和压缩。无损编码后每个 RGB 像素都应等于 PDF 渲染值。
    for path in files:
        with Image.open(path) as saved:
            if (
                saved.mode != "RGB"
                or saved.size != rgb_image.size
                or not np.allclose(saved.info["dpi"], 600, atol=0.02)
            ):
                raise ValueError(f"{path.name} 的颜色模式、尺寸或分辨率错误")
            if path.suffix == ".tif" and saved.info.get("compression") != "tiff_lzw":
                raise ValueError(f"{path.name} 应使用 LZW 压缩")
            if ImageChops.difference(saved, rgb_image).getbbox() is not None:
                raise ValueError(f"{path.name} 的像素与 PDF 渲染值不一致")


def require_inputs(directory: Path, names: list[str]) -> None:
    """开始绘图前检查所选模式实际读取的输入，列出缺失文件。无返回值。"""
    missing = [directory / name for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError("缺少绘图输入：\n" + "\n".join(str(path) for path in missing))


def require_cohort_inputs(data_dir: Path, *, supplementary: bool) -> None:
    """核对两种版式共用的冻结队列表，仅在需要补图时检查统一注释和表达输入。无返回值。"""
    names = [
        "cohort_manifest.csv",
        "ac_vs_other_ligand_contrasts.csv",
        "meta_ligand_results.csv",
        "patient_ac_fraction.csv",
        "regional_ac_fraction_tests.csv",
        "regional_communication_tests.csv",
    ]
    if supplementary:
        names.extend(
            [
                "patient_state_expression.csv",
                "uniform_annotation_patient_contrasts.csv",
                "uniform_annotation_cohort_results.csv",
                "uniform_annotation_summary.json",
            ]
        )
    require_inputs(data_dir, names)
