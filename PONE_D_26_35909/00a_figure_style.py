# ======================================================================
# 00a_figure_style.py — shared figure functions for PONE-D-26-35909
# The original and revision analyses both use this module.
#
# Purpose
# Share colors, panel headings and PDF export while preserving each figure set's layout.
# Inputs are figures and tables from the caller. Each function documents its outputs and failure conditions.
# Usage: look up each function at its call site in the analysis script.
# Use Go to Definition (Cmd-click on a Mac) or search for the function name.
# The analysis scripts import this library. It requires no separate run.
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

# The submission and revision figures use the same frozen colors and gradient order for each cell state and cohort.
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

# Both figure sets use the original cohort order. GSE162631 enters only the specified sensitivity analyses.
cohort_order = ["GSE211376", "GSE273274", "GSE274987", "GSE276177", "GSE286419"]
contrast_ligands = ["TGFB2", "CSF1", "ANXA1", "CCL2", "TGFB1", "SPP1"]


def configure_plottie_style() -> None:
    """Set submission and revision fonts, line widths, legends and fixed canvas size. Returns None."""
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
    """Place the letter beside the left title, using the title's actual size and weight. Returns None."""
    title = ax.get_title(loc="left")
    elements = [TextArea(label, textprops={"fontsize": 10.0, "fontweight": "bold", "color": ink})]
    if title:

        # Read the title through the public Artist API to distinguish it from annotations and default font settings.
        for artist in ax.get_children():
            if not isinstance(artist, Text) or artist in ax.texts:
                continue
            if artist.get_ha() == "left" and artist.get_text() == title:
                props = {"fontsize": artist.get_fontsize(), "fontweight": artist.get_fontweight()}
                elements.append(TextArea(title, textprops={**props, "color": ink, "linespacing": 1.0}))
                break
        else:
            raise ValueError(f"Could not find the left title for panel {label}")
    ax.set_title("", loc="left")

    # Top-align the letter and title, then place the group 5 points above the upper left of the axes.
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
    """Place the early panel letter at its original axes coordinates, preserving the title position. Returns None."""
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def open_axes(ax: plt.Axes, grid_axis: str | None = None, *, tick_width=0.65, tick_length=2.6) -> None:
    """Hide top and right spines, apply each layout's tick sizes and optionally add a grid on one axis. Returns None."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=tick_width, length=tick_length)
    if grid_axis:
        ax.grid(axis=grid_axis, color=grid, linewidth=0.55, zorder=0)


def clean_axes(ax: plt.Axes) -> None:
    """Remove spines and tick marks while retaining tick labels. Returns None."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def median_bar(ax, center, value, width=0.34, color=ink, linewidth=2.0) -> None:
    """Draw a short horizontal line at the supplied group median. Returns None."""
    ax.plot(
        [center - width / 2, center + width / 2],
        [value, value],
        color     = color,
        linewidth = linewidth,
        solid_capstyle = "round",
        zorder    = 4,
    )


def add_spatial_panel(fig, ax, x, y, values, *, cmap, title, point_size=3.8):
    """Plot spatial values and return the scatter artist, using the original coordinate and color-limit conventions."""
    valid = mpl.cbook.safe_masked_invalid(values).compressed()
    vmin, vmax = 0.0, 1.0
    if len(valid):
        vmin, vmax = float(valid.min()), float(valid.max())

    # The original API uses the 1st and 99th percentiles above ten valid points and a 0–1 range for constant maps.
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
    """Expand one frozen contrast into patient pairs. Reject duplicate IDs or a count inconsistent with the row."""
    patient_table = pd.DataFrame(json.loads(contrast_row["patient_values"]))
    if patient_table["patient"].isna().any() or patient_table["patient"].duplicated().any():
        raise ValueError("Paired TGFB2 values contain missing or duplicate patient IDs")
    if len(patient_table) != contrast_row["n_patients"]:
        raise ValueError("The number of plotted TGFB2 pairs differs from the frozen contrast table")
    return patient_table


def state_expression_heatmap_table(expression_table: pd.DataFrame) -> pd.DataFrame:
    """Return ligand-wise z scores by cohort and malignant state. Groups without eligible patients remain missing."""
    states = ["AC-like", "MES-like", "OPC-like", "NPC-like"]
    target_region = np.where(expression_table["cohort"].eq("GSE274987"), "Core", "Tumor")

    # Rows represent patient, region and state. Include patients with at least 20 cells of the state in the target
    # region.
    eligible = expression_table["region"].eq(target_region) & expression_table["n_cells"].ge(20)
    eligible_table = expression_table.loc[eligible & expression_table["state"].isin(states)]
    mean_columns = [f"mean_{gene}" for gene in contrast_ligands]
    median_table = eligible_table.groupby(["cohort", "state"])[mean_columns].median()
    ordered_index = pd.MultiIndex.from_product([cohort_order, states], names=["cohort", "state"])
    median_table = median_table.reindex(ordered_index)
    median_table.columns = contrast_ligands

    # Standardize each ligand across cohorts and states using the original population standard deviation (ddof=0).
    z_score_table = (median_table - median_table.mean(axis=0)) / median_table.std(axis=0, ddof=0)
    return z_score_table


def outside_canvas_text(fig: plt.Figure) -> list:
    """Return visible off-canvas text and its pixel bounds for the public submission layout report."""
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

    # Axes can cache ticks outside the visible range. Measure drawn labels with the original one-screen-pixel tolerance.
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
    """Write the public bounds report and identify any text outside the canvas. Returns None."""
    overflow = outside_canvas_text(fig)
    path.write_text(json.dumps({"outside_canvas": overflow}, indent=2), encoding="utf-8")
    if overflow:
        raise ValueError(f"{path.stem} contains text outside the canvas: {overflow}")


def save_figure(fig, directory, stem, *, panels="", tiff=False, layout_report=False) -> None:
    """Export PDF, PNG and optional main-figure TIFF. Reject invalid pages, panels or raster images. Returns None."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        fig.canvas.draw()

        # Write and read back the new files in this directory before replacing the public files.
        with TemporaryDirectory(dir=directory) as staging:
            work = Path(staging)
            pdf_path = work / f"{stem}.pdf"
            fig.savefig(pdf_path, format="pdf", dpi=600, facecolor=white)
            page_size = check_figure_pdf(pdf_path, fig, panels)
            write_raster_copies(pdf_path, page_size, tiff)

            # The submission lists layout reports. Preserve their JSON fields so readers can locate text outside the
            # canvas.
            if layout_report:
                write_layout_report(fig, work / f"{stem}_layout.json")

            for artifact in work.iterdir():
                artifact.replace(directory / artifact.name)
    finally:
        plt.close(fig)


def check_figure_pdf(pdf_path: Path, fig: plt.Figure, panels: str) -> tuple:
    """Read the single PDF page, check its text, panel letters and fixed canvas, and return its dimensions in points."""
    reader = PdfReader(pdf_path)
    if len(reader.pages) != 1:
        raise ValueError(f"{pdf_path.name} must contain one page")
    page = reader.pages[0]
    panel_labels = set()

    # Read bold text fragments separately because word splitting can join a panel letter to its title.
    def collect_panel_label(text, _page_matrix, _text_matrix, font, _size) -> None:
        """Collect panel letters from the font and text fragments supplied by pypdf. Returns None."""
        label = text.strip()
        if len(label) == 1 and font and "bold" in str(font.get("/BaseFont", "")).lower():
            panel_labels.add(label)

    page_text = page.extract_text(visitor_text=collect_panel_label)
    if not page_text.strip() or any(letter not in panel_labels for letter in panels):
        raise ValueError(f"{pdf_path.name} lacks text or one of the expected panel letters ({panels})")

    # PDF font type 42 requires embedded TrueType glyphs. The local system selects fonts in the declared order.
    if mpl.rcParams["pdf.fonttype"] == 42:
        for reference in page["/Resources"]["/Font"].get_object().values():
            font = reference.get_object()
            descendants = font.get("/DescendantFonts", [])
            embedded = any("/FontFile2" in entry.get_object()["/FontDescriptor"] for entry in descendants)
            if not embedded:
                raise ValueError(f"{pdf_path.name} lacks the embedded font {font.get('/BaseFont')}")

    # Public-cohort PDFs use tight cropping. Check submission and revision canvases at full size. Rasters use the PDF
    # page.
    page_size = (float(page.mediabox.width), float(page.mediabox.height))
    if min(page_size) <= 0:
        raise ValueError(f"{pdf_path.name} has an empty page")
    same_size = np.allclose(page_size, fig.get_size_inches() * 72, rtol=0, atol=1e-5)
    if mpl.rcParams["savefig.bbox"] is None and not same_size:
        raise ValueError(f"The page size of {pdf_path.name} differs from the figure canvas")
    return page_size


def write_raster_copies(pdf_path: Path, page_size: tuple, tiff: bool) -> None:
    """Render the checked PDF at 600 dpi and use the same RGB pixels for PNG and optional LZW TIFF. Returns None."""
    width, height = (round(points * 600 / 72) for points in page_size)
    with TemporaryDirectory(dir=pdf_path.parent) as rendered:
        prefix = Path(rendered) / "page"

        # Use integer pixel bounds to avoid a rounding-induced white row. Render once without resizing, cropping data or
        # redrawing points.
        subprocess.run(
            ["pdftocairo", "-png", "-singlefile", "-r", "600", "-W", str(width), "-H", str(height), pdf_path, prefix],
            check=True,
        )
        with Image.open(prefix.with_suffix(".png")) as source:
            rgb_image = source.convert("RGB")

    if rgb_image.size != (width, height) or all(low == high for low, high in rgb_image.getextrema()):
        raise ValueError(f"The raster of {pdf_path.name} has the wrong dimensions or contains only one color")
    png_path = pdf_path.with_suffix(".png")
    rgb_image.save(png_path, dpi=(600, 600))
    files = [png_path]

    # The submission system uses Fig1.tif and similar names. Supplementary and revision figures default to PDF and PNG.
    if tiff:
        number = pdf_path.stem.removeprefix("Figure")
        if not number.isdecimal() or not pdf_path.stem.startswith("Figure"):
            raise ValueError("Submission TIFF export requires a numbered main figure with a Figure prefix")
        tiff_path = pdf_path.with_name(f"Fig{number}.tif")
        rgb_image.save(tiff_path, compression="tiff_lzw", dpi=(600, 600))
        files.append(tiff_path)

    # Read back format, dimensions, resolution and compression. Lossless RGB output must match every rendered PDF pixel.
    for path in files:
        with Image.open(path) as saved:
            if (
                saved.mode != "RGB"
                or saved.size != rgb_image.size
                or not np.allclose(saved.info["dpi"], 600, atol=0.02)
            ):
                raise ValueError(f"{path.name} has the wrong color mode, dimensions or resolution")
            if path.suffix == ".tif" and saved.info.get("compression") != "tiff_lzw":
                raise ValueError(f"{path.name} must use LZW compression")
            if ImageChops.difference(saved, rgb_image).getbbox() is not None:
                raise ValueError(f"The pixels in {path.name} differ from the rendered PDF")


def require_inputs(directory: Path, names: list[str]) -> None:
    """Check the inputs read by this layout and list missing files before plotting. Returns None."""
    missing = [directory / name for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing figure inputs:\n" + "\n".join(str(path) for path in missing))


def require_cohort_inputs(data_dir: Path, *, supplementary: bool) -> None:
    """Check shared cohort tables and, for supplements, uniform annotations and expression inputs. Returns None."""
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
