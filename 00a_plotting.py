# ======================================================================
# 00a_plotting.py — Shared plotting functions for the GBM astrocyte study
# Used by the original study figures and graphical abstract in the project root.
#
# Purpose
# Read the same cohort results and use consistent denominators across figures. Render each single-page PDF at 300 dpi.
# Changes to a shared reader or export function apply to every figure that calls it.
# When reading a figure script, look up the function here at its call site.
# An editor's Go to Definition command (Ctrl-click or Cmd-click) can open the shared definition.
# You can also search this file for the function name.

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

# Resolve inputs and outputs relative to this module, including when the scripts are copied to a test directory.
project_dir = Path(__file__).resolve().parent
source_dir = project_dir / "01_figure_sources"

# Use the original figure font. Embed selectable glyphs so labels can be checked in the exported PDF.
mpl.rcParams.update(
    {"font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
)


def analysis_path(archive_name, required=True):
    """Find a current or archived output. Report both missing paths. Optional inputs have public figure values."""
    current_names = {
        "noncircular_cnv_validation.json": "01_sngbm_immune_reference_cnv/01_sngbm_immune_reference_cnv.json",
        "gse211376_marker_noncircular.json": (
            "03_gse211376_marker_defined_astrocytes/01_gse211376_marker_defined_astrocyte_cnv.json"
        ),
        "gse174554_cluster.npz": "04_gse174554_clustering_and_markers/01_gse174554_clusters.npz",
        "gse174554_leiden_cnv.json": "05_gse174554_cluster_cnv/02_gse174554_cluster_cnv.json",
        "communication_analysis_results.json": (
            "07_scgbm_communication_and_mixture_adjustment/01_communication_and_mixture_adjustment.json"
        ),
        "liana_validation.json": "08_sngbm_ligand_ranks/01_liana_ligand_ranks.json",
        "gse211376_communication_decontam.json": "09_gse211376_sender_filtering/01_sender_filtering_scores.json",
        "gse174554_communication_decontam.json": "10_gse174554_sender_filtering/01_sender_filtering_scores.json",
        "fig6_concordance_gse211376_vs_scgbm.csv": (
            "12_gse211376_subtype_expression_and_ranks/02_cross_cohort_ac_ranks.csv"
        ),
    }

    # Prefer an existing current result. The reader will report a corrupt file, permission error, or missing field.
    current_path = project_dir / "03_result" / current_names[archive_name]
    if current_path.exists() or current_path.is_symlink():
        return current_path

    archive_path = project_dir / "data" / "intermediate" / archive_name
    if not (archive_path.exists() or archive_path.is_symlink()):
        if required:
            raise FileNotFoundError(
                f"Generate {current_path.relative_to(project_dir)} or supply the local archive "
                f"{archive_path.relative_to(project_dir)}."
            )
        warnings.warn(f"No upstream {archive_name}; using values published in the original code.", stacklevel=2)
        result = None
        return result
    warnings.warn(f"Using the archived {archive_name} from data/intermediate.", stacklevel=2)
    return archive_path


def read_analysis_json(archive_name, required=True):
    """Read current or archived JSON using its original field names. Return None only for an absent optional input."""
    path = analysis_path(archive_name, required)
    if path is None:
        return path
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    return result


def read_cnv_table(require_references=False):
    """Return cohort sizes, platforms, and CNV percentages. Prefer upstream fields and preserve missing groups."""
    cohort_table = pd.read_csv(source_dir / "01_cohort_sizes_and_astrocyte_cnv_rates.csv", index_col="cohort")
    reference_table = pd.read_csv(source_dir / "02_reference_cell_cnv_rates.csv", index_col="cohort")
    cnv_table_total = cohort_table.join(reference_table, validate="one_to_one")

    # Stage 03 defines the target by markers. Stage 02 uses the authors' labels and a different threshold.
    discovery = read_analysis_json("gse211376_marker_noncircular.json", required=require_references)
    if discovery is not None:
        cnv_table_total.at["GSE211376", "cells"] = discovery["n_cells"]
        cnv_table_total.at["GSE211376", "astro_pct"] = discovery["marker_defined_astrocyte"]["cnv_pos_pct"]
        cnv_table_total.at["GSE211376", "myeloid_pct"] = discovery["reference_FPR"]["myeloid"]
        cnv_table_total.at["GSE211376", "oligo_pct"] = discovery["reference_FPR"]["oligodendrocyte"]
        cnv_table_total.at["GSE211376", "author_ac_pct"] = discovery["author_AClike"]["cnv_pos_pct"]
        cnv_table_total.at["GSE211376", "author_astro_pct"] = discovery["author_astrocyte_only"]["cnv_pos_pct"]

    # Stage 01 supplies the strict snGBM rate and two reference rates. For scGBM, retain the original figure values
    # because the root pipeline has no corresponding analysis with the same definitions.
    nuclear = read_analysis_json("noncircular_cnv_validation.json", required=False)
    if nuclear is not None:
        independent = nuclear["snGBM_independent"]
        cnv_table_total.at["GSE276177+286419", "astro_pct"] = independent["strict_CNVpos_pct"]
        cnv_table_total.at["GSE276177+286419", "myeloid_pct"] = independent["reference_FPR"]["Macrophage"]
        cnv_table_total.at["GSE276177+286419", "oligo_pct"] = independent["reference_FPR"]["Oligodendrocyte"]

    # Stage 05 supplies the Leiden cluster and reference rates. Each leiden entry from stage 04 is one included nucleus.
    leiden = read_analysis_json("gse174554_leiden_cnv.json", required=require_references)
    if leiden is not None:
        cnv_table_total.at["GSE174554", "astro_pct"] = leiden["astrocyte_cluster"]["cnv_pos_pct"]
        cnv_table_total.at["GSE174554", "myeloid_pct"] = leiden["reference_FPR"]["myeloid"]
        cnv_table_total.at["GSE174554", "oligo_pct"] = leiden["reference_FPR"]["oligodendrocyte"]
    cluster_path = analysis_path("gse174554_cluster.npz", required=False)
    if cluster_path is not None:
        with np.load(cluster_path, allow_pickle=False) as clustering:
            cnv_table_total.at["GSE174554", "cells"] = len(clustering["leiden"])

    # The original code reports all overall percentages. Fig2 needs measured reference rates from stages 03 and 05.
    # The original figure marks the scGBM oligodendrocyte reference as unavailable.
    if not cnv_table_total["astro_pct"].between(0, 100).all():
        raise ValueError("The target-group CNV percentage is missing or outside 0–100.")
    if require_references:
        for column in reference_table.columns:
            values = cnv_table_total[column]
            missing = (values.index == "GSE162631+273274") & (column == "oligo_pct")
            if not np.array_equal(values.isna(), missing) or not values.dropna().between(0, 100).all():
                raise ValueError(f"Unexpected missing groups or out-of-range percentages in reference column {column}.")
    if not cnv_table_total[["cells", "patients"]].gt(0).all().all():
        raise ValueError("Cohort cell counts and patient counts must be positive.")
    return cnv_table_total


def draw_cnv_rates(axis, cnv_table_total, decimals):
    """Plot target-group CNV positivity in cohort order and return the bar labels for checking the PDF."""
    positions = np.arange(len(cnv_table_total))
    rates = cnv_table_total["astro_pct"]
    colors = ["#27AE60" if cohort == "GSE211376" else "#C0392B" for cohort in cnv_table_total.index]
    bars = axis.bar(positions, rates, color=colors, width=0.66, zorder=2)
    labels = [f"{rate:.{decimals}f}%" for rate in rates]
    axis.bar_label(bars, labels=labels, padding=2, fontsize=7, fontweight="bold")

    # The dashed line marks the nominal upper tail of the 95th-percentile threshold. Fig2c shows observed rates.
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
    """Read the five reported changes for Fig5 or just the TGFB2–TGFBR2 before/after scores for the abstract."""
    pairs = ["TGFB2->TGFBR2"]
    if not for_abstract:
        pairs.extend(["TGFB2->TGFBR1", "CSF1->CSF1R", "TGFB1->TGFBR1", "ANXA1->FPR1"])
    score_rows = []
    for cohort in ["GSE211376", "GSE174554"]:
        result = read_analysis_json(f"{cohort.lower()}_communication_decontam.json")
        if result["dataset"] != cohort:
            raise ValueError(f"The sender-filtering output has a different cohort name than {cohort}.")

        # Use the supplied upstream result or local original archive. To reproduce the historical figure,
        # supply stage 10 output generated from the frozen partition.
        # Read the fields displayed in this panel. Fig5 uses changes calculated upstream from unrounded scores.
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
        raise ValueError("The plotted ligand–receptor pairs require finite percentage changes.")
    return scores_table_total


def draw_tgfb2_scores(axis, scores_table_total):
    """Plot TGFB2–TGFBR2 scores using the percentages supplied by the caller. Raise an error for missing scores."""
    pair_table = scores_table_total.xs("TGFB2->TGFBR2", level="pair")
    pair_table = pair_table.loc[["GSE211376", "GSE174554"], ["before", "after", "pct_change"]]
    if not np.isfinite(pair_table).all().all() or (pair_table[["before", "after"]] < 0).any().any():
        raise ValueError("TGFB2–TGFBR2 scores are missing or invalid.")

    # Gray bars use all senders. Purple bars use CNV-negative cells from the same sender group.
    # The calling figure determines how to calculate the percentage labels.
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
    """Add the panel index in axes coordinates. No return value."""
    axis.text(-0.12, 1.09, letter, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")


def save_figure(figure, module_name, required_text):
    """Write a single-page PDF and a 300 dpi PNG. Verify content, size, and pixels. No return value."""
    output_dir = project_dir / "03_result" / module_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Render from the PDF drawn for this run. Check both files in a temporary directory beside the final outputs.
    with TemporaryDirectory(dir=output_dir) as temporary:
        pdf_path = Path(temporary) / f"{module_name}.pdf"
        FigureCanvasAgg(figure).draw()
        figure.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight", facecolor="white")

        # Poppler gives page dimensions in points (72 per inch). Set its output language to read the field names.
        info = subprocess.run(
            ["pdfinfo", pdf_path], check=True, capture_output=True, text=True, env={**os.environ, "LC_ALL": "C"}
        )
        if not re.search(r"(?m)^Pages:\s+1\s*$", info.stdout):
            raise ValueError("The output PDF must have exactly one page.")
        size = re.search(r"(?m)^Page size:\s+([\d.]+) x ([\d.]+) pts", info.stdout)
        if size is None:
            raise ValueError("Could not read the PDF page dimensions.")

        # Extract text from the saved page and verify the requested panel labels and values across line breaks.
        content = subprocess.run(["pdftotext", "-raw", "-enc", "UTF-8", pdf_path, "-"], check=True, capture_output=True)
        page_text = "".join(content.stdout.decode("utf-8").split())
        for label in required_text:
            if "".join(label.split()) not in page_text:
                raise ValueError(f"Expected figure text is missing from the output PDF: {label!r}")

        # Rasterize this page. Allow Poppler to round fractional edge pixels and check for visible content.
        prefix = pdf_path.with_suffix("")
        subprocess.run(["pdftoppm", "-r", "300", "-singlefile", "-png", pdf_path, prefix], check=True)
        png_path = pdf_path.with_suffix(".png")
        with Image.open(png_path) as raster:
            raster.load()
            pixel_size = np.asarray(raster.size)
            expected_size = np.asarray(size.groups(), dtype=float) * 300 / 72
            if raster.format != "PNG" or np.any(np.abs(pixel_size - expected_size) > 1.1):
                raise ValueError("PNG dimensions do not match the PDF rendered at 300 dpi.")
            low, high = raster.convert("L").getextrema()
            if low == high:
                raise ValueError("The entire PNG has a single color.")

        # Publish the two checked files from this run.
        pdf_path.replace(output_dir / pdf_path.name)
        png_path.replace(output_dir / png_path.name)
