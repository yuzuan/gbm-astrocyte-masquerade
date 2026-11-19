# ==============================================================================
# 00a_project_paths.py - Shared paths and file checks for the GBM patient expression analysis
# PONE-D-26-35909, original public-cohort analysis and reproduction.
#
# Purpose: Locate the analysis package, public inputs and current outputs. Check files written by multiple stages.
# Inputs are environment variables, result tables and figures. Outputs are paths or validated CSV, JSON, PDF and PNG
# files.
# Usage: Load this library with import_module("00a_project_paths") from the analysis scripts.
# Look up a function here when you encounter its call in an analysis script.
# To jump from a call to its definition, Ctrl-click in your editor (Cmd-click on macOS).
# Find definitions by their names at the call sites.

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def project_root() -> Path:
    """Locate the source project using GLIOMA_VALIDATION_ROOT, or the parent of code in a standalone package."""
    configured = os.environ.get("GLIOMA_VALIDATION_ROOT")
    root = Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[1]
    return root


def submission_package(root: Path) -> Path:
    """Locate the submission package in either layout, giving PUBLIC_ONLY_PACKAGE precedence."""
    configured = os.environ.get("PUBLIC_ONLY_PACKAGE")
    if configured:
        package = Path(configured).expanduser().resolve()
        return package
    has_data = (root / "data").is_dir() and (root / "source_data").is_dir()
    if (root / "manuscript_public_only.md").is_file() or has_data:
        package = root
    else:
        package = root / "claude_outputs" / "public_only_submission"
    return package


def public_input_root(root: Path) -> Path:
    """Return the public input directory containing raw_inputs, processed_inputs and upstream_source."""
    configured = os.environ.get("PUBLIC_INPUT_ROOT")
    directory = Path(configured).expanduser().resolve() if configured else root / "public_inputs"
    return directory


def analysis_output_root() -> Path:
    """Give releases a separate reanalysis directory. Explicit working packages keep their usual data layout."""
    package = submission_package(project_root())
    released_package = Path(__file__).resolve().parents[1]
    if package == released_package or (package / "source_data").is_dir() or (package / "RELEASE_NOTES.md").is_file():
        package = package / "01_cohort_reanalysis"
    return package


def primary_input_files() -> dict[str, Path]:
    """Map the five published input names to paths. The files need not have been downloaded yet."""
    inputs = public_input_root(project_root())
    files = {
        "GSE211376_counts": inputs / "raw_inputs" / "GSE211376_raw_counts.tsv.gz",
        "GSE211376_meta": inputs / "raw_inputs" / "GSE211376_metadata.csv.gz",
        "scGBM": inputs / "processed_inputs" / "scGBM_GSE162631_GSE273274.h5ad",
        "snGBM": inputs / "processed_inputs" / "snGBM_GSE276177_GSE286419.h5ad",
        "GSE274987": inputs / "processed_inputs" / "regional_GSE274987_GSE241037.h5ad",
    }
    return files


def file_sha256(path: Path) -> str:
    """Read a file in chunks for its published manifest digest. Propagate file access errors."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    value = digest.hexdigest()
    return value


def write_table(result_table, path: Path, row_columns: list[str]) -> None:
    """Read back keys, column order and values from the temporary CSV before replacing the target. Returns None."""
    import pandas as pd

    if result_table.columns.has_duplicates or result_table.duplicated(row_columns).any():
        raise ValueError(f"{path.name}: duplicate column names or row keys in {row_columns}")
    if result_table[row_columns].isna().any(axis=None):
        raise ValueError(f"{path.name}: missing row keys")

    # Read text, including numeric-looking patient IDs, as text. Restore empty numeric fields as missing.
    text_types = {}
    missing_values = {}
    chinese_pattern = r"[\u3400-\u9fff]"
    has_chinese = result_table.columns.astype(str).str.contains(chinese_pattern).any()
    for column in result_table.columns:
        if pd.api.types.is_numeric_dtype(result_table[column]):
            missing_values[column] = [""]
        else:
            text_types[column] = str
            if not has_chinese:
                has_chinese = result_table[column].astype("string").str.contains(chinese_pattern, na=False).any()

    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as scratch:
        target = Path(scratch) / path.name
        result_table.to_csv(target, index=False, encoding="utf-8-sig" if has_chinese else "utf-8")
        saved_table = pd.read_csv(
            target, dtype=text_types, keep_default_na=False, na_values=missing_values, float_precision="round_trip"
        )
        pd.testing.assert_frame_equal(
            result_table.reset_index(drop=True),
            saved_table,
            check_dtype=False,
            check_categorical=False,
            check_exact=True,
            obj=path.name,
        )
        target.replace(path)


def write_json(record: dict, path: Path) -> None:
    """Read the saved summary back, preserving all published fields and non-estimable values. Returns None."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as scratch:
        target = Path(scratch) / path.name
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        saved = json.loads(target.read_text(encoding="utf-8"))
        if json.dumps(saved, sort_keys=True) != json.dumps(record, sort_keys=True):
            raise ValueError(f"{path.name}: fields or values changed when the JSON was saved")
        target.replace(path)


def save_figure(figure, pdf_path: Path) -> None:
    """Render a PDF, convert that page to a 600 dpi PNG and check page count, pixel size and content. Returns None."""
    import math
    import re
    import subprocess

    from PIL import Image

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=pdf_path.parent) as scratch:
        pdf = Path(scratch) / pdf_path.name
        figure.canvas.draw()
        figure.savefig(pdf, format="pdf", facecolor="white", bbox_inches="tight")

        # Poppler reports page size in points (72 per inch). Allow one pixel for rounding the reported dimensions.
        pdf_info = subprocess.run(
            ["pdfinfo", str(pdf)], capture_output=True, text=True, check=True, env={**os.environ, "LC_ALL": "C"}
        )
        pages = re.search(r"^Pages:\s+(\d+)", pdf_info.stdout, re.MULTILINE)
        dimensions = re.search(r"^Page size:\s+([\d.]+) x ([\d.]+) pts", pdf_info.stdout, re.MULTILINE)
        if pages is None or pages.group(1) != "1" or dimensions is None:
            raise ValueError(f"{pdf.name}: unexpected PDF page count or dimensions")
        width, height = (math.ceil(float(value) * 600 / 72) for value in dimensions.groups())
        png = pdf.with_suffix(".png")
        subprocess.run(
            ["pdftocairo", "-png", "-singlefile", "-r", "600", str(pdf), str(pdf.with_suffix(""))], check=True
        )

        with Image.open(png) as picture:
            if picture.format != "PNG" or abs(picture.width - width) > 1 or abs(picture.height - height) > 1:
                raise ValueError(f"{png.name}: unexpected image format or pixel dimensions for the PDF page")
            picture.thumbnail((256, 256))
            low, high = picture.convert("L").getextrema()
            if low == high:
                raise ValueError(f"{pdf.name}: the rasterized page is a single color")

        pdf.replace(pdf_path)
        png.replace(pdf_path.with_suffix(".png"))


def result_input_directory(*names: str) -> Path:
    """Read all tables from one reanalysis. Use frozen tables when none of the requested outputs exist."""
    directory = analysis_output_root() / "data"
    if not any((directory / name).exists() for name in names):
        directory = submission_package(project_root()) / "data"
    for name in names:
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    return directory
