# ==============================================================================
# 00a_项目路径.py —— GBM 患者表达分析公共函数库
# PONE-D-26-35909，公共队列原分析与复现。
#
# 目的：定位分析包、公共输入和本轮输出，统一检查多个脚本写出的文件。
# 输入为环境变量、各脚本产生的数据表或图，输出为路径、经核对的 CSV、JSON 或 PDF/PNG。
# 运行方式为在分析脚本中用 import_module("00a_项目路径") 引入。
# 阅读分析脚本时，遇到这里的函数再查阅。
# 在编辑器的调用处用 Ctrl + 单击（Mac 为 Cmd + 单击）可跳到定义。
# 函数按调用位置查找。

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def project_root() -> Path:
    """由 GLIOMA_VALIDATION_ROOT 指定源工程，独立包默认取 code 的上一级。"""
    configured = os.environ.get("GLIOMA_VALIDATION_ROOT")
    root = Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[1]
    return root


def submission_package(root: Path) -> Path:
    """返回投稿包位置。保留独立包、源工程布局及 PUBLIC_ONLY_PACKAGE 的优先级。"""
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
    """返回含 raw_inputs、processed_inputs 和 upstream_source 的公共输入目录。"""
    configured = os.environ.get("PUBLIC_INPUT_ROOT")
    directory = Path(configured).expanduser().resolve() if configured else root / "public_inputs"
    return directory


def analysis_output_root() -> Path:
    """冻结包旁另建重算目录。明确指定的独立工作包继续使用原有 data 等子目录。"""
    package = submission_package(project_root())
    released_package = Path(__file__).resolve().parents[1]
    if package == released_package or (package / "source_data").is_dir() or (package / "RELEASE_NOTES.md").is_file():
        package = package / "01_公共队列重算"
    return package


def primary_input_files() -> dict[str, Path]:
    """清单名对应原论文的五个输入；定位本身不要求文件已经下载。"""
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
    """为既有 manifest 的摘要字段分块读文件。打不开文件时向调用者报告。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    value = digest.hexdigest()
    return value


def write_table(result_table, path: Path, row_columns: list[str]) -> None:
    """按指定列标识每一行；回读列顺序、标识和数值后替换目标 CSV。无返回值。"""
    import pandas as pd

    if result_table.columns.has_duplicates or result_table.duplicated(row_columns).any():
        raise ValueError(f"{path.name} 的列名或行标识重复，行标识列为 {row_columns}")
    if result_table[row_columns].isna().any(axis=None):
        raise ValueError(f"{path.name} 有空的行标识")

    # 患者编号等文本即使看起来是数字也按文本回读；空数值恢复为缺失值。
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
    """回读整份摘要，保留旧 JSON 的字段及不可估计值。无返回值。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as scratch:
        target = Path(scratch) / path.name
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        saved = json.loads(target.read_text(encoding="utf-8"))
        if json.dumps(saved, sort_keys=True) != json.dumps(record, sort_keys=True):
            raise ValueError(f"{path.name} 写出的字段或数值不一致")
        target.replace(path)


def save_figure(figure, pdf_path: Path) -> None:
    """先绘制 PDF，再由同一页生成 600 dpi PNG；核实页数、像素尺寸和可见内容。无返回值。"""
    import math
    import re
    import subprocess

    from PIL import Image

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=pdf_path.parent) as scratch:
        pdf = Path(scratch) / pdf_path.name
        figure.canvas.draw()
        figure.savefig(pdf, format="pdf", facecolor="white", bbox_inches="tight")

        # Poppler 报告的页面大小以 pt 表示；按 72 pt/英寸换算，容许页面尺寸显示舍入引起的一像素差。
        pdf_info = subprocess.run(
            ["pdfinfo", str(pdf)], capture_output=True, text=True, check=True, env={**os.environ, "LC_ALL": "C"}
        )
        pages = re.search(r"^Pages:\s+(\d+)", pdf_info.stdout, re.MULTILINE)
        dimensions = re.search(r"^Page size:\s+([\d.]+) x ([\d.]+) pts", pdf_info.stdout, re.MULTILINE)
        if pages is None or pages.group(1) != "1" or dimensions is None:
            raise ValueError(f"{pdf.name} 的 PDF 页数或纸张尺寸异常")
        width, height = (math.ceil(float(value) * 600 / 72) for value in dimensions.groups())
        png = pdf.with_suffix(".png")
        subprocess.run(
            ["pdftocairo", "-png", "-singlefile", "-r", "600", str(pdf), str(pdf.with_suffix(""))], check=True
        )

        with Image.open(png) as picture:
            if picture.format != "PNG" or abs(picture.width - width) > 1 or abs(picture.height - height) > 1:
                raise ValueError(f"{png.name} 的格式或像素尺寸与 PDF 不符")
            picture.thumbnail((256, 256))
            low, high = picture.convert("L").getextrema()
            if low == high:
                raise ValueError(f"{pdf.name} 栅格化后整页为单色")

        pdf.replace(pdf_path)
        png.replace(pdf_path.with_suffix(".png"))


def result_input_directory(*names: str) -> Path:
    """优先读取同一轮重算的整组表；没有该组重算表时使用冻结包，不混拼两处文件。"""
    directory = analysis_output_root() / "data"
    if not any((directory / name).exists() for name in names):
        directory = submission_package(project_root()) / "data"
    for name in names:
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    return directory
