# ==============================================================================
# 00a_空间分析.py —— GBM 空间与蛋白外部验证公共函数库
# PONE-D-26-35909，原分析的外部验证。
#
# 目的：
# 统一切片内相关、患者汇总、配体检验和空间表的写出。各队列在调用处选择自己的调整方式。
# CPTAC 的两份分析也共用同一套本地优先的原表读取规则。
# 阅读其它分析脚本时，遇到这里的函数再查阅。
# 在编辑器调用处按 Ctrl + 单击（Mac 按 Cmd + 单击）或按函数名跳到定义。
# 按调用处查找函数即可。

from __future__ import annotations

import gzip
import hashlib
import subprocess
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

# 面板顺序来自原分析；CPTAC 的置换与 bootstrap 按此顺序共用随机序列。
ligands = ["TGFB2", "TGFB1", "ANXA1", "SPP1", "CSF1", "CCL2"]
receptors = ["TGFBR1", "TGFBR2"]
ac_markers = ["GFAP", "S100B", "HOPX", "SLC1A3", "MLC1"]


def cptac_input(local_path, url, cache_dir, required_fields):
    """返回本地原表或已校验的 HTTPS gzip 缓存。缺文件且没给 URL、下载失败或内容异常时停止。"""
    if local_path.is_file():
        return local_path
    if not url:
        raise FileNotFoundError(f"请提供包含 {local_path.name} 的本地输入，或指定该文件的 HTTPS 地址")
    if urlsplit(url).scheme != "https" or not urlsplit(url).hostname:
        raise ValueError("CPTAC 原表地址须为完整的 HTTPS URL")

    # 同名文件的不同来源分别缓存；命中时不联网、不重写，也不在产物中存放原始 URL。
    source_id = hashlib.sha256(url.encode("utf-8")).hexdigest()
    cached_path = cache_dir / source_id / local_path.name
    if cached_path.is_file():
        return cached_path
    cached_path.parent.mkdir(parents=True, exist_ok=True)

    # 限定重定向协议；先读到 gzip 末尾核实校验码和表头，再原子公布完整缓存。
    with TemporaryDirectory(dir=cached_path.parent) as directory:
        partial = Path(directory) / "download.gz"
        command = [
            "curl", "--disable", "--fail", "--silent", "--show-error", "--location", "--proto", "=https",
            "--proto-redir", "=https", "--max-redirs", "5", "--connect-timeout", "15", "--max-time", "600",
            "--output", str(partial), "--", url,
        ]
        subprocess.run(command, check=True)
        with gzip.open(partial, "rb") as stream:
            columns = stream.readline().decode("utf-8-sig").strip().lower().split("\t")
            if not set(required_fields).issubset(columns) or not stream.readline().strip():
                raise ValueError(f"{local_path.name} 的下载内容缺少预期表头或数据行")
            while stream.read(1024 * 1024):
                pass

        # 同时下载同一文件时，保留最早通过核实的完整副本。
        try:
            cached_path.hardlink_to(partial)
        except FileExistsError:
            if not cached_path.is_file():
                raise

    return cached_path


def bh_adjust(p_values):
    """对同一指标的有效 P 值做 BH 校正，原位保留缺失；返回与输入索引一致的序列。"""
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().astype(float)
    if valid.empty:
        return result

    # 保留原程序先乘检验数再除秩的运算次序，按原行号放回校正值。
    order = np.argsort(valid.to_numpy())
    ranked = valid.to_numpy()[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[valid.index[order]] = np.minimum(adjusted, 1.0)
    return result


def signflip_p(values, gbmspace_rng=None):
    """返回患者或 plex 均值的双侧符号翻转 P 值；没有有限观测时返回缺失。"""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    count = len(values)
    if not count:
        missing = np.nan
        return missing

    # 只有 GBM-Space 在超过 20 名患者时按历史方案抽样，分批顺序、比较阈值与加一校正均保留。
    observed = abs(values.mean())
    if gbmspace_rng is not None and count > 20:
        extreme = 0
        for start in range(0, 100_000, 5_000):
            signs = gbmspace_rng.choice((-1.0, 1.0), size=(min(5_000, 100_000 - start), count))
            extreme += np.count_nonzero(np.abs((signs * values).mean(axis=1)) >= observed)
        probability = float((1 + extreme) / 100_001)
        return probability

    # 精确分支穷举所有正负组合，分块只控制内存；比较沿用已发表分析的舍入容差。
    extreme = 0
    patterns = 2**count
    for start in range(0, patterns, 5_000):
        indices = np.arange(start, min(start + 5_000, patterns))
        signs = 1 - 2 * ((indices[:, None] >> np.arange(count)) & 1)
        null = np.abs((signs * values).mean(axis=1))
        extreme += np.count_nonzero(null >= observed - 1e-15)
    probability = float(extreme / patterns)
    return probability


def patient_metrics(values, gbmspace_rng=None):
    """汇总一个指标的有效患者数、方向与检验；Wilcoxon 去掉零差，不可估计值留空。"""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    nonzero = values[values != 0]
    wilcoxon_p = np.nan
    if len(nonzero) >= 2:
        method = "auto" if gbmspace_rng is not None and len(nonzero) > 25 else "exact"
        wilcoxon_p = float(wilcoxon(nonzero, alternative="two-sided", method=method).pvalue)

    metrics = {
        "n": len(values), "median": float(np.median(values)) if len(values) else np.nan,
        "mean": float(np.mean(values)) if len(values) else np.nan, "n_positive": int(np.sum(values > 0)),
        "signflip_p": signflip_p(values, gbmspace_rng), "wilcoxon_p": wilcoxon_p,
    }
    return metrics


def spot_spearman(first, second):
    """返回同一切片内的 Spearman 相关；有效点位不足 100 个或变量为常量时留空。"""
    valid = np.isfinite(first) & np.isfinite(second)
    correlation = np.nan
    if valid.sum() >= 100 and np.unique(first[valid]).size >= 2 and np.unique(second[valid]).size >= 2:
        correlation = float(spearmanr(first[valid], second[valid]).statistic)
    return correlation


def rank_residual(values, controls, standardize_controls):
    """在切片完整点位上拟合含截距的秩回归，返回与原点位对齐的残差；不足 100 个点位时留空。"""
    rank_table = pd.DataFrame({"target": values})
    for i in range(controls.shape[1]):
        rank_table[f"control_{i}"] = controls[:, i]
    valid = np.isfinite(rank_table.to_numpy()).all(axis=1)
    residual = np.full(len(values), np.nan)
    if valid.sum() < 100:
        return residual

    # 各列各自取平均秩，再除以有效点位数加一。只在 GBM-Space 中把协变量中心化并除以标准差。
    ranked_table = rank_table.loc[valid].rank(method="average") / (valid.sum() + 1.0)
    outcome = ranked_table.pop("target").to_numpy(dtype=float)
    covariates = ranked_table.to_numpy(dtype=float)
    if standardize_controls:
        means = covariates.mean(axis=0)
        deviations = covariates.std(axis=0)
        keep = deviations > 1e-12
        covariates = (covariates[:, keep] - means[keep]) / deviations[keep]
    else:
        keep = np.nanstd(covariates, axis=0) > 1e-12
        covariates = covariates[:, keep]

    design = np.column_stack([np.ones(len(outcome)), covariates])
    coefficients, _, _, _ = np.linalg.lstsq(design, outcome, rcond=1e-10)
    residual[valid] = outcome - np.einsum("ij,j->i", design, coefficients)
    return residual


def write_spatial_tables(
    output_dir, section_ligand_table, patient_ligand_table, section_table, patient_table, summary_table,
    representative_table
):
    """核实切片、配体、患者与代表点位的对应关系，逐份写出并回读 CSV。无返回值。"""
    files = import_module("00a_项目路径")
    section_index_table = section_table.set_index("sample", verify_integrity=True)
    counts = section_ligand_table.groupby("sample")["ligand"].nunique()
    if set(counts.index) != set(section_index_table.index) or not counts.eq(len(ligands)).all():
        raise ValueError("切片配体表未为每张切片保留完整面板")
    if set(section_ligand_table["ligand"]) != set(ligands) or set(summary_table["ligand"]) != set(ligands):
        raise ValueError("切片或汇总表的配体与预设面板不一致")

    # 逐行核切片的患者及点位数，再核患者级表与切片表中的患者、配体组合。
    samples = section_ligand_table["sample"]
    for column in ["patient", "n_spots"]:
        if not section_ligand_table[column].eq(samples.map(section_index_table[column])).all():
            raise ValueError(f"切片配体表与空间表的 {column} 不一致")
    expected = set(zip(section_ligand_table["patient"], section_ligand_table["ligand"]))
    actual = set(zip(patient_ligand_table["patient"], patient_ligand_table["ligand"]))
    if actual != expected or set(patient_table["patient"]) != set(section_table["patient"]):
        raise ValueError("患者级表与切片表的患者或配体不一致")

    # 代表表应完整保留一张切片的点位，GBM-Space 的有效点位筛选只用于绘图。
    samples = representative_table["sample"].unique()
    if len(samples) != 1 or len(representative_table) != section_index_table.at[samples[0], "n_spots"]:
        raise ValueError("代表切片的点位行数与切片汇总不一致")
    coordinates = ["x", "y"] if "x" in representative_table else ["pixel_col", "pixel_row"]

    files.write_table(section_ligand_table, output_dir / "01_切片配体指标.csv", ["sample", "ligand"])
    files.write_table(patient_ligand_table, output_dir / "02_患者配体指标.csv", ["patient", "ligand"])
    files.write_table(section_table, output_dir / "03_切片空间指标.csv", ["sample"])
    files.write_table(patient_table, output_dir / "04_患者空间指标.csv", ["patient"])
    files.write_table(summary_table, output_dir / "05_配体汇总.csv", ["ligand"])
    files.write_table(representative_table, output_dir / "06_代表切片点位.csv", coordinates)
