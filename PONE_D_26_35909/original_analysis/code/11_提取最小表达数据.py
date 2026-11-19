#!/usr/bin/env python3
# 目的：从原公共输入提取主分析与统一注释共同需要的细胞、基因和元数据。
# 输入：PUBLIC_INPUT_ROOT 中的 GSE211376_raw_counts.tsv.gz、GSE211376_metadata.csv.gz 和三个 processed_inputs/*.h5ad；
# 发布 minimal_cell_data/minimal_cell_data_manifest.csv 提供基因数、细胞数、obs 列的历史对照。
# 输出：重算包 minimal_cell_data/ 下 GSE211376_minimal.h5ad、scGBM_minimal.h5ad、snGBM_minimal.h5ad、
# regional_GSE274987_minimal.h5ad 和 minimal_cell_data_manifest.csv。原始输入中的患者编号、细胞名按原样提取；
# 随论文发布的副本另做过重编码。
# 运行方式：在 code 目录运行 python 11_提取最小表达数据.py。
# 保留 Python/AnnData 的 X、obs、var、uns 和原 gzip 存储方式，独立包写入 01_公共队列重算/。

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

primary = import_module("01_公共队列患者表达分析")
uniform = import_module("02_统一注释敏感性")
paths = import_module("00a_项目路径")


def main() -> None:
    """核对所有对象后再发布本次的五个指定文件，保留输出目录内的其他文件。无返回值。"""
    primary.stop_if_missing()
    files = paths.primary_input_files()
    output = paths.analysis_output_root() / "minimal_cell_data"
    output.mkdir(parents=True, exist_ok=True)

    # 在同一文件系统暂存整套结果，任一对象失败时不会先覆盖已有对象或 manifest。
    with TemporaryDirectory(dir=output) as scratch:
        staging = Path(scratch)
        rows = [write_gse211376(staging)]
        rows.append(
            write_minimal_h5ad(
                files["scGBM"],
                "scGBM_minimal.h5ad",
                ["dataset", "patient_id", "tissue_type", "Minor_cluster", "Major_category"],
                select_sc_cells,
                "GSE162631 malignant cells used by uniform-marker sensitivity "
                "and GSE273274 malignant or myeloid cells used by primary or uniform-marker analyses.",
                directory=staging,
            )
        )
        rows.append(
            write_minimal_h5ad(
                files["snGBM"],
                "snGBM_minimal.h5ad",
                ["dataset", "patient", "tissue", "celltype"],
                select_sn_cells,
                "GSE276177 and GSE286419 malignant or myeloid cells used by primary or uniform-marker analyses.",
                directory=staging,
            )
        )
        rows.append(
            write_minimal_h5ad(
                files["GSE274987"],
                "regional_GSE274987_minimal.h5ad",
                ["dataset", "patient_id", "tissue_type", "celltype", "Platform"],
                select_regional_cells,
                "Joint GSE274986/GSE274987 malignant or myeloid cells used by primary or uniform-marker analyses. "
                "The legacy dataset value is GSE274987; Platform maps Multiome to GSE274986 and ParseBio to GSE274987. "
                "All GSE241037 cells were excluded.",
                directory=staging,
            )
        )

        # 历史 manifest 的字节数和摘要对应重编码前的文件。细胞数、基因数和列名可与公开版本核对。
        manifest_table = pd.DataFrame(rows)
        manifest_name = "minimal_cell_data_manifest.csv"
        reference = Path(__file__).resolve().parents[1] / "minimal_cell_data" / manifest_name
        reference_table = pd.read_csv(reference)
        columns = ["file", "n_cells", "n_genes", "obs_columns"]
        expected_table = reference_table[columns].sort_values("file").reset_index(drop=True)
        actual_table = manifest_table[columns].sort_values("file").reset_index(drop=True)
        pd.testing.assert_frame_equal(expected_table, actual_table)
        paths.write_table(manifest_table, staging / manifest_name, ["file"])

        for name in [*manifest_table["file"], manifest_name]:
            (staging / name).replace(output / name)
    print(f"提取 {len(manifest_table)} 个对象，共 {manifest_table['n_cells'].sum()} 个细胞或细胞核。")


def select_sc_cells(obs_table: pd.DataFrame) -> pd.Series:
    """GSE162631 仅取原 Tumor；GSE273274 取主分析与统一注释候选细胞的并集。"""
    dataset = obs_table["dataset"].astype(str)
    tumor = uniform.sc_malignant(obs_table)
    annotated = obs_table["Minor_cluster"].astype(str).isin(primary.sc_state_labels)
    eligible = tumor | annotated | primary.sc_myeloid(obs_table)
    selected = (dataset.eq("GSE162631") & tumor) | (dataset.eq("GSE273274") & eligible)
    return selected


def select_sn_cells(obs_table: pd.DataFrame) -> pd.Series:
    """仅留两个单核队列中四种恶性状态与三个髓系标签。"""
    dataset = obs_table["dataset"].astype(str).isin(["GSE276177", "GSE286419"])
    selected = dataset & (uniform.celltype_malignant(obs_table) | primary.celltype_myeloid(obs_table))
    return selected


def select_regional_cells(obs_table: pd.DataFrame) -> pd.Series:
    """联合区域队列由历史 dataset 值识别；原样保存两种 Platform，排除小鼠来源。"""
    dataset = obs_table["dataset"].astype(str).eq("GSE274987")
    selected = dataset & (uniform.celltype_malignant(obs_table) | primary.celltype_myeloid(obs_table))
    return selected


def write_gse211376(directory: Path | None = None) -> dict:
    """从全基因原计数提取已归一化目标矩阵，并保存各细胞归一化前的总计数。返回 manifest 一行。"""
    metadata_table, expression, totals = primary.read_count_expression(target_genes)
    selected = metadata_table["predicted.high_hierarchy"].isin(primary.malignant_states) | primary.tam_myeloid(
        metadata_table
    )
    obs_table = metadata_table.loc[selected, ["predicted.high_hierarchy", "patient"]].astype(str).copy()
    obs_table["dataset"] = "GSE211376"
    obs_table["region"] = "Tumor"
    obs_table["total_counts"] = totals[selected]
    files = paths.primary_input_files()
    sources = f"{files['GSE211376_counts'].name};{files['GSE211376_meta'].name}"

    annotations = {
        "data_role": (
            "Minimal normalized cell-level expression and metadata supporting the reported patient-level analyses."
        ),
        "normalization": "Library-size normalization to 10,000 followed by log1p.",
        "source_files": sources,
    }
    target = (directory or paths.analysis_output_root() / "minimal_cell_data") / "GSE211376_minimal.h5ad"

    # cell_scope 是原 manifest 字段，旧阶段号属于已发表文本。
    record = save_minimal(
        target, expression[selected], obs_table, annotations, sources,
        "Malignant states and TAM-MG/TAM-BDM cells used by stage45 or stage48/51."
    )
    return record


# 先排主分析的配体、受体，再按统一注释顺序补未出现的 marker，保留发表矩阵的列序。
target_genes = list(dict.fromkeys([*primary.primary_genes, *uniform.selected_genes]))


def write_minimal_h5ad(
    source: Path, target_name: str, obs_columns: list[str], cell_mask, scope: str, *, directory: Path | None = None
) -> dict:
    """按来源筛选细胞与列并保留表达原值；缺基因、缺元数据列时停止。返回 manifest 一行。"""
    source_object = ad.read_h5ad(source, backed="r")
    try:
        absent_genes = sorted(set(target_genes) - set(source_object.var_names))
        absent_columns = sorted(set(obs_columns) - set(source_object.obs.columns))
        if absent_genes or absent_columns:
            raise ValueError(f"{source.name} 缺少基因 {absent_genes} 或元数据列 {absent_columns}")

        keep = np.asarray(cell_mask(source_object.obs), dtype=bool)
        obs_table = source_object.obs.loc[keep, obs_columns].astype(str).copy()
        obs_table.index = source_object.obs_names[keep].astype(str)
        matrix = source_object[keep, target_genes].X
    finally:
        source_object.file.close()

    annotations = {
        "data_role": "Minimal cell-level expression and metadata supporting the reported patient-level analyses; "
        "not a replacement for the public raw archive.",
        "cell_scope": scope,
        "source_file": source.name,
        "removed_components": (
            "Unused genes, layers, raw matrix, embeddings, graphs, and unneeded observation metadata."
        ),
    }
    target = (directory or paths.analysis_output_root() / "minimal_cell_data") / target_name
    record = save_minimal(target, matrix, obs_table, annotations, source.name, scope)
    return record


def save_minimal(
    target: Path, matrix, obs_table: pd.DataFrame, annotations: dict, source_name: str, scope: str
) -> dict:
    """保存 CSR float32；重新打开对象，逐项核对顺序、obs、uns 和表达，再填写来源清单。"""
    expression = normalize_matrix(matrix)
    if expression.shape != (len(obs_table), len(target_genes)):
        raise ValueError(f"{target.name} 的表达矩阵与细胞、基因数量不一致")
    result = ad.AnnData(X=expression, obs=obs_table, var=pd.DataFrame(index=pd.Index(target_genes, name="gene")))
    result.uns.update(annotations)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(target, compression="gzip")

    # 小矩阵可以完整回读，两个稀疏矩阵按坐标相减可检出数值错位或写入截断。
    saved = ad.read_h5ad(target)
    if saved.var_names.tolist() != target_genes or dict(saved.uns) != annotations:
        raise ValueError(f"{target.name} 的基因顺序或来源字段发生变化")
    pd.testing.assert_frame_equal(obs_table, saved.obs, check_dtype=False, check_categorical=False)
    if saved.shape != expression.shape or saved.X.dtype != np.float32 or (saved.X != expression).nnz:
        raise ValueError(f"{target.name} 的表达维度、精度或值发生变化")
    record = {
        "file": target.name,
        "source": source_name,
        "n_cells": saved.n_obs,
        "n_genes": saved.n_vars,
        "obs_columns": ";".join(saved.obs.columns),
        "cell_scope": scope,
        "bytes": target.stat().st_size,
        "sha256": paths.file_sha256(target),
    }
    return record


def normalize_matrix(matrix) -> sparse.csr_matrix:
    """统一稀疏存储格式并去掉显式零；输入中的表达值已按来源处理。"""
    if sparse.issparse(matrix):
        expression = matrix.tocsr().astype(np.float32)
    else:
        expression = sparse.csr_matrix(np.asarray(matrix, dtype=np.float32))
    expression.eliminate_zeros()
    return expression


if __name__ == "__main__":
    main()
