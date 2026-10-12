#!/usr/bin/env python3
"""生成支持患者级结论所需的精简细胞级表达与元数据。"""

from __future__ import annotations

import csv
import gzip
import hashlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from stage_public_paths import project_root, public_input_root, submission_package


ROOT = project_root()
PACKAGE = submission_package(ROOT)
INPUT_ROOT = public_input_root(ROOT)
OUT = PACKAGE / "minimal_cell_data"

TARGET_GENES = [
    "TGFB2",
    "TGFB1",
    "ANXA1",
    "SPP1",
    "CSF1",
    "CCL2",
    "TGFBR1",
    "TGFBR2",
    "FPR1",
    "CD44",
    "CSF1R",
    "CCR2",
    "GFAP",
    "S100B",
    "HOPX",
    "SLC1A3",
    "MLC1",
    "VIM",
    "CHI3L1",
    "HILPDA",
    "DDIT3",
    "ENO2",
    "PLP1",
    "OLIG1",
    "OMG",
    "TNR",
    "ALCAM",
    "SOX4",
    "SOX11",
    "DCX",
    "CD24",
    "STMN1",
    "STMN2",
]
MALIGNANT_STATES = {"AC-like", "MES-like", "OPC-like", "NPC-like"}
MYELOID_STATES = {"Macrophage", "Microglia", "Monocyte"}
EXPECTED_OBS = {
    "GSE211376_minimal.h5ad": 20656,
    "scGBM_minimal.h5ad": 54600,
    "snGBM_minimal.h5ad": 44869,
    "regional_GSE274987_minimal.h5ad": 100908,
}


def sha256(path: Path) -> str:
    """分块计算文件校验和。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_matrix(matrix) -> sparse.csr_matrix:
    """统一保存为 float32 稀疏矩阵。"""
    if sparse.issparse(matrix):
        result = matrix.tocsr().astype(np.float32)
    else:
        result = sparse.csr_matrix(np.asarray(matrix, dtype=np.float32))
    result.eliminate_zeros()
    return result


def write_minimal_h5ad(
    source: Path,
    target_name: str,
    obs_columns: list[str],
    cell_mask,
    scope: str,
) -> dict[str, object]:
    """从大型 AnnData 中只提取论文实际需要的细胞和基因。"""
    adata = ad.read_h5ad(source, backed="r")
    missing_genes = sorted(set(TARGET_GENES) - set(adata.var_names))
    missing_obs = sorted(set(obs_columns) - set(adata.obs.columns))
    if missing_genes or missing_obs:
        adata.file.close()
        raise ValueError(
            f"{source.name} 缺少基因 {missing_genes} 或元数据列 {missing_obs}"
        )
    mask = np.asarray(cell_mask(adata.obs), dtype=bool)
    obs = adata.obs.loc[mask, obs_columns].copy()
    obs = obs.apply(lambda column: column.astype(str))
    obs.index = adata.obs_names[mask].astype(str)
    matrix = adata[mask, TARGET_GENES].X
    result = ad.AnnData(
        X=normalize_matrix(matrix),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(TARGET_GENES, name="gene")),
    )
    result.uns["data_role"] = (
        "Minimal cell-level expression and metadata supporting the reported "
        "patient-level analyses; not a replacement for the public raw archive."
    )
    result.uns["cell_scope"] = scope
    result.uns["source_file"] = source.name
    result.uns["removed_components"] = (
        "Unused genes, layers, raw matrix, embeddings, graphs, and unneeded "
        "observation metadata."
    )
    target = OUT / target_name
    result.write_h5ad(target, compression="gzip")
    adata.file.close()
    return {
        "file": target.name,
        "source": source.name,
        "n_cells": result.n_obs,
        "n_genes": result.n_vars,
        "obs_columns": ";".join(obs_columns),
        "cell_scope": scope,
        "bytes": target.stat().st_size,
        "sha256": sha256(target),
    }


def write_gse211376() -> dict[str, object]:
    """流式提取 GSE211376 的目标基因并保留归一化所需总计数。"""
    counts_path = INPUT_ROOT / "raw_inputs" / "GSE211376_raw_counts.tsv.gz"
    metadata_path = INPUT_ROOT / "raw_inputs" / "GSE211376_metadata.csv.gz"
    metadata = pd.read_csv(metadata_path, index_col=0)
    with gzip.open(counts_path, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        cells = header[1:] if header[0] == "" else header
        total = np.zeros(len(cells), dtype=np.float64)
        selected = {
            gene: np.zeros(len(cells), dtype=np.float32) for gene in TARGET_GENES
        }
        seen: set[str] = set()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            values = np.asarray(parts[1:], dtype=np.float32)
            if len(values) != len(cells):
                raise ValueError(f"GSE211376 行长度错误: {parts[0]}")
            total += values
            if parts[0] in selected:
                selected[parts[0]] = values
                seen.add(parts[0])
    missing = sorted(set(TARGET_GENES) - seen)
    if missing:
        raise ValueError(f"GSE211376 缺少目标基因: {missing}")
    metadata = metadata.reindex(cells)
    if metadata[["predicted.high_hierarchy", "patient"]].isna().any(axis=None):
        raise ValueError("GSE211376 计数列与元数据未完全对齐")
    keep_labels = MALIGNANT_STATES | {"TAM-MG", "TAM-BDM"}
    keep = metadata["predicted.high_hierarchy"].astype(str).isin(keep_labels).to_numpy()
    scale = total / 1e4
    scale[scale == 0] = 1.0
    expression = np.column_stack(
        [np.log1p(selected[gene] / scale) for gene in TARGET_GENES]
    ).astype(np.float32)
    obs = metadata.loc[
        keep, ["predicted.high_hierarchy", "patient"]
    ].copy()
    obs = obs.apply(lambda column: column.astype(str))
    obs["dataset"] = "GSE211376"
    obs["region"] = "Tumor"
    obs["total_counts"] = total[keep]
    result = ad.AnnData(
        X=normalize_matrix(expression[keep]),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(TARGET_GENES, name="gene")),
    )
    result.uns["data_role"] = (
        "Minimal normalized cell-level expression and metadata supporting the "
        "reported patient-level analyses."
    )
    result.uns["normalization"] = "Library-size normalization to 10,000 followed by log1p."
    result.uns["source_files"] = f"{counts_path.name};{metadata_path.name}"
    target = OUT / "GSE211376_minimal.h5ad"
    result.write_h5ad(target, compression="gzip")
    return {
        "file": target.name,
        "source": f"{counts_path.name};{metadata_path.name}",
        "n_cells": result.n_obs,
        "n_genes": result.n_vars,
        "obs_columns": ";".join(result.obs.columns),
        "cell_scope": "Malignant states and TAM-MG/TAM-BDM cells used by stage45 or stage48/51.",
        "bytes": target.stat().st_size,
        "sha256": sha256(target),
    }


def main() -> None:
    """生成四个精简对象及其可核验清单。"""
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*"):
        if old.is_file():
            old.unlink()

    rows = [write_gse211376()]
    rows.append(
        write_minimal_h5ad(
            INPUT_ROOT / "processed_inputs" / "scGBM_GSE162631_GSE273274.h5ad",
            "scGBM_minimal.h5ad",
            [
                "dataset",
                "patient_id",
                "tissue_type",
                "Minor_cluster",
                "Major_category",
            ],
            lambda obs: (
                (
                    obs["dataset"].astype(str).eq("GSE162631")
                    & obs["Major_category"].astype(str).eq("Tumor")
                )
                | (
                    obs["dataset"].astype(str).eq("GSE273274")
                    & (
                        obs["Major_category"].astype(str).eq("Tumor")
                        | obs["Minor_cluster"].astype(str).isin(
                            {"AC", "MES1", "OPC", "NPC1", "NPC2"}
                        )
                        | obs["Major_category"].astype(str).isin(
                            {"TAM/Macrophage", "Microglia", "Monocyte"}
                        )
                    )
                )
            ),
            (
                "GSE162631 malignant cells used by uniform-marker sensitivity "
                "and GSE273274 malignant or myeloid cells used by primary or "
                "uniform-marker analyses."
            ),
        )
    )
    rows.append(
        write_minimal_h5ad(
            INPUT_ROOT / "processed_inputs" / "snGBM_GSE276177_GSE286419.h5ad",
            "snGBM_minimal.h5ad",
            ["dataset", "patient", "tissue", "celltype"],
            lambda obs: (
                obs["dataset"].astype(str).isin({"GSE276177", "GSE286419"})
                & obs["celltype"].astype(str).isin(
                    MALIGNANT_STATES | MYELOID_STATES
                )
            ),
            (
                "GSE276177 and GSE286419 malignant or myeloid cells used by "
                "primary or uniform-marker analyses."
            ),
        )
    )
    rows.append(
        write_minimal_h5ad(
            INPUT_ROOT / "processed_inputs" / "regional_GSE274987_GSE241037.h5ad",
            "regional_GSE274987_minimal.h5ad",
            ["dataset", "patient_id", "tissue_type", "celltype", "Platform"],
            lambda obs: (
                obs["dataset"].astype(str).eq("GSE274987")
                & obs["celltype"].astype(str).isin(
                    MALIGNANT_STATES | MYELOID_STATES
                )
            ),
            (
                "Joint GSE274986/GSE274987 malignant or myeloid cells used by "
                "primary or uniform-marker analyses. The legacy dataset value "
                "is GSE274987; Platform maps Multiome to GSE274986 and ParseBio "
                "to GSE274987. All GSE241037 cells were excluded."
            ),
        )
    )

    for row in rows:
        expected = EXPECTED_OBS[row["file"]]
        if row["n_cells"] != expected:
            raise ValueError(
                f"{row['file']} 细胞数 {row['n_cells']} 与预期 {expected} 不符"
            )

    manifest = OUT / "minimal_cell_data_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Generated {len(rows)} minimal cell-level files: "
        f"{sum(int(row['n_cells']) for row in rows):,} cells, "
        f"{sum(int(row['bytes']) for row in rows) / (1024 ** 2):.2f} MiB"
    )


if __name__ == "__main__":
    main()
