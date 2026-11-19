"""目的　从保留的全基因计数提取 TGFB2 与标志基因，并核对已存的归一化表达。
输入　GLIOMA_VALIDATION_ROOT/public_inputs 中的 processed_inputs 三个 H5AD 和 raw_inputs 的 GSE211376 计数及注释。
输出　01_计数与预处理/01_计数输入核对.csv 与 09_队列细胞缓存 中逐队列的细胞表和面板矩阵。
运行　在 revision_analysis 下用 Python 运行 code/01a_读取全基因计数.py。AnnData 和 SciPy 读取稀疏层。
"""

import gzip
import os
from importlib import import_module
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem
from scipy import sparse

patient = import_module("00a_患者表达")
source_root = (
    Path(os.environ.get("GLIOMA_VALIDATION_ROOT", Path(__file__).resolve().parents[1])) / "public_inputs"
)
cache = patient.result_root() / "01_计数与预处理/09_队列细胞缓存"
cache.mkdir(parents=True, exist_ok=True)
audit_rows = []
cache_rows = []


def csr_block(group, start, stop):
    """从 AnnData 的 CSR 编码读取连续细胞行。其余编码没有相同的指针含义。"""
    if group.attrs["encoding-type"] != "csr_matrix":
        raise ValueError("输入层须使用 AnnData CSR 编码")
    pointers = group["indptr"][start : stop + 1]
    left, right = int(pointers[0]), int(pointers[-1])
    values = group["data"][left:right]
    columns = group["indices"][left:right]
    matrix = sparse.csr_matrix(
        (values, columns, pointers - left), shape=(stop - start, int(group.attrs["shape"][1]))
    )
    return matrix


# 每次只读一个队列对象。文库分母使用该细胞的全部保留基因，面板矩阵只用于后续打分。
for filename in (
    "scGBM_GSE162631_GSE273274.h5ad",
    "snGBM_GSE276177_GSE286419.h5ad",
    "regional_GSE274987_GSE241037.h5ad",
    "GSE211376",
):
    if filename != "GSE211376":
        with h5py.File(source_root / "processed_inputs" / filename, "r") as source:
            cell_table = read_elem(source["obs"])
            gene_table = read_elem(source["var"])
            columns = gene_table.index.get_indexer(patient.genes)
            if not cell_table.index.is_unique or not gene_table.index.is_unique or np.any(columns < 0):
                raise ValueError(f"细胞、基因编号重复或缺少面板基因：{filename}")

            # H5AD 以细胞为行。按块计算每个细胞的文库、检出基因数和线粒体比例。
            mitochondrial = gene_table.index.str.upper().str.startswith("MT-")
            n_genes = len(gene_table)
            selected = np.zeros((len(cell_table), len(columns)), dtype=np.float32)
            stored = np.zeros_like(selected)
            library = np.zeros(len(cell_table))
            detected = np.zeros(len(cell_table), dtype=int)
            mt_percent = np.zeros(len(cell_table))
            for start in range(0, len(cell_table), 2048):
                end = min(start + 2048, len(cell_table))
                block = csr_block(source["layers/counts"], start, end)
                valid = np.isfinite(block.data) & (block.data >= 0) & (block.data == np.rint(block.data))
                if not valid.all():
                    raise ValueError(f"计数不是有限非负整数：{filename}")
                library[start:end] = np.asarray(block.sum(axis=1)).ravel()
                detected[start:end] = np.asarray((block > 0).sum(axis=1)).ravel()
                mt_percent[start:end] = (
                    100 * np.asarray(block[:, mitochondrial].sum(axis=1)).ravel() / library[start:end]
                )
                selected[start:end] = block[:, columns].toarray()
                stored[start:end] = csr_block(source["X"], start, end)[:, columns].toarray()

        # scGBM 的历史标签存于 Minor_cluster。其他对象直接提供细胞状态。仅保留人队列。
        cell_table["cell_id"] = cell_table.index.astype(str)
        cell_table["cohort"] = cell_table["dataset"].astype(str)
        cell_table["patient"] = cell_table["patient_id" if "patient_id" in cell_table else "patient"].astype(
            str
        )
        cell_table["region"] = cell_table["tissue_type" if "tissue_type" in cell_table else "tissue"].astype(
            str
        )
        if filename.startswith("scGBM"):
            names = {
                "AC": "AC-like",
                "MES1": "MES-like",
                "OPC": "OPC-like",
                "NPC1": "NPC-like",
                "NPC2": "NPC-like",
            }
            cell_table["primary_state"] = (
                cell_table["Minor_cluster"].astype(str).map(names).fillna("Excluded")
            )
            cell_table["malignant_pool"] = cell_table["Major_category"].astype(str).eq("Tumor")
        else:
            malignant = cell_table["celltype"].isin(patient.states)
            cell_table["primary_state"] = cell_table["celltype"].astype(str).where(malignant, "Excluded")
            cell_table["malignant_pool"] = malignant
        keep = cell_table["cohort"].ne("GSE241037").to_numpy()
        scope = "retained cells and genes in complete project AnnData"
        source_name = filename

    else:

        # GEO 计数以基因为行，文件首行是细胞。逐行累加，避免载入完整稠密矩阵。
        metadata_table = pd.read_csv(source_root / "raw_inputs/GSE211376_metadata.csv.gz", index_col=0)
        with gzip.open(source_root / "raw_inputs/GSE211376_raw_counts.tsv.gz", "rt") as source:
            header = source.readline().rstrip("\n").split("\t")
            cells = header[1:] if header[0] == "" else header
            selected = np.zeros((len(cells), len(patient.genes)), dtype=np.float32)
            library = np.zeros(len(cells))
            detected = np.zeros(len(cells), dtype=int)
            mt_counts = np.zeros(len(cells))
            found = set()
            n_genes = 0
            for line in source:
                gene, values = line.rstrip("\n").split("\t", 1)
                counts = np.fromstring(values, sep="\t", dtype=np.float64)
                valid = np.isfinite(counts) & (counts == np.rint(counts)) & (counts >= 0)
                if len(counts) != len(cells) or not valid.all():
                    raise ValueError(f"GSE211376 计数行不完整或数值非法：{gene}")
                library += counts
                detected += counts > 0
                n_genes += 1
                if gene.upper().startswith("MT-"):
                    mt_counts += counts
                if gene in found:
                    raise ValueError(f"GSE211376 面板基因重复：{gene}")
                if gene in patient.genes:
                    selected[:, patient.genes.index(gene)] = counts
                    found.add(gene)

        # 按计数表的细胞顺序对齐患者注释。GEO 来源没有独立存储的 X。
        cell_table = metadata_table.reindex(cells).copy()
        if cell_table["patient"].isna().any() or found != set(patient.genes):
            raise ValueError("GSE211376 缺少患者标识或面板基因")
        cell_table["cell_id"] = cells
        cell_table["cohort"] = "GSE211376"
        cell_table["region"] = "Tumor"
        original_labels = cell_table["predicted.high_hierarchy"]
        cell_table["primary_state"] = original_labels.where(original_labels.isin(patient.states), "Excluded")
        cell_table["malignant_pool"] = cell_table["primary_state"].isin(patient.states)
        mt_percent = 100 * mt_counts / library
        stored = np.log1p(selected / library[:, None] * 1e4).astype(np.float32)
        keep = np.ones(len(cell_table), dtype=bool)
        scope = "public count matrix"
        source_name = "GSE211376 GEO counts"

    # 核对每个来源后再按 GEO 编号拆分。缓存保留与矩阵相同的细胞行序。
    if np.any(library <= 0):
        raise ValueError(f"存在文库总数为零的细胞：{source_name}")
    cell_table["count_library"] = library
    cell_table["count_n_genes"] = detected
    cell_table["count_mt_pct"] = mt_percent
    cell_table = cell_table.loc[keep].reset_index(drop=True)
    selected = selected[keep]
    stored = stored[keep]
    rebuilt = np.log1p(selected / library[keep, None] * 1e4)
    difference = np.abs(rebuilt - stored)

    # 发表表以 0 记录 GEO 输入缺少独立 X；05a 只对有存储 X 的三个对象检验重建误差。
    audit_rows.append(
        {
            "object": source_name,
            "n_cells": len(cell_table),
            "n_full_genes": n_genes,
            "nonnegative_integer_counts": True,
            "max_abs_X_reconstruction_error": float(difference.max()) if filename != "GSE211376" else 0.0,
            "mean_abs_X_reconstruction_error": float(difference.mean()) if filename != "GSE211376" else 0.0,
            "input_scope": scope,
        }
    )
    for cohort, cohort_table in cell_table.groupby("cohort", sort=True):
        positions = cohort_table.index.to_numpy()
        cell_ids = cohort_table["cell_id"].to_numpy(dtype=str)
        cohort_table.to_csv(cache / f"01_{cohort}细胞.csv.gz", index=False)
        matrix_path = cache / f"02_{cohort}表达.npz"
        np.savez_compressed(
            matrix_path, counts=selected[positions], stored=stored[positions], genes=patient.genes, cell_ids=cell_ids
        )
        saved_table, saved_counts, saved_expression = patient.read_cohort_cache(cache, cohort)
        if not np.array_equal(saved_table["cell_id"].to_numpy(), cell_ids):
            raise ValueError(f"细胞元数据与输入细胞编号顺序不同：{cohort}")
        counts_match = np.array_equal(saved_counts, selected[positions])
        if not counts_match or not np.array_equal(saved_expression, stored[positions]):
            raise ValueError(f"写入缓存的计数或表达与来源矩阵不同：{cohort}")
        cache_rows.append({"object": source_name, "cohort": cohort, "n_cells": len(saved_table)})

# 01b 逐项读取本次成功提取的队列；现有缓存文件的个数不能单独证明所有队列都已处理。
cache_table = pd.DataFrame(cache_rows)
if cache_table["cohort"].duplicated().any():
    raise ValueError("同一队列出现在多个来源对象中")
catalog = cache / "03_队列细胞清单.csv"
cache_table.to_csv(catalog, index=False)
pd.testing.assert_frame_equal(pd.read_csv(catalog), cache_table)

audit_table = pd.DataFrame(audit_rows)
patient.write_table("count_input_audit", audit_table)
print("输入来源", len(audit_table), "个，保留细胞", audit_table["n_cells"].sum(), "个")
