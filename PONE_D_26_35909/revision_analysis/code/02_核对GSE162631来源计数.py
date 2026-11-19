"""目的　对照 10x 原始计数，核对 GSE162631 保留细胞的面板计数和全基因归一化。
输入　GSE162631_RAW_ROOT 下 R*_T 目录的 matrix.mtx.gz、特征和条码，以及 01a、01b 的结果。
输出　02_来源计数核对 下的 01_肿瘤样本计数核对.csv 至 04_逐患者剔除.csv。
运行　在 revision_analysis 下用 Python 运行本脚本。SciPy 读取 Matrix Market，AnnData 统一重复基因名。
"""

import gzip
import os
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from anndata.utils import make_index_unique
from scipy.io import mmread

patient = import_module("00a_患者表达")
source_root = Path(os.environ["GSE162631_RAW_ROOT"]).expanduser()
cache = patient.result_root() / "01_计数与预处理/09_队列细胞缓存"
cell_table, cached_counts, _ = patient.read_cohort_cache(cache, "GSE162631")

# 一份 10x 矩阵为基因×条码。按该样本特征名和保留对象的细胞条码重新索引。
audit_rows = []
contrast_rows = []
eligibility_rows = []
for folder in sorted(source_root.glob("R*_T")):
    features_path = folder / "features.tsv.gz"
    if not features_path.is_file():
        features_path = folder / "genes.tsv.gz"
    feature_table = pd.read_csv(features_path, sep="\t", header=None)
    genes = make_index_unique(pd.Index(feature_table.iloc[:, 1].astype(str)))
    barcode_table = pd.read_csv(folder / "barcodes.tsv.gz", header=None)
    barcodes = pd.Index(barcode_table.iloc[:, 0].astype(str))
    indices = genes.get_indexer(patient.genes)
    if np.any(indices < 0) or not barcodes.is_unique:
        raise ValueError(f"面板基因不完整或条码重复：{folder.name}")

    # 文库分母和线粒体比例用每个条码的全部基因计算。只在比对面板时取指定行。
    with gzip.open(folder / "matrix.mtx.gz", "rb") as source:
        matrix = mmread(source).tocsc()
    valid_counts = (matrix.data >= 0) & (matrix.data == np.rint(matrix.data))
    if matrix.shape != (len(genes), len(barcodes)) or not valid_counts.all():
        raise ValueError(f"10x 矩阵维数或计数有误：{folder.name}")
    library = np.asarray(matrix.sum(axis=0)).ravel()
    detected = np.asarray((matrix > 0).sum(axis=0)).ravel()
    mt_rows = genes.str.upper().str.startswith("MT-")
    mitochondrial = np.asarray(matrix[mt_rows].sum(axis=0)).ravel()

    # 小队列同一条码可跨样本重复，先选定患者、区域再比对。
    subject = "GSE162631_" + folder.name.split("_")[0]
    tumor_cells = (cell_table["patient"] == subject) & (cell_table["region"] == "Tumor")
    subset_table = cell_table.loc[tumor_cells].copy()
    positions = subset_table.index.to_numpy()
    barcode_index = barcodes.get_indexer(subset_table["cell_id"].str.rsplit("_", n=1).str[-1])
    if subset_table.empty or np.any(barcode_index < 0):
        raise ValueError(f"原始样本与保留细胞条码不符：{folder.name}")
    counts = matrix[indices, :][:, barcode_index].T.toarray().astype(np.float32)
    disagreement = np.abs(counts - cached_counts[positions]).max()
    if disagreement != 0:
        raise ValueError(f"面板基因与缓存计数不一致：{folder.name}")

    # 在全基因分母上重算状态，同时保留原始恶性注释一致的子集作为第二个比较。
    retained_library = subset_table["count_library"].to_numpy()
    subset_table["count_library"] = library[barcode_index]
    subset_table["count_n_genes"] = detected[barcode_index]
    subset_table["count_mt_pct"] = 100 * mitochondrial[barcode_index] / library[barcode_index]
    subset_table = subset_table.reset_index(drop=True)
    expression = np.log1p(counts / library[barcode_index, None] * 1e4).astype(np.float32)
    labels, _ = patient.assign_states(expression)
    malignant = subset_table["malignant_pool"].to_numpy()
    labels[~malignant] = "Excluded"
    original_tumor = subset_table["cell_type_original"].isin(["Tumor", "Prol.tumor"]).to_numpy()
    restricted = labels.copy()
    restricted[~original_tumor] = "Excluded"
    for scenario, selected in [
        ("source_full_gene_10k", labels),
        ("source_original_tumor_concordant", restricted),
    ]:
        contrasts, eligibility = patient.summarize_patients(
            subset_table, expression, counts, selected, "uniform", scenario
        )
        contrast_rows.extend(contrasts)
        eligibility_rows.extend(eligibility)

    ratio = library[barcode_index] / retained_library
    audit_rows.append(
        {
            "sample": folder.name,
            "source_cells": len(barcodes),
            "source_genes": len(genes),
            "retained_cells": len(subset_table),
            "retained_fraction": len(subset_table) / len(barcodes),
            "malignant_pool_cells": int(malignant.sum()),
            "max_panel_count_disagreement": float(disagreement),
            "median_full_over_retained_library_ratio": float(np.median(ratio)),
            "max_full_over_retained_library_ratio": float(np.max(ratio)),
            "n_original_tumor_concordant": int((original_tumor & malignant).sum()),
        }
    )

# 确认每个有肿瘤细胞的患者都读到了来源文件。空文件夹或少一个样本都会在此报错。
expected = set(cell_table.loc[cell_table["region"] == "Tumor", "patient"].unique())
observed = {"GSE162631_" + row["sample"].split("_")[0] for row in audit_rows}
if observed != expected:
    raise ValueError("读取的肿瘤样本与保留细胞表的患者集合不一致")
patient.write_table("GSE162631_public_count_audit", pd.DataFrame(audit_rows))
patient.write_table("GSE162631_source_sensitivity", pd.DataFrame(contrast_rows))
patient.write_table("GSE162631_source_eligibility", pd.DataFrame(eligibility_rows))

# 对基准情景逐一剔除患者，记录其余患者的描述性结果。
contrast_table = patient.read_table("preprocessing_patient_contrasts")
baseline_table = contrast_table.query(
    "scenario == 'stored_X' and labels == 'uniform' and cohort == 'GSE162631'"
)
omission_rows = []
for subject in baseline_table["patient"]:
    remaining = baseline_table.loc[baseline_table["patient"] != subject, "difference"]
    omission_rows.append(
        {
            "omitted": subject,
            "n": len(remaining),
            "median_difference": remaining.median(),
            "mean_difference": remaining.mean(),
            "positive_patients": int((remaining > 0).sum()),
            "inference": f"descriptive n={len(remaining)}; no p value",
        }
    )
patient.write_table("GSE162631_leave_patient_out", pd.DataFrame(omission_rows))
print("核对肿瘤样本", len(audit_rows), "份，逐患者剔除", len(omission_rows), "次")
