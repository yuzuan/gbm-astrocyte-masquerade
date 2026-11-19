# ==============================================================================
# 00a_患者表达.py —— PONE-D-26-35909 返修公共函数库
# 返修患者表达及随机效应分析。
#
# 目的：
# 统一表达标志、细胞与矩阵的对应关系、逐患者统计及返修表的读写。
# 输入的患者号只在各自队列内唯一。
# 阅读分析脚本时遇到相应调用，再查阅本文件。
# 可用编辑器中的 Ctrl + 单击（Mac 使用 Cmd + 单击）跳转到函数定义。
# 函数按调用定位。

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


# 保留投稿补充数据的英文表标识，以同一张表对应的中文路径读写本次运行的结果。
table_files = {
    "count_input_audit": "01_计数与预处理/01_计数输入核对.csv",
    "preprocessing_patient_contrasts": "01_计数与预处理/02_患者表达差.csv",
    "preprocessing_eligibility": "01_计数与预处理/03_患者纳入情况.csv",
    "preprocessing_cell_membership": "01_计数与预处理/04_纳入细胞数.csv",
    "GSE162631_state_audit": "01_计数与预处理/05_GSE162631细胞状态.csv",
    "GSE162631_annotation_crosswalk": "01_计数与预处理/06_GSE162631注释对应.csv",
    "preprocessing_meta_stouffer": "01_计数与预处理/07_队列合并检验.csv",
    "preprocessing_cohort_summary": "01_计数与预处理/08_逐队列检验.csv",
    "GSE162631_public_count_audit": "02_来源计数核对/01_肿瘤样本计数核对.csv",
    "GSE162631_source_sensitivity": "02_来源计数核对/02_来源计数患者表达差.csv",
    "GSE162631_source_eligibility": "02_来源计数核对/03_来源计数纳入情况.csv",
    "GSE162631_leave_patient_out": "02_来源计数核对/04_逐患者剔除.csv",
    "meta_patient_inputs": "03_患者模型输入/01_模型患者表达差.csv",
    "meta_cohort_effects": "04_随机效应模型/01_队列效应量.csv",
    "meta_summary": "04_随机效应模型/02_合并效应量.csv",
    "meta_leave_one_out": "04_随机效应模型/03_逐队列剔除.csv",
}


def result_root():
    """从调用时的环境读取结果目录，便于同一会话在新目录中复现。"""
    default_root = Path(__file__).resolve().parents[1] / "03_result"
    directory = Path(os.environ.get("PLOS_REVISION_DIR", default_root)).expanduser()
    return directory


def read_table(name):
    """按发表表标识读回本次运行的结果。缺表时显示实际路径。"""
    result_table = pd.read_csv(result_root() / table_files[name])
    return result_table


def write_table(name, result_table):
    """逐表建立目录并回读表头和维数。无返回值。"""
    path = result_root() / table_files[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    result_table.to_csv(path, index=False)
    saved_table = pd.read_csv(path)
    if saved_table.shape != result_table.shape or list(saved_table) != list(result_table):
        raise ValueError(f"结果表行列与写入内容不符：{path.name}")


def patient_from_tumor_sample(sample):
    """由本研究来源样本或公开样本编号取得患者号。检测到未知编码时给出具体样本。"""
    public = re.fullmatch(r"(P[1-9]\d*)-S1", sample)
    if public:
        subject = public.group(1)
        return subject
    source = re.fullmatch(r"(R[1-9]\d*)_T", sample)
    if source is None:
        raise ValueError(f"无法从肿瘤样本编号识别患者：{sample}")
    subject = "GSE162631_" + source.group(1)
    return subject


def read_cohort_cache(cache, cohort):
    """按细胞号和面板基因顺序配对注释、计数和表达。两份文件中任一维度不匹配即停止。"""
    cell_table = pd.read_csv(cache / f"01_{cohort}细胞.csv.gz", dtype={"cell_id": str})
    if cell_table.empty or not cell_table["cohort"].eq(cohort).all():
        raise ValueError(f"缓存中的细胞队列有误：{cohort}")
    ids = cell_table["cell_id"]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError(f"细胞编号缺失或重复：{cohort}")

    with np.load(cache / f"02_{cohort}表达.npz", allow_pickle=False) as saved:
        counts = saved["counts"]
        stored = saved["stored"]
        expected_shape = (len(ids), len(genes))
        if counts.shape != expected_shape or stored.shape != expected_shape:
            raise ValueError(f"细胞表与两份表达矩阵的维数不符：{cohort}")
        if not np.array_equal(saved["cell_ids"], ids.to_numpy()) or saved["genes"].tolist() != genes:
            raise ValueError(f"细胞顺序或面板基因顺序不同：{cohort}")
    result = cell_table, counts, stored
    return result


# 标志顺序定义四个恶性状态的得分列。TGFB2 是待比较基因，不能参与打分。
state_markers = {
    "AC-like": ["GFAP", "S100B", "HOPX", "SLC1A3", "MLC1"],
    "MES-like": ["VIM", "CD44", "CHI3L1", "HILPDA", "DDIT3", "ENO2"],
    "OPC-like": ["PLP1", "OLIG1", "OMG", "TNR", "ALCAM"],
    "NPC-like": ["SOX4", "SOX11", "DCX", "CD24", "STMN1", "STMN2"],
}
lineage_markers = {
    "Endothelial": ["PECAM1", "VWF", "CLDN5", "KDR"],
    "Myeloid": ["PTPRC", "LST1", "TYROBP"],
    "Oligodendrocyte": ["MBP", "MOG", "MOBP"],
}
states = list(state_markers)
marker_genes = sum(state_markers.values(), []) + sum(lineage_markers.values(), [])
genes = list(dict.fromkeys(["TGFB2", *marker_genes]))


def assign_states(expression):
    """输入是细胞×面板基因的矩阵，返回各细胞的状态和最高两项得分之差。"""
    state_scores = []
    for markers in state_markers.values():
        columns = [genes.index(gene) for gene in markers]
        state_scores.append(expression[:, columns].mean(axis=1))

    # 沿状态轴找前两项，沿细胞轴取对应得分。0.05 是原分析的最小分差。
    scores = np.column_stack(state_scores)
    order = np.argsort(scores, axis=1)
    first = order[:, -1]
    rows = np.arange(len(expression))
    margin = scores[rows, first] - scores[rows, order[:, -2]]
    labels = np.array(states, dtype=object)[first]
    labels[margin < 0.05] = "Ambiguous"
    result = labels, margin
    return result


def summarize_patients(cell_table, expression, counts, labels, label_method, scenario, pseudobulk=False):
    """返回患者内 TGFB2 表达差记录和包含未达细胞数门槛的纳入记录。每次输入一个队列。"""
    cohort = cell_table["cohort"].iat[0]
    region = "Core" if cohort == "GSE274987" else "Tumor"
    annotated_table = cell_table.assign(state=labels)
    contrast_rows = []
    eligibility_rows = []
    gene_index = genes.index("TGFB2")

    # 保留细胞表原索引来取矩阵行。原始注释要求每个参照状态至少20个细胞，统一注释合并参照。
    for patient, patient_table in annotated_table.loc[annotated_table["region"] == region].groupby("patient"):
        state_counts = patient_table["state"].value_counts()
        other_states = states[1:]
        if label_method == "primary":
            other_states = [state for state in other_states if state_counts.get(state, 0) >= 20]
        ac = patient_table.index[patient_table["state"] == "AC-like"].to_numpy()
        other = patient_table.index[patient_table["state"].isin(other_states)].to_numpy()
        eligible = len(ac) >= 20 and len(other) >= 20
        identity = {
            "scenario": scenario,
            "labels": label_method,
            "cohort": cohort,
            "patient": patient,
            "region": region,
            "n_ac": len(ac),
            "n_other": len(other),
        }
        eligibility_rows.append({**identity, "eligible": eligible})
        if not eligible:
            continue

        # 细胞均值与汇总计数对应不同表达尺度，保留各自原定分母和伪计数。
        if pseudobulk:
            ac_value = np.log2(
                (counts[ac, gene_index].sum() + 0.5) / (cell_table.loc[ac, "count_library"].sum() + 1) * 1e6
            )
            other_value = np.log2(
                (counts[other, gene_index].sum() + 0.5)
                / (cell_table.loc[other, "count_library"].sum() + 1)
                * 1e6
            )
            scale = "pseudobulk log2 CPM"
        else:
            ac_value = float(expression[ac, gene_index].mean())
            other_value = float(expression[other, gene_index].mean())
            scale = "mean log1p counts per 1000" if scenario == "counts_1k" else "mean log1p counts per 10000"

        # 表内一行是一名患者的两类细胞。阳性比例来自计数，文库/基因数/线粒体比例来自同一批细胞。
        contrast_rows.append(
            {
                **identity,
                "ac_mean": ac_value,
                "other_mean": other_value,
                "difference": ac_value - other_value,
                "ac_detected_fraction": float(np.mean(counts[ac, gene_index] > 0)),
                "other_detected_fraction": float(np.mean(counts[other, gene_index] > 0)),
                "ac_median_library": float(np.median(cell_table.loc[ac, "count_library"])),
                "other_median_library": float(np.median(cell_table.loc[other, "count_library"])),
                "ac_median_genes": float(np.median(cell_table.loc[ac, "count_n_genes"])),
                "other_median_genes": float(np.median(cell_table.loc[other, "count_n_genes"])),
                "ac_median_mt_pct": float(np.median(cell_table.loc[ac, "count_mt_pct"])),
                "other_median_mt_pct": float(np.median(cell_table.loc[other, "count_mt_pct"])),
                "scale": scale,
            }
        )

    result = contrast_rows, eligibility_rows
    return result
