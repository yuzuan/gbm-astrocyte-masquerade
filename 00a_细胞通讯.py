# ======================================================================
# 00a_细胞通讯.py —— GBM 星形样细胞分析公共函数库
# 根目录历史分析，供模块 07 至 13 使用。
#
# 目的：
# 共用同一套 scGBM 分组、抽样规则、表达矩阵读取与结果写出方式。
# 函数改变时，需要检查调用它的全部分析脚本。
# 本函数库可在阅读具体分析的调用处时再查阅。
# 在编辑器中用 Ctrl + 左键（Mac 为 Cmd + 左键）点函数名可以跳转到定义。
# 查找函数以编辑器跳转或按名称搜索为准。

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy import sparse


def scgbm_state_masks(minor_cluster):
    """输入 scGBM 细胞亚群列，返回四种发送亚型的布尔向量。NPC 合并原来的两个标签。"""
    masks = {
        "AC": minor_cluster.eq("AC"),
        "MES1": minor_cluster.eq("MES1"),
        "OPC": minor_cluster.eq("OPC"),
        "NPC": minor_cluster.isin(["NPC1", "NPC2"]),
    }
    return masks


def sampled_scgbm_positions(cell_mask):
    """输入按表达矩阵排序的组别向量，返回历史抽样顺序中的细胞行号。"""
    positions = np.flatnonzero(cell_mask)

    # 各组超过 5000 才抽样。每组重新使用历史 MT19937 种子 42，移到循环外会改变入选细胞和均值。
    if len(positions) > 5000:
        generator = np.random.RandomState(42)
        positions = generator.choice(positions, 5000, replace=False)

    return positions


def cell_expression_table(dataset, positions, genes):
    """取 X 中指定细胞和基因，返回行是细胞、列是基因的表。空组、缺基因或重名基因停止。"""
    if not len(positions) or not len(genes):
        raise ValueError("表达均值需要至少一个细胞和一个可用基因。")
    gene_positions = dataset.var_names.get_indexer(genes)
    if (gene_positions < 0).any():
        raise ValueError("表达矩阵缺少请求的基因。")

    # 先取稀疏矩阵的细胞行，再取目标基因列，避免把两组下标误作一一对应。保留抽样行序。
    cell_matrix = dataset.X[positions, :]
    selected_matrix = cell_matrix[:, gene_positions]
    if sparse.issparse(selected_matrix):
        selected_matrix = selected_matrix.toarray(order="F")
    else:
        selected_matrix = np.asfortranarray(selected_matrix)

    # 连续存放每个基因的向量，使 float32 的逐基因均值沿用旧脚本的加法顺序。
    expression_table = pd.DataFrame(selected_matrix, columns=genes)
    return expression_table


def scgbm_product_scores(dataset, receiver):
    """输入 scGBM 对象和接收群标签，返回每对配受体的四种发送亚型得分。未测到的基因对跳过。"""
    if receiver == "TAM/Macrophage":
        pairs = [("TGFB2", "TGFBR1"), ("TGFB2", "TGFBR2"), ("CCL2", "CCR2"), ("CSF1", "CSF1R"), ("CXCL12", "CXCR4")]
    elif receiver == "T/NK":
        pairs = [("CD274", "PDCD1"), ("NECTIN2", "TIGIT"), ("PVR", "TIGIT"), ("PDCD1LG2", "PDCD1")]
    else:
        raise ValueError(f"通讯面板没有定义接收群 {receiver}。")

    # 两端都须出现在基因索引。只取完整配对，未测到与平均表达为零在旧结果中有不同含义。
    available_pairs = []
    for ligand, receptor in pairs:
        if ligand in dataset.var_names and receptor in dataset.var_names:
            available_pairs.append((ligand, receptor))

    scores = {}
    if not available_pairs:
        return scores

    # 同一接收群的受体均值供四种发送亚型共用。每个基因在一维原精度向量上求均值。
    receptors = list(dict.fromkeys(receptor for ligand, receptor in available_pairs))
    positions = sampled_scgbm_positions(dataset.obs["Major_category"].eq(receiver))
    receiver_table = cell_expression_table(dataset, positions, receptors)
    receptor_means = {gene: float(np.mean(receiver_table[gene].to_numpy())) for gene in receptors}

    # 旧发布 JSON 的亚型名与原始对象大小写一致。各亚型独立抽样，乘积使用 Python 浮点数。
    ligands = list(dict.fromkeys(ligand for ligand, receptor in available_pairs))
    for state, mask in scgbm_state_masks(dataset.obs["Minor_cluster"]).items():
        sender_table = cell_expression_table(dataset, sampled_scgbm_positions(mask), ligands)
        for ligand, receptor in available_pairs:
            pair_name = f"{ligand}-{receptor}"
            ligand_mean = float(np.mean(sender_table[ligand].to_numpy()))
            scores.setdefault(pair_name, {})[state] = ligand_mean * receptor_means[receptor]

    return scores


def count_matrix_cells(stream):
    """读取 GSE211376 计数表第一行的全部细胞编号。重复或空编号停止，返回按原顺序排列的索引。"""
    cells = pd.Index(stream.readline().rstrip("\r\n").split("\t"))
    if not cells.is_unique or "" in cells:
        raise ValueError("计数表首行应为唯一的细胞编号，每列一个。")
    return cells


def read_gene_counts(line, cell_count):
    """解析一行基因计数，返回基因名和全体细胞的向量。列数、计数值或基因名异常时停止。"""
    gene, separator, values = line.partition("\t")
    counts = np.fromstring(values, sep="\t")
    if not gene or not separator or len(counts) != cell_count:
        raise ValueError(f"计数行 {gene!r} 与表头的 {cell_count} 个细胞不匹配。")
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError(f"基因 {gene} 的原始计数有缺失或负值。")
    result = gene, counts
    return result


def library_size_factors(total_counts):
    """输入每个细胞的全基因计数，返回每万计数的除数。零文库沿用历史极小正除数。"""
    factors = total_counts / 1e4
    factors[factors == 0] = 1e-9
    return factors


def write_gene_table(path, result_table, expected_genes):
    """写出每基因一行的 CSV，无返回值。核对来源基因顺序与唯一性，再回读所有列和数值。"""
    if not result_table["gene"].is_unique or result_table["gene"].tolist() != list(expected_genes):
        raise ValueError("输出表的基因、顺序或行数与输入面板不符。")

    # 临时文件与目标在同一目录，读者只会看到已核过的完整表。保留分类文字 NA 和真正的空数值。
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_path.parent) as folder:
        pending_path = Path(folder) / output_path.name
        result_table.to_csv(pending_path, index=False, encoding="utf-8-sig")
        saved_table = pd.read_csv(pending_path, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
        pd.testing.assert_frame_equal(
            saved_table, result_table.reset_index(drop=True), check_dtype=False, check_exact=False,
            rtol=1e-7, atol=1e-12,
        )
        pending_path.replace(output_path)
