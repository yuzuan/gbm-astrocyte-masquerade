# 目的：在 GSE211376 用标志基因确定发送细胞，比较按 +7/−10 阈值去除 CNV 阳性细胞前后的配受体得分。
# 输入：data/external/scGBM/adata2.h5ad 的 var 基因名和 chromosome，以及
# data/intermediate/GSE211376_raw_counts.tsv.gz（首行为细胞编号，后续每行一个基因及各细胞原始计数）。
# 输出：03_result/09_GSE211376发送细胞筛选/01_筛选前后通讯得分.json
# 运行方式：在仓库根目录用项目 Python 执行 python 09_GSE211376发送细胞筛选.py，沿用 HDF5 和 NumPy 流式计数读取。

import gzip
from importlib import import_module
from pathlib import Path

import numpy as np

communication = import_module("00a_细胞通讯")
upstream = import_module("00a_基因表达与聚类")

# 染色体和谱系基因与前面的 CNV 分析共用定义。此队列的免疫组另外纳入 C1QA、C1QB。
gene_sets = upstream.chromosome_gene_sets()
gene_sets["astro"] = set(upstream.lineage_genes["astro"])
gene_sets["immune"] = set(upstream.lineage_genes["imm"]) | {"C1QA", "C1QB"}
gene_sets["oligo"] = set(upstream.lineage_genes["oligo"])

# 五对配受体沿用历史分析；总文库计数使用全部基因，只有下游要用的配受体保存逐细胞向量。
pairs = [("TGFB2", "TGFBR1"), ("TGFB2", "TGFBR2"), ("TGFB1", "TGFBR1"), ("ANXA1", "FPR1"), ("CSF1", "CSF1R")]
pair_genes = set()
for ligand, receptor in pairs:
    pair_genes.update((ligand, receptor))
with gzip.open("data/intermediate/GSE211376_raw_counts.tsv.gz", "rt", encoding="utf-8") as stream:
    cells = communication.count_matrix_cells(stream)
    total_counts = np.zeros(len(cells))
    group_sums = {name: np.zeros(len(cells)) for name in gene_sets}
    gene_counts = dict.fromkeys(gene_sets, 0)
    pair_counts = {}

    # 同一基因可以同时属于染色体和标志基因组。各组都按计数文件的原始顺序累加。
    for line in stream:
        gene, counts = communication.read_gene_counts(line, len(cells))
        total_counts += counts
        for name, members in gene_sets.items():
            if gene in members:
                group_sums[name] += counts
                gene_counts[name] += 1
        if gene in pair_genes:
            pair_counts[gene] = counts

# 每个细胞先做每万计数归一化，再除以该组实际读到的基因数并取 log1p；这是组内计数均值的对数。
size_factors = communication.library_size_factors(total_counts)
marker_scores = {}
for name, summed_counts in group_sums.items():
    if gene_counts[name] == 0:
        raise ValueError(f"计数表没有 {name} 基因，不能确定细胞群或 CNV 分数。")
    marker_scores[name] = np.log1p(summed_counts / size_factors / gene_counts[name])

# 原方案要求星形分数高于上四分位、免疫分数不高于中位数、少突分数小于 1。髓系接收群取免疫分数最高的一端。
astro = marker_scores["astro"] > np.quantile(marker_scores["astro"], 0.75)
astro &= marker_scores["immune"] <= np.quantile(marker_scores["immune"], 0.5)
astro &= marker_scores["oligo"] < 1.0
myeloid = marker_scores["immune"] > np.quantile(marker_scores["immune"], 0.90)
if not myeloid.any():
    raise ValueError("免疫高分参考群为空，无法估计 CNV 阈值。")

# 分别扣除髓系参考群的 chr7、chr10 均值，再取二者的差。等于参考群第 95 百分位的发送细胞保留。
cnv_score, threshold, _, _ = upstream.centered_cnv(marker_scores["chr7"], marker_scores["chr10"], myeloid, myeloid)
clean = astro & (cnv_score <= threshold)

# 每个配受体基因单独做 log1p，每对用发送群配体均值乘同一个髓系受体均值。缺基因或空组保留缺失。
expression = {gene: np.log1p(counts / size_factors) for gene, counts in pair_counts.items()}
pair_scores = {}
for ligand, receptor in pairs:
    pair_name = f"{ligand}->{receptor}"
    record = {"before": None, "after": None, "pct_change": None}
    if ligand in expression and receptor in expression and astro.any():
        receptor_mean = expression[receptor][myeloid].mean()
        before = float(expression[ligand][astro].mean() * receptor_mean)
        record["before"] = round(before, 4)
        if clean.any():
            after = float(expression[ligand][clean].mean() * receptor_mean)
            record["after"] = round(after, 4)
            record["pct_change"] = round(100 * (after - before) / before, 1) if before else 0.0
    pair_scores[pair_name] = record

# 对照已删除的发送细胞数，并检查每一对都占一个位置，包括缺测项。
sender_count = int(astro.sum())
clean_count = int(clean.sum())
if sender_count - clean_count != int((astro & (cnv_score > threshold)).sum()) or len(pair_scores) != len(pairs):
    raise ValueError("发送细胞移除数或配受体记录数不符。")
result = {"dataset": "GSE211376", "n_sender": sender_count, "n_clean": clean_count, "pairs": pair_scores}
upstream.write_checked_json(Path("03_result/09_GSE211376发送细胞筛选/01_筛选前后通讯得分.json"), result)
print(f"发送细胞 {sender_count} 个，按 CNV 阈值保留 {clean_count} 个。")
