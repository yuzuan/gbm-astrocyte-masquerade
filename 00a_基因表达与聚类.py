# ======================================================================
# 00a_基因表达与聚类.py —— GBM 星形胶质细胞项目公共函数库
# 服务于根目录 01–06 的历史 CNV 分析。
#
# 目的：
# 统一几份分析共用的基因集合、逐细胞计算和产物读写。
# 细胞筛选条件和各队列的统计分母在对应步骤中给出。
# 可在阅读分析脚本遇到调用时再读相应函数。
# 在编辑器中按 Ctrl（Mac 上按 Cmd）并点击函数名可跳至定义，也可搜索函数名。
# 按调用位置查找定义即可。

import gzip
import json
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import pandas as pd
from scipy.io import mminfo, mmread


def chromosome_gene_sets():
    """从 scGBM 的 var 分类编码读取两条染色体的基因名。注释缺项会报错。"""
    with h5py.File("data/external/scGBM/adata2.h5ad", "r") as reference:
        genes = reference["var/_index"].asstr()[:]
        labels = reference["var/chromosome/categories"].asstr()[:].tolist()
        codes = reference["var/chromosome/codes"][:]

    # 缺失分类的编码是 -1。直接比较对应编码，就不会把缺失误取成最后一条染色体。
    gene_sets = {}
    for chromosome in ("chr7", "chr10"):
        gene_sets[chromosome] = set(genes[codes == labels.index(chromosome)])
    if gene_sets["chr7"] & gene_sets["chr10"]:
        raise ValueError("同一基因同时标到 chr7 和 chr10，请核对 scGBM 注释。")
    return gene_sets


# GSE211376 和 GSE174554 共用这些谱系标志。GSE211376 的髓系面板在 03 另加两个补体基因。
lineage_genes = {
    "astro": ["GFAP", "AQP4", "SLC1A3", "S100B", "ALDOC", "CLU"],
    "imm": ["PTPRC", "CD68", "CD14", "AIF1", "ITGAM", "CSF1R"],
    "oligo": ["MOG", "PLP1", "MBP", "MOBP"],
}


def gse211376_expression(gene_sets, count_dtype):
    """流式读取基因×细胞计数，返回细胞×基因组得分表。行宽、条码或基因组有问题时停止。"""
    with gzip.open("data/intermediate/GSE211376_raw_counts.tsv.gz", "rt", encoding="utf-8") as counts:
        header = counts.readline().rstrip("\r\n").split("\t")
        first_gene = next(counts)
        cell_count = first_gene.count("\t")

        # GEO 原表头只有条码；带额外基因列标题的 TSV 也以首行计数的宽度识别。
        if len(header) == cell_count + 1:
            header = header[1:]
        if len(header) != cell_count or not cell_count or "" in header or len(set(header)) != cell_count:
            raise ValueError("GSE211376 计数表的条码重复、为空，或与首行计数列数不符。")
        library_counts = np.zeros(cell_count)
        group_counts = {name: np.zeros(cell_count) for name in gene_sets}
        matched_genes = dict.fromkeys(gene_sets, 0)

        # 细胞文库大小累加全部基因。一个基因可以同时属于染色体和谱系面板。
        for line_number, line in enumerate(chain([first_gene], counts), start=2):
            gene, _, count_text = line.partition("\t")
            values = np.fromstring(count_text, sep="\t", dtype=count_dtype)
            if values.size != cell_count or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"GSE211376 第 {line_number} 行（{gene}）计数不全或有非法值。")
            library_counts += values
            for name, genes in gene_sets.items():
                if gene in genes:
                    group_counts[name] += values
                    matched_genes[name] += 1

    # 每细胞先换算成每万计数，再除以实际匹配基因数，最后 log1p。全零文库沿用原来的极小除数。
    library_scale = library_counts / 1e4
    library_scale[library_scale == 0] = 1e-9
    expression_table = pd.DataFrame(index=pd.Index(header, name="cell_id"))
    for name, values in group_counts.items():
        if matched_genes[name] == 0:
            raise ValueError(f"GSE211376 计数表没有 {name} 组的任何基因。")
        expression_table[name] = np.log1p(values / library_scale / matched_genes[name])
    return expression_table


def gse211376_metadata(cell_ids):
    """将每行一个条码的 GEO 元数据排成计数矩阵的次序。条码不唯一或缺少注释时停止。"""
    metadata_table = pd.read_csv("data/intermediate/GSE211376_metadata.csv.gz", index_col=0)
    if not metadata_table.index.is_unique:
        raise ValueError("GSE211376 元数据出现重复细胞条码。")
    aligned_table = metadata_table.reindex(cell_ids)
    if aligned_table["predicted.high_hierarchy"].isna().any():
        raise ValueError("部分 GSE211376 计数条码缺少作者细胞注释。")
    return aligned_table


def centered_cnv(chr7, chr10, center_mask, threshold_mask):
    """返回中心化的逐细胞 +7/−10 分数、严格阈值和两个参照均值。空参照或非有限分数会报错。"""
    if not np.any(center_mask) or not np.any(threshold_mask):
        raise ValueError("没有可用于染色体中心化或确定 CNV 阈值的参照细胞。")
    center7 = chr7[center_mask].mean()
    center10 = chr10[center_mask].mean()
    scores = (chr7 - center7) - (chr10 - center10)
    if not np.isfinite(scores).all():
        raise ValueError("逐细胞 CNV 分数有缺失或无穷值。")

    # 三份原分析都在参照细胞中取第 95 百分位，比较时使用未取位的阈值。
    threshold = np.percentile(scores[threshold_mask], 95)
    result = scores, threshold, center7, center10
    return result


def positive_summary(positive, mask):
    """按选中的细胞返回分母和一位小数的阳性百分比。空组的百分比写为 JSON null。"""
    selected = np.asarray(positive)[mask]
    percent = None
    if selected.size:
        percent = round(float(100 * selected.mean()), 1)
    result = {"n": int(selected.size), "cnv_pos_pct": percent}
    return result


def inhouse_cnv_summary(score_path, metadata_path):
    """读取自有队列逐细胞评分和条件，返回两种口径的星形细胞结果及参照组比例。缺失或错配时停止。"""
    score_table = pd.read_csv(score_path, index_col=0, dtype={"cnv_pos": "boolean"})
    metadata_table = pd.read_csv(metadata_path, index_col=0)
    if not score_table.index.is_unique or not metadata_table.index.is_unique:
        raise ValueError("自有队列评分表或元数据的细胞条码重复。")
    conditions = metadata_table["condition"].reindex(score_table.index)
    if conditions.isna().any() or score_table[["chr7", "chr10", "cell_type", "cnv_pos"]].isna().any().any():
        raise ValueError("自有队列有未匹配的条件、缺失的评分或细胞类型。")

    # 严格阈值取两个组织条件合并后的 Myeloid 和 T_NK。宽松阳性直接沿用上游评分表的 cnv_pos。
    cell_types = score_table["cell_type"]
    immune = cell_types.isin(["Myeloid", "T_NK"])
    scores = score_table["chr7"] - score_table["chr10"]
    if not immune.any() or not np.isfinite(scores).all():
        raise ValueError("自有队列缺少免疫参照或含非有限 CNV 分数。")
    threshold = np.quantile(scores[immune], 0.95)
    positive = scores > threshold

    # 主要分母仅含 GBM 星形细胞。历史 peritumoral 字段对应元数据的 Normal，组织学位置未单列。
    astrocytes = cell_types == "Astrocyte"
    gbm = astrocytes & conditions.eq("GBM")
    strict = positive_summary(positive, gbm)
    if strict["n"] == 0:
        raise ValueError("自有队列没有 GBM 星形胶质细胞。")
    loose = positive_summary(score_table["cnv_pos"].to_numpy(dtype=bool), gbm)
    normal = positive_summary(positive, astrocytes & conditions.eq("Normal"))

    # 三个参照组都使用各自全部细胞作分母，包括没有参与定阈值的少突胶质细胞。
    reference_groups = {}
    for cell_type in ("Myeloid", "T_NK", "Oligodendrocyte"):
        reference_groups[cell_type] = positive_summary(positive, cell_types == cell_type)
    result = {
        "n": strict["n"],
        "strict_matched_pct": strict["cnv_pos_pct"],
        "loose_cnv_pos_pct": loose["cnv_pos_pct"],
        "strict_ref_FPR": {name: group["cnv_pos_pct"] for name, group in reference_groups.items()},
        "peritumoral_astro_strict_pct": normal["cnv_pos_pct"],
        "strict_threshold": float(threshold),
        "reference_n": {name: group["n"] for name, group in reference_groups.items()},
        "normal_astro_n": normal["n"],
    }
    return result


def feature_names(feature_path):
    """读取一份 RNA 特征表第一列的基因名，保持矩阵行序。文件缺失时停止。"""
    with gzip.open(feature_path, "rt", encoding="utf-8") as features:
        genes = [line.rstrip("\r\n").split("\t")[0] for line in features]
    return genes


def gse174554_libraries():
    """返回共同基因顺序、RNA 矩阵路径及同序细胞表。等长却错位的基因或条码会报错。"""
    directory = Path("data/GSE174554")
    references = sorted(directory.glob("*_features.tsv.gz"))
    if not references:
        raise FileNotFoundError(f"找不到 {directory} 的 RNA 特征表。")
    genes = feature_names(references[0])
    matrices = []
    library_tables = []

    # 复现原文件名排序和损坏文库排除。用矩阵头识别特征数不同的文件（本批为 snATAC）。
    for matrix_path in sorted(directory.glob("*_matrix.mtx.gz")):
        prefix = matrix_path.name.removesuffix("_matrix.mtx.gz")
        if "GSM5319529_SF8963" in prefix:
            continue
        barcode_path = directory / f"{prefix}_barcodes.tsv.gz"
        if not barcode_path.is_file():
            continue
        gene_count, cell_count, *_ = mminfo(str(matrix_path))
        if gene_count != len(genes):
            continue

        # 只有相同长度且逐基因同序的 RNA 矩阵才可以纵向拼接。完整文件前缀能区分同一样本的不同批。
        if feature_names(directory / f"{prefix}_features.tsv.gz") != genes:
            raise ValueError(f"{prefix} 与参照 RNA 特征表的基因次序不同。")
        with gzip.open(barcode_path, "rt", encoding="utf-8") as barcodes:
            barcode_ids = [line.strip() for line in barcodes]
        sample = next((part for part in prefix.split("_") if part.startswith("SF")), prefix)
        library_table = pd.DataFrame({"barcode": barcode_ids, "sample": sample})
        library_table["cell_id"] = prefix + ":" + library_table["barcode"]
        library_table["tissue_id"] = sample + "_" + library_table["barcode"].str.split("-").str[0]
        if len(library_table) != cell_count:
            raise ValueError(f"{prefix} 的条码数与 RNA 矩阵列数不同。")
        library_tables.append(library_table[["cell_id", "sample", "tissue_id"]])
        matrices.append(matrix_path)

    # 保留文库与条码原序。tissue_id 供匹配历史组织注释，完整 cell_id 供新产物逐行核对。
    if not matrices:
        raise ValueError("GSE174554 没有符合历史纳入条件的 RNA 文库。")
    cell_table_total = pd.concat(library_tables, ignore_index=True)
    if not cell_table_total["cell_id"].is_unique:
        raise ValueError("GSE174554 文库前缀与完整条码组成的细胞标识重复。")
    result = np.asarray(genes), matrices, cell_table_total
    return result


def read_cell_matrix(matrix_path):
    """把单文库 Matrix Market 基因×条码计数转换成历史 float32 细胞×基因稀疏矩阵。"""
    gene_matrix = mmread(matrix_path).tocsr()
    cell_matrix = gene_matrix.T.astype(np.float32)
    return cell_matrix


def result_or_frozen(result_path, old_name):
    """依次选取新产物、data/intermediate 冻结文件、旧脚本在根目录写出的文件。均缺失时停止。"""
    for path in (result_path, Path("data/intermediate") / old_name, Path(old_name)):
        if path.is_file():
            return path
    raise FileNotFoundError(f"找不到 {result_path} 或旧文件 {old_name}。")


def write_checked_json(path, result):
    """写 UTF-8 JSON 并回读全部字段和值，非有限数或不一致时停止。无返回值。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temporary:
        draft = Path(temporary) / path.name
        draft.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if json.loads(draft.read_text(encoding="utf-8")) != result:
            raise ValueError(f"{path.name} 回读时字段或数值发生变化。")
        draft.replace(path)


def write_checked_arrays(path, arrays):
    """压缩保存具名数组，回读核对字段、维度、类型和每个值，再发布文件。无返回值。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temporary:
        draft = Path(temporary) / path.name
        np.savez_compressed(draft, **arrays)
        with np.load(draft, allow_pickle=False) as stored:
            if set(stored.files) != set(arrays):
                raise ValueError(f"{path.name} 回读时数组字段缺失或多出。")
            for name, expected in arrays.items():
                observed = stored[name]
                if observed.dtype != expected.dtype or not np.array_equal(observed, expected):
                    raise ValueError(f"{path.name} 的 {name} 数组维度、类型或逐项取值不符。")
        draft.replace(path)
