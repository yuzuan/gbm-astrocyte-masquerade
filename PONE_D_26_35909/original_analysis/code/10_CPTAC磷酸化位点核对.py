#!/usr/bin/env python3
# 目的：核实 CPTAC-GBM 清洁磷酸化矩阵是否收录 TGFB2/TGFBR/SMAD 轴及 SMAD 经典激活位点。
# 输入：phosphoproteome_mssm_per_gene_clean.v4.0.tsv.gz，按 gene、site 和原分析的列分母核对。
# 输出：默认在 external_validation/10_CPTAC磷酸化位点核对 写 01_磷酸化位点覆盖.csv、02_磷酸化位点摘要.json。
# 运行方式：python 10_CPTAC磷酸化位点核对.py --input-file 本地矩阵.tsv.gz [--output-dir 复算目录]
# Python 延续原分块读表；可用 --input-dir 指定 Proteome 目录，或用 --input-url 指定原表 HTTPS 直链。

import argparse
import re
from importlib import import_module
from pathlib import Path

import pandas as pd

spatial = import_module("00a_空间分析")
files = import_module("00a_项目路径")


def main():
    """从磷酸化表逐行核实覆盖；找不到唯一基因列或位点列时停止，无返回值。"""
    external_dir = Path(__file__).resolve().parents[1] / "external_validation"
    parser = argparse.ArgumentParser(description="CPTAC-GBM 磷酸化位点覆盖")
    local = parser.add_mutually_exclusive_group()
    local.add_argument("--input-file", type=Path, help="清洁磷酸化矩阵")
    local.add_argument("--input-dir", type=Path, default=external_dir / "raw/cptac_gbm/Proteome")
    parser.add_argument("--input-url", help="本地无表时使用的 HTTPS 直链")
    parser.add_argument("--output-dir", type=Path, default=external_dir / Path(__file__).stem)
    args = parser.parse_args()

    # 先用本地表，再使用同来源缓存。网络取回的压缩表须完整且含 gene 表头。
    local_path = args.input_file or (args.input_dir / "phosphoproteome_mssm_per_gene_clean.v4.0.tsv.gz")
    path = spatial.cptac_input(local_path, args.input_url, args.output_dir / "raw", ["gene"])
    header_table = pd.read_csv(path, sep="\t", nrows=0)
    gene_names = [name for name in header_table if name.lower() == "gene"]
    site_names = [name for name in header_table if name.lower() in {"site", "phosphosite", "phospho_site"}]
    if len(gene_names) != 1 or len(site_names) != 1:
        raise ValueError("磷酸化表须各有一个可识别的 gene 和 site 列")
    gene_column, site_column = gene_names[0], site_names[0]

    # 保留已发表 total_sample_columns 的历史定义。v4.0 的 refseq_id 留在分母内，
    # 该列实际存放序列标识，数值转换后缺失；这个分母比真实样本列数多一。
    annotations = {gene_column, site_column, "Reference Sequence", "Peptide", "Sequence", "refseq", "peptide"}
    historical_columns = [name for name in header_table if name not in annotations]
    target_genes = ["SMAD2", "SMAD3", "SMAD4", "TGFB2", "TGFBR1", "TGFBR2"]
    selected_tables = []
    for chunk_table in pd.read_csv(path, sep="\t", chunksize=50_000, low_memory=False):
        genes = chunk_table[gene_column].astype(str).str.upper()
        keep = genes.isin(target_genes)
        if keep.any():
            selected_tables.append(chunk_table.loc[keep].copy())
    target_table_total = pd.concat(selected_tables, ignore_index=True) if selected_tables else header_table.copy()

    # 一行表示一个原表位点。未收录基因用空位点行表示；此时经典标记只说明该基因属于核查目标。
    canonical_sites = {"SMAD2": {"S465", "S467"}, "SMAD3": {"S423", "S425"}}
    records = []
    target_symbols = target_table_total[gene_column].astype(str).str.upper()
    for gene in target_genes:
        gene_table = target_table_total.loc[target_symbols == gene]
        if gene_table.empty:
            records.append({
                "gene": gene, "site": "", "row_present": False, "nonmissing_samples": 0,
                "total_sample_columns": len(historical_columns), "canonical_activation_site": gene in canonical_sites,
            })
            continue

        # 从源标签取完整氨基酸和位置，避免把 S4650 当作 S465；是否收录与是否有定量分别保存。
        for _, site_row in gene_table.iterrows():
            site = str(site_row[site_column])
            residues = set(re.findall(r"[STY][0-9]+", site.upper()))
            quantified = pd.to_numeric(site_row[historical_columns], errors="coerce")
            records.append({
                "gene": gene, "site": site, "row_present": True, "nonmissing_samples": int(quantified.notna().sum()),
                "total_sample_columns": len(historical_columns),
                "canonical_activation_site": bool(residues & canonical_sites.get(gene, set())),
            })
    coverage_table = pd.DataFrame(records).sort_values(["gene", "row_present", "site"], ascending=[True, False, True])

    # 原始目标行应逐行保留，缺基因也须留行。经典行必须同时满足目标基因、位点匹配和真实存在。
    present_count = int(coverage_table["row_present"].sum())
    if present_count != len(target_table_total) or set(coverage_table["gene"]) != set(target_genes):
        raise ValueError("位点表没有完整保留目标原始行或缺失基因")
    if not coverage_table["nonmissing_samples"].between(0, len(historical_columns)).all():
        raise ValueError("定量列数超过原分析分母")
    files.write_table(coverage_table, args.output_dir / "01_磷酸化位点覆盖.csv", ["gene", "site"])
    is_canonical = coverage_table["gene"].isin(canonical_sites) & coverage_table["canonical_activation_site"]
    present_sites = is_canonical & coverage_table["row_present"]
    detected_table = coverage_table.loc[present_sites, ["gene", "site", "nonmissing_samples"]]

    # 已发表的 available 字段表示矩阵中有对应行；通路活化需要独立的定量与比较证据。
    if detected_table.empty:
        interpretation = (
            "Canonical SMAD2 S465/S467 and SMAD3 S423/S425 activation sites are "
            "not represented in the clean CPTAC GBM phosphoproteome matrix; "
            "downstream TGF-beta pathway activation cannot be validated from this file."
        )
    else:
        interpretation = (
            "Canonical SMAD activation site rows are present. Quantification coverage is listed per site; "
            "site representation alone does not establish downstream pathway activation."
        )
    summary = {
        "source_file": path.name, "sample_columns": len(historical_columns), "target_genes": target_genes,
        "canonical_sites_requested": {gene: sorted(sites) for gene, sites in canonical_sites.items()},
        "canonical_sites_detected": detected_table.to_dict(orient="records"),
        "canonical_smad_activation_available": not detected_table.empty, "interpretation": interpretation,
    }
    files.write_json(summary, args.output_dir / "02_磷酸化位点摘要.json")
    print(f"目标基因 {len(target_genes)} 个，矩阵内目标位点 {len(target_table_total)} 行，经典 SMAD 位点 {len(detected_table)} 行")


if __name__ == "__main__":
    main()
