#!/usr/bin/env python3
# 目的：核对公共处理对象里实际包含的患者、来源和区域，并保存可追溯的上游 notebook 快照。
# 输入：PUBLIC_INPUT_ROOT/processed_inputs 中三份 h5ad 的 obs、X 维度，以及 upstream_source/ 中的八份原 notebook；
# 公开 TableS9b_processed_cohort_membership.csv 提供联合区域队列按来源登记的患者人数。
# 输出：重算包 data/processed_input_objects.csv、data/processed_cohort_membership.csv、
# data/upstream_notebook_snapshots.csv、data/processed_input_audit.json；code/upstream_processing/ 中
# scgbm_00_integrate_two_datasets.ipynb、scgbm_01_preprocess_scRNAdata.ipynb、scgbm_02_marker_gene_legacy.ipynb、
# scgbm_03_CNV_validation_legacy.ipynb、scgbm_04_Sub_clustering_legacy.ipynb、scgbm_06_Combine_subclusters_legacy.ipynb、
# sngbm_01_snRNA_processing.ipynb、gse274987_00_integrate_two_datasets.ipynb 和 README.md。
# 运行方式：在 code 目录运行 python 04_已处理输入核对.py。独立包写入 01_公共队列重算/。

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import anndata as ad
import pandas as pd

paths = import_module("00a_项目路径")


def main() -> None:
    """生成每对象、每对象来源和每 notebook 各一行的三张发表清单。无返回值。"""
    object_table, membership_table, assertions = audit_objects()
    notebook_table = freeze_notebooks()
    output = paths.analysis_output_root() / "data"
    paths.write_table(object_table, output / "processed_input_objects.csv", ["object_name"])
    paths.write_table(membership_table, output / "processed_cohort_membership.csv", ["object_name", "source_accession"])
    paths.write_table(notebook_table, output / "upstream_notebook_snapshots.csv", ["snapshot"])

    # 原始计数队列来自 GSE211376；另三份处理对象中仅选入标记为本研究使用的来源。
    selected = membership_table.loc[membership_table["selected_for_current_study"], "source_accession"]
    summary = {
        "public_source_accessions": sorted({"GSE211376", *selected}),
        "processed_objects_audited": len(object_table),
        "upstream_notebooks_frozen": len(notebook_table),
        "all_required_columns_present": bool(object_table["required_columns_present"].all()),
        "all_obs_names_unique": bool(object_table["obs_names_unique"].all()),
        "object_assertions": assertions,
        "reproducibility_boundary": (
            "Exact processed objects must be deposited; GEO plus downstream scripts alone is insufficient."
        ),
    }
    paths.write_json(summary, output / "processed_input_audit.json")
    print(
        f"核对 {len(object_table)} 个处理对象、{len(membership_table)} 个对象来源、{len(notebook_table)} 份上游快照。"
    )


def audit_objects() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """逐个读取对象头与 obs，返回对象表、全部来源成员表及发表摘要字段。"""
    files = paths.primary_input_files()
    shared_region = {
        "dataset_value": "GSE274987",
        "column": "Platform",
        "values": {"Multiome": "GSE274986", "ParseBio": "GSE274987"},
    }
    specifications = {
        "scGBM_processed": {
            "path": files["scGBM"],
            "patient": "patient_id",
            "region": "tissue_type",
            "datasets": ["GSE162631", "GSE273274"],
            "accessions": ["GSE162631", "GSE273274"],
            "required": ["dataset", "patient_id", "tissue_type", "Major_category", "Minor_cluster"],
            "role": "GSE273274 primary analysis; GSE162631 uniform-marker sensitivity",
        },
        "snGBM_processed": {
            "path": files["snGBM"],
            "patient": "patient",
            "region": "tissue",
            "datasets": ["GSE276177", "GSE286419"],
            "accessions": ["GSE276177", "GSE286419"],
            "required": ["dataset", "patient", "tissue", "celltype"],
            "role": "GSE276177 and GSE286419 primary and uniform-marker analyses",
        },
        "regional_processed": {
            "path": files["GSE274987"],
            "patient": "patient_id",
            "region": "tissue_type",
            "datasets": ["GSE274987"],
            "accessions": ["GSE274986", "GSE274987"],
            "required": ["dataset", "patient_id", "tissue_type", "celltype", "Platform"],
            "legacy_source_accession_map": shared_region,
            "role": "Joint GSE274986/GSE274987 human regional cohort selected through "
            "the legacy dataset value GSE274987; GSE241037 cells excluded",
        },
    }

    # 每个对象读取完即关闭，后面的来源统计只依赖 obs，不载入大表达矩阵。
    object_rows = []
    member_rows = []
    assertions = {}
    for name, spec in specifications.items():
        path = spec["path"]
        source = ad.read_h5ad(path, backed="r")
        try:
            obs_table = source.obs.copy()
            n_cells, n_genes = source.shape
        finally:
            source.file.close()
        missing = set(spec["required"]) - set(obs_table.columns)
        if missing:
            raise ValueError(f"{name} 缺少字段 {sorted(missing)}")

        datasets = sorted(obs_table["dataset"].astype(str).unique())
        accessions = derive_source_accession(obs_table, "dataset", spec)
        all_accessions = sorted(accessions.unique())
        absent_datasets = set(spec["datasets"]) - set(datasets)
        absent_accessions = set(spec["accessions"]) - set(all_accessions)
        if absent_datasets or absent_accessions:
            raise ValueError(f"{name} 缺少 dataset {sorted(absent_datasets)} 或来源 {sorted(absent_accessions)}")

        # all_dataset_members 记录对象旧标签；source_accessions 分开记录实际来源，两者均供 S9 使用。
        assertions[name] = {
            "required_columns_present": not missing,
            "selected_dataset_values_present": not absent_datasets,
            "selected_source_accessions_present": not absent_accessions,
            "obs_names_unique": bool(obs_table.index.is_unique),
            "selected_dataset_values": spec["datasets"],
            "selected_source_accessions": spec["accessions"],
            "all_dataset_members": datasets,
            "source_accessions": all_accessions,
        }
        object_rows.append(
            {
                "object_name": name,
                "source_file_name": path.name,
                "repository_target_path": f"processed_inputs/{path.name}",
                "bytes": path.stat().st_size,
                "sha256": paths.file_sha256(path),
                "n_obs_total": n_cells,
                "n_vars_total": n_genes,
                "all_dataset_members": "; ".join(datasets),
                "source_accessions": "; ".join(all_accessions),
                "selected_dataset_values": "; ".join(spec["datasets"]),
                "selected_source_accessions": "; ".join(spec["accessions"]),
                "dataset_filter_column": "dataset",
                "patient_column": spec["patient"],
                "region_column": spec["region"],
                "analysis_role": spec["role"],
                "required_columns_present": not missing,
                "obs_names_unique": bool(obs_table.index.is_unique),
            }
        )
        for accession in all_accessions:
            member_table = obs_table.loc[accessions.eq(accession)]
            member_rows.append(
                {
                    "object_name": name,
                    "source_accession": accession,
                    "dataset_filter_values": "; ".join(sorted(member_table["dataset"].astype(str).unique())),
                    "selected_for_current_study": accession in spec["accessions"],
                    "n_cells_or_nuclei_in_object": len(member_table),
                    "n_patients_in_object": member_table[spec["patient"]].astype(str).nunique(),
                    "regions_in_object": "; ".join(sorted(member_table[spec["region"]].astype(str).unique())),
                }
            )

    object_table = pd.DataFrame(object_rows)
    membership_table = pd.DataFrame(member_rows)
    by_object = membership_table.groupby("object_name")["n_cells_or_nuclei_in_object"].sum()
    for row in object_table.itertuples(index=False):
        if by_object[row.object_name] != row.n_obs_total:
            raise ValueError(f"{row.object_name} 按来源分组后细胞总数与对象维度不一致")

    # 两个平台共用旧 dataset，分别与发表的来源人数核对；小鼠来源只进入完整成员表。
    reference_path = Path(__file__).resolve().parents[1] / "source_data" / "TableS9b_processed_cohort_membership.csv"
    reference_table = pd.read_csv(reference_path)
    for accession in shared_region["values"].values():
        current_rows = membership_table["source_accession"].eq(accession)
        reference_rows = reference_table["source_accession"].eq(accession)
        observed = membership_table.loc[current_rows, "n_patients_in_object"].item()
        expected = reference_table.loc[reference_rows, "n_patients_in_object"].item()
        if observed != expected:
            raise ValueError(f"{accession} 处理对象有 {observed} 位患者，发表来源表记载 {expected} 位")
    result = object_table, membership_table, assertions
    return result


def derive_source_accession(obs_table: pd.DataFrame, dataset_col: str, spec: dict) -> pd.Series:
    """对旧区域 dataset 值按 Platform 恢复来源；出现无映射的平台即停止。"""
    accession = obs_table[dataset_col].astype(str).copy()
    mapping = spec.get("legacy_source_accession_map")
    if mapping is None:
        return accession
    legacy = accession.eq(mapping["dataset_value"])
    platform = obs_table.loc[legacy, mapping["column"]].astype(str)
    unknown = set(platform) - set(mapping["values"])
    if unknown:
        raise ValueError(f"无法从 {mapping['column']} 映射来源 {sorted(unknown)}")
    accession.loc[legacy] = platform.map(mapping["values"])
    return accession


def freeze_notebooks() -> pd.DataFrame:
    """从原始上游 notebook 清除运行输出，核对代码单元内容与顺序，生成快照来源表。"""
    source_directory = paths.public_input_root(paths.project_root()) / "upstream_source"
    snapshots = {
        "scgbm_00_integrate_two_datasets.ipynb": "scGBM/0_integrate_two_datasets.ipynb",
        "scgbm_01_preprocess_scRNAdata.ipynb": "scGBM/1_preprocess_scRNAdata_v1.1.ipynb",
        "scgbm_02_marker_gene_legacy.ipynb": "scGBM/过时ipynb/2_marker_gene.ipynb",
        "scgbm_03_CNV_validation_legacy.ipynb": "scGBM/过时ipynb/3_CNV_validation.ipynb",
        "scgbm_04_Sub_clustering_legacy.ipynb": "scGBM/过时ipynb/4_Sub_clustering.ipynb",
        "scgbm_06_Combine_subclusters_legacy.ipynb": "scGBM/过时ipynb/6_Combine_subclusters.ipynb",
        "sngbm_01_snRNA_processing.ipynb": "snGBM/1_snRNA_processing.ipynb",
        "gse274987_00_integrate_two_datasets.ipynb": "三位置 GBM/0_integrate_two_datasets.ipynb",
    }
    for subpath in snapshots.values():
        if not (source_directory / subpath).is_file():
            raise FileNotFoundError(source_directory / subpath)

    output = paths.analysis_output_root() / "code" / "upstream_processing"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, subpath in snapshots.items():
        source = source_directory / subpath
        target = output / name
        notebook = json.loads(source.read_text(encoding="utf-8"))
        sources = [(cell["cell_type"], cell["source"]) for cell in notebook["cells"]]
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                cell["outputs"] = []
                cell["execution_count"] = None

        # 用写盘后的 JSON 核对所有单元的代码、顺序、清空的输出和 execution_count。
        target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
        saved = json.loads(target.read_text(encoding="utf-8"))
        saved_sources = [(cell["cell_type"], cell["source"]) for cell in saved["cells"]]
        code_cells = [cell for cell in saved["cells"] if cell["cell_type"] == "code"]
        if not sources or saved_sources != sources or not code_cells:
            raise ValueError(f"{name} 的单元缺失或内容、顺序发生变化")
        if any(cell["outputs"] or cell["execution_count"] is not None for cell in code_cells):
            raise ValueError(f"{name} 的运行输出或执行计数未清除")
        rows.append(
            {
                "snapshot": name,
                "source_project_subpath": subpath,
                "source_bytes": source.stat().st_size,
                "snapshot_bytes": target.stat().st_size,
                "snapshot_sha256": paths.file_sha256(target),
            }
        )

    # 旧处理对象经历过多轮注释。这里保存本地上游来源；公开快照另做过重编码与元数据清理。
    readme = (
        "# Upstream processing snapshots\n\n"
        "These snapshots retain source cells in their original order and clear execution outputs. "
        "They document the processing history. Rebuilding the precise AnnData also requires the original inputs "
        "and annotation history. Public release copies underwent separate identifier and metadata editing.\n\n"
        "TableS9a identifies the original objects. TableS9b lists their source accessions. "
        "In the regional object the legacy GSE274987 label covers both Multiome (GSE274986) and ParseBio "
        "(GSE274987), analyzed as one source-study cohort.\n"
    )
    description = output / "README.md"
    description.write_text(readme, encoding="utf-8")
    notebook_table = pd.DataFrame(rows)
    return notebook_table


if __name__ == "__main__":
    main()
