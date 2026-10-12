#!/usr/bin/env python3
"""审计公共数据处理对象，并冻结上游 notebook 的无输出快照。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import pandas as pd

from stage_public_paths import project_root, public_input_root, submission_package


ROOT = project_root()
PACKAGE = submission_package(ROOT)
INPUT_ROOT = public_input_root(ROOT)
UPSTREAM_SOURCE_ROOT = INPUT_ROOT / "upstream_source"
DATA = PACKAGE / "data"
UPSTREAM_OUT = PACKAGE / "code" / "upstream_processing"

OBJECTS = {
    "scGBM_processed": {
        "path": INPUT_ROOT / "processed_inputs" / "scGBM_GSE162631_GSE273274.h5ad",
        "dataset_col": "dataset",
        "patient_col": "patient_id",
        "region_col": "tissue_type",
        "selected_dataset_values": ["GSE162631", "GSE273274"],
        "selected_source_accessions": ["GSE162631", "GSE273274"],
        "required_columns": ["dataset", "patient_id", "tissue_type", "Major_category", "Minor_cluster"],
        "analysis_role": "GSE273274 primary analysis; GSE162631 uniform-marker sensitivity",
        "repository_target_path": "processed_inputs/scGBM_GSE162631_GSE273274.h5ad",
    },
    "snGBM_processed": {
        "path": INPUT_ROOT / "processed_inputs" / "snGBM_GSE276177_GSE286419.h5ad",
        "dataset_col": "dataset",
        "patient_col": "patient",
        "region_col": "tissue",
        "selected_dataset_values": ["GSE276177", "GSE286419"],
        "selected_source_accessions": ["GSE276177", "GSE286419"],
        "required_columns": ["dataset", "patient", "tissue", "celltype"],
        "analysis_role": "GSE276177 and GSE286419 primary and uniform-marker analyses",
        "repository_target_path": "processed_inputs/snGBM_GSE276177_GSE286419.h5ad",
    },
    "regional_processed": {
        "path": INPUT_ROOT / "processed_inputs" / "regional_GSE274987_GSE241037.h5ad",
        "dataset_col": "dataset",
        "patient_col": "patient_id",
        "region_col": "tissue_type",
        "selected_dataset_values": ["GSE274987"],
        "selected_source_accessions": ["GSE274986", "GSE274987"],
        "required_columns": ["dataset", "patient_id", "tissue_type", "celltype", "Platform"],
        "legacy_source_accession_map": {
            "dataset_value": "GSE274987",
            "column": "Platform",
            "values": {
                "Multiome": "GSE274986",
                "ParseBio": "GSE274987",
            },
        },
        "analysis_role": (
            "Joint GSE274986/GSE274987 human regional cohort selected through "
            "the legacy dataset value GSE274987; GSE241037 cells excluded"
        ),
        "repository_target_path": "processed_inputs/regional_GSE274987_GSE241037.h5ad",
    },
}

NOTEBOOKS = {
    "scgbm_00_integrate_two_datasets.ipynb": UPSTREAM_SOURCE_ROOT / "scGBM" / "0_integrate_two_datasets.ipynb",
    "scgbm_01_preprocess_scRNAdata.ipynb": UPSTREAM_SOURCE_ROOT / "scGBM" / "1_preprocess_scRNAdata_v1.1.ipynb",
    "scgbm_02_marker_gene_legacy.ipynb": UPSTREAM_SOURCE_ROOT / "scGBM" / "过时ipynb" / "2_marker_gene.ipynb",
    "scgbm_03_CNV_validation_legacy.ipynb": UPSTREAM_SOURCE_ROOT / "scGBM" / "过时ipynb" / "3_CNV_validation.ipynb",
    "scgbm_04_Sub_clustering_legacy.ipynb": UPSTREAM_SOURCE_ROOT / "scGBM" / "过时ipynb" / "4_Sub_clustering.ipynb",
    "scgbm_06_Combine_subclusters_legacy.ipynb": UPSTREAM_SOURCE_ROOT / "scGBM" / "过时ipynb" / "6_Combine_subclusters.ipynb",
    "sngbm_01_snRNA_processing.ipynb": UPSTREAM_SOURCE_ROOT / "snGBM" / "1_snRNA_processing.ipynb",
    "gse274987_00_integrate_two_datasets.ipynb": UPSTREAM_SOURCE_ROOT / "三位置 GBM" / "0_integrate_two_datasets.ipynb",
}


def sha256(path: Path) -> str:
    """分块计算大型 h5ad 的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_source_accession(
    obs: pd.DataFrame,
    dataset_col: str,
    spec: dict[str, object],
) -> pd.Series:
    """把历史合并标签还原为真实 GEO accession。"""
    source_accession = obs[dataset_col].astype(str).copy()
    mapping = spec.get("legacy_source_accession_map")
    if mapping is None:
        return source_accession

    dataset_value = str(mapping["dataset_value"])
    mapping_column = str(mapping["column"])
    value_map = {str(key): str(value) for key, value in mapping["values"].items()}
    legacy_mask = source_accession.eq(dataset_value)
    observed_values = obs.loc[legacy_mask, mapping_column].astype(str)
    unknown = sorted(set(observed_values) - set(value_map))
    if unknown:
        raise ValueError(
            f"{dataset_value} 无法从 {mapping_column} 映射来源 accession: {unknown}"
        )
    source_accession.loc[legacy_mask] = observed_values.map(value_map)
    return source_accession


def audit_objects() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """检查对象字段、成员 accession 和分析实际使用的患者。"""
    object_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, object]] = []
    assertions: dict[str, object] = {}
    for object_name, spec in OBJECTS.items():
        path = Path(spec["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        adata = ad.read_h5ad(path, backed="r")
        obs = adata.obs.copy()
        missing_columns = sorted(set(spec["required_columns"]) - set(obs.columns))
        if missing_columns:
            adata.file.close()
            raise ValueError(f"{object_name} 缺少字段: {missing_columns}")
        dataset_col = str(spec["dataset_col"])
        patient_col = str(spec["patient_col"])
        region_col = str(spec["region_col"])
        members = sorted(obs[dataset_col].astype(str).unique())
        selected_dataset_values = list(spec["selected_dataset_values"])
        absent_dataset_values = sorted(set(selected_dataset_values) - set(members))
        if absent_dataset_values:
            adata.file.close()
            raise ValueError(
                f"{object_name} 缺少目标 dataset 值: {absent_dataset_values}"
            )
        source_accession = derive_source_accession(obs, dataset_col, spec)
        source_accessions = sorted(source_accession.unique())
        selected_source_accessions = list(spec["selected_source_accessions"])
        absent_source_accessions = sorted(
            set(selected_source_accessions) - set(source_accessions)
        )
        if absent_source_accessions:
            adata.file.close()
            raise ValueError(
                f"{object_name} 缺少目标来源 accession: {absent_source_accessions}"
            )
        object_rows.append(
            {
                "object_name": object_name,
                "source_file_name": path.name,
                "repository_target_path": spec["repository_target_path"],
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "n_obs_total": adata.n_obs,
                "n_vars_total": adata.n_vars,
                "all_dataset_members": "; ".join(members),
                "source_accessions": "; ".join(source_accessions),
                "selected_dataset_values": "; ".join(selected_dataset_values),
                "selected_source_accessions": "; ".join(
                    selected_source_accessions
                ),
                "dataset_filter_column": dataset_col,
                "patient_column": patient_col,
                "region_column": region_col,
                "analysis_role": spec["analysis_role"],
                "required_columns_present": True,
                "obs_names_unique": bool(obs.index.is_unique),
            }
        )
        for accession in source_accessions:
            frame = obs[source_accession.eq(accession)]
            membership_rows.append(
                {
                    "object_name": object_name,
                    "source_accession": accession,
                    "dataset_filter_values": "; ".join(
                        sorted(frame[dataset_col].astype(str).unique())
                    ),
                    "selected_for_current_study": (
                        accession in selected_source_accessions
                    ),
                    "n_cells_or_nuclei_in_object": len(frame),
                    "n_patients_in_object": frame[patient_col].astype(str).nunique(),
                    "regions_in_object": "; ".join(sorted(frame[region_col].astype(str).unique())),
                }
            )
        assertions[object_name] = {
            "required_columns_present": not missing_columns,
            "selected_dataset_values_present": not absent_dataset_values,
            "selected_source_accessions_present": not absent_source_accessions,
            "obs_names_unique": bool(obs.index.is_unique),
            "selected_dataset_values": selected_dataset_values,
            "selected_source_accessions": selected_source_accessions,
            "all_dataset_members": members,
            "source_accessions": source_accessions,
        }
        adata.file.close()
    regional_membership = pd.DataFrame(membership_rows)
    regional_membership = regional_membership[
        regional_membership["object_name"].eq("regional_processed")
    ].set_index("source_accession")
    expected_patients = {"GSE274986": 6, "GSE274987": 4}
    observed_patients = {
        accession: int(regional_membership.loc[accession, "n_patients_in_object"])
        for accession in expected_patients
    }
    if observed_patients != expected_patients:
        raise ValueError(
            "联合区域对象来源患者数与 GEO 记录不符: "
            f"{observed_patients} != {expected_patients}"
        )
    return pd.DataFrame(object_rows), pd.DataFrame(membership_rows), assertions


def freeze_notebooks() -> pd.DataFrame:
    """去除执行输出后复制 notebook，避免把大型缓存写入投稿包。"""
    missing = [str(path) for path in NOTEBOOKS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少上游 notebook：\n" + "\n".join(missing))
    UPSTREAM_OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for target_name, source in NOTEBOOKS.items():
        notebook = json.loads(source.read_text(encoding="utf-8"))
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
        target = UPSTREAM_OUT / target_name
        target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
        rows.append(
            {
                "snapshot": target_name,
                "source_project_subpath": str(source.relative_to(UPSTREAM_SOURCE_ROOT)),
                "source_bytes": source.stat().st_size,
                "snapshot_bytes": target.stat().st_size,
                "snapshot_sha256": sha256(target),
            }
        )
    readme = """# Upstream processing snapshots

These output-cleared notebooks document the local processing and annotation path used to build the project AnnData inputs from public GEO data.

They are provenance snapshots, not a claim that the current processed objects can be reconstructed byte-for-byte from these notebooks alone. The annotation objects evolved through iterative project runs. The exact processed inputs are identified by SHA-256 in `TableS9a_processed_input_objects.csv` and must be deposited with the final code release. Until a permanent repository DOI is inserted, the submission package is not an end-to-end GEO-to-result archive.

The statistical analysis separates cohorts by the `dataset` metadata field before calculating patient-level summaries. Shared storage or shared preprocessing does not merge patients across accessions, but it is disclosed because the processed objects are common upstream dependencies.

The regional notebook and AnnData object use the legacy value `dataset == "GSE274987"` for a joint Wu et al. cohort assembled from two source accessions. `Platform == "ParseBio"` maps to GSE274987 (4 patients, 11 regional profiles), whereas `Platform == "Multiome"` maps to GSE274986 (6 patients, 26 RNA profiles). The current analysis retains this joint 10-patient source-study cohort and does not count the two accessions as independent replications. The source-accession mapping is audited in `TableS9b_processed_cohort_membership.csv`.
"""
    (UPSTREAM_OUT / "README.md").write_text(readme, encoding="utf-8")
    return pd.DataFrame(rows)


def main() -> None:
    """写出处理对象、队列成员和 notebook 快照清单。"""
    DATA.mkdir(parents=True, exist_ok=True)
    objects, membership, assertions = audit_objects()
    notebooks = freeze_notebooks()
    objects.to_csv(DATA / "processed_input_objects.csv", index=False)
    membership.to_csv(DATA / "processed_cohort_membership.csv", index=False)
    notebooks.to_csv(DATA / "upstream_notebook_snapshots.csv", index=False)
    summary = {
        "public_source_accessions": [
            "GSE162631",
            "GSE211376",
            "GSE273274",
            "GSE274986",
            "GSE274987",
            "GSE276177",
            "GSE286419",
        ],
        "processed_objects_audited": len(objects),
        "upstream_notebooks_frozen": len(notebooks),
        "all_required_columns_present": bool(objects["required_columns_present"].all()),
        "all_obs_names_unique": bool(objects["obs_names_unique"].all()),
        "object_assertions": assertions,
        "reproducibility_boundary": "Exact processed objects must be deposited; GEO plus downstream scripts alone is insufficient.",
    }
    (DATA / "processed_input_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
