#!/usr/bin/env python3
# Purpose: Audit patients, source accessions and regions in the public processed objects and preserve upstream notebook
# provenance.
# Inputs: obs and X dimensions of three h5ad files in PUBLIC_INPUT_ROOT/processed_inputs and eight original notebooks in
# upstream_source/.
# TableS9b_processed_cohort_membership.csv records regional patient counts by source.
# TableS9c_upstream_notebook_snapshots.csv records the original notebook paths.
# Outputs: data/processed_input_objects.csv, data/processed_cohort_membership.csv,
# data/upstream_notebook_snapshots.csv and data/processed_input_audit.json. Under code/upstream_processing/, write
# scgbm_00_integrate_two_datasets.ipynb, scgbm_01_preprocess_scRNAdata.ipynb, scgbm_02_marker_gene_legacy.ipynb,
# scgbm_03_CNV_validation_legacy.ipynb, scgbm_04_Sub_clustering_legacy.ipynb, scgbm_06_Combine_subclusters_legacy.ipynb,
# sngbm_01_snRNA_processing.ipynb, gse274987_00_integrate_two_datasets.ipynb and README.md.
# Usage: Run python 04_audit_processed_inputs.py from code/. Standalone releases write to 01_cohort_reanalysis/.

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import anndata as ad
import pandas as pd

paths = import_module("00a_project_paths")


def main() -> None:
    """Write inventories by object, object/source accession and upstream notebook. Returns None."""
    object_table, membership_table, assertions = audit_objects()
    notebook_table = freeze_notebooks()
    output = paths.analysis_output_root() / "data"
    paths.write_table(object_table, output / "processed_input_objects.csv", ["object_name"])
    paths.write_table(membership_table, output / "processed_cohort_membership.csv", ["object_name", "source_accession"])
    paths.write_table(notebook_table, output / "upstream_notebook_snapshots.csv", ["snapshot"])

    # GSE211376 supplies raw counts. For the other three objects, include only the accessions selected for this study in
    # the public source list.
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
        f"Audited {len(object_table)} processed objects, {len(membership_table)} object/source rows "
        f"and {len(notebook_table)} upstream snapshots."
    )


def audit_objects() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read each header and obs. Return the object inventory, all source memberships and published summary fields."""
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

    # Close each object once its metadata have been read. Source membership uses obs and does not need the large
    # expression matrix.
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
            raise ValueError(f"{name} is missing fields: {sorted(missing)}")

        datasets = sorted(obs_table["dataset"].astype(str).unique())
        accessions = derive_source_accession(obs_table, "dataset", spec)
        all_accessions = sorted(accessions.unique())
        absent_datasets = set(spec["datasets"]) - set(datasets)
        absent_accessions = set(spec["accessions"]) - set(all_accessions)
        if absent_datasets or absent_accessions:
            raise ValueError(
                f"{name} is missing datasets {sorted(absent_datasets)} or accessions {sorted(absent_accessions)}"
            )

        # S9 reports the original object labels in all_dataset_members and the source accessions separately in
        # source_accessions.
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
            raise ValueError(f"{row.object_name}: cell counts by source do not sum to the object's row count")

    # Compare the two platforms sharing a legacy dataset label to the published patient counts by accession. Keep mouse
    # cells only in the full membership table.
    reference_path = Path(__file__).resolve().parents[1] / "source_data" / "TableS9b_processed_cohort_membership.csv"
    reference_table = pd.read_csv(reference_path)
    for accession in shared_region["values"].values():
        current_rows = membership_table["source_accession"].eq(accession)
        reference_rows = reference_table["source_accession"].eq(accession)
        observed = membership_table.loc[current_rows, "n_patients_in_object"].item()
        expected = reference_table.loc[reference_rows, "n_patients_in_object"].item()
        if observed != expected:
            raise ValueError(
                f"{accession}: observed {observed} patients in the object. The published table records {expected}"
            )
    result = object_table, membership_table, assertions
    return result


def derive_source_accession(obs_table: pd.DataFrame, dataset_col: str, spec: dict) -> pd.Series:
    """Map the legacy regional dataset to source accessions by Platform. Stop for an unmapped platform."""
    accession = obs_table[dataset_col].astype(str).copy()
    mapping = spec.get("legacy_source_accession_map")
    if mapping is None:
        return accession
    legacy = accession.eq(mapping["dataset_value"])
    platform = obs_table.loc[legacy, mapping["column"]].astype(str)
    unknown = set(platform) - set(mapping["values"])
    if unknown:
        raise ValueError(f"Cannot map the following {mapping['column']} values to source accessions: {sorted(unknown)}")
    accession.loc[legacy] = platform.map(mapping["values"])
    return accession


def freeze_notebooks() -> pd.DataFrame:
    """Clear original notebook outputs, check cell content and order, and return the snapshot inventory."""
    source_directory = paths.public_input_root(paths.project_root()) / "upstream_source"

    # S9c records the literal paths and snapshot order from the original working project.
    # Read these fields verbatim so private source directories can retain their original names.
    reference = Path(__file__).resolve().parents[1] / "source_data" / "TableS9c_upstream_notebook_snapshots.csv"
    source_table = pd.read_csv(reference, usecols=["snapshot", "source_project_subpath"], dtype=str)
    snapshots = source_table.set_index("snapshot", verify_integrity=True)["source_project_subpath"].to_dict()
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

        # Reopen the saved JSON to verify every cell's source and order, plus the cleared outputs and execution counts.
        target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
        saved = json.loads(target.read_text(encoding="utf-8"))
        saved_sources = [(cell["cell_type"], cell["source"]) for cell in saved["cells"]]
        code_cells = [cell for cell in saved["cells"] if cell["cell_type"] == "code"]
        if not sources or saved_sources != sources or not code_cells:
            raise ValueError(f"{name}: missing cells or changes to cell content or order")
        if any(cell["outputs"] or cell["execution_count"] is not None for cell in code_cells):
            raise ValueError(f"{name}: execution outputs or execution counts were retained")
        rows.append(
            {
                "snapshot": name,
                "source_project_subpath": subpath,
                "source_bytes": source.stat().st_size,
                "snapshot_bytes": target.stat().st_size,
                "snapshot_sha256": paths.file_sha256(target),
            }
        )

    # The objects underwent several annotation rounds. Preserve local upstream sources here; public copies underwent
    # separate recoding and metadata edits.
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
