#!/usr/bin/env python3
# Purpose: Check coverage of the TGFB2/TGFBR/SMAD axis and canonical SMAD activation sites in CPTAC-GBM.
# Input: phosphoproteome_mssm_per_gene_clean.v4.0.tsv.gz, using gene, site and the original column denominator.
# Output: 01_phosphosite_coverage.csv and 02_phosphosite_summary.json under
# external_validation/10_cptac_phosphosite_audit/.
# Usage: python 10_cptac_phosphosite_audit.py --input-file LOCAL_MATRIX.tsv.gz [--output-dir OUTPUT_DIRECTORY]
# Read the matrix in chunks. Use --input-dir for a local Proteome directory or --input-url for a direct HTTPS link.

import argparse
import re
from importlib import import_module
from pathlib import Path

import pandas as pd

spatial = import_module("00a_spatial_analysis")
files = import_module("00a_project_paths")


def main():
    """Audit phosphosite coverage by row. Stop without exactly one gene column and one site column. No return."""
    external_dir = Path(__file__).resolve().parents[1] / "external_validation"
    parser = argparse.ArgumentParser(description="CPTAC-GBM phosphosite coverage")
    local = parser.add_mutually_exclusive_group()
    local.add_argument("--input-file", type=Path, help="Clean phosphoproteome matrix")
    local.add_argument("--input-dir", type=Path, default=external_dir / "raw/cptac_gbm/Proteome")
    parser.add_argument("--input-url", help="Direct HTTPS URL to use when the local matrix is absent")
    parser.add_argument("--output-dir", type=Path, default=external_dir / Path(__file__).stem)
    args = parser.parse_args()

    # Prefer the local table, then its source-specific cache. A downloaded gzip must be complete and include a gene
    # header.
    local_path = args.input_file or (args.input_dir / "phosphoproteome_mssm_per_gene_clean.v4.0.tsv.gz")
    path = spatial.cptac_input(local_path, args.input_url, args.output_dir / "raw", ["gene"])
    header_table = pd.read_csv(path, sep="\t", nrows=0)
    gene_names = [name for name in header_table if name.lower() == "gene"]
    site_names = [name for name in header_table if name.lower() in {"site", "phosphosite", "phospho_site"}]
    if len(gene_names) != 1 or len(site_names) != 1:
        raise ValueError("The phosphoproteome table must have exactly one recognizable gene column and one site column")
    gene_column, site_column = gene_names[0], site_names[0]

    # Retain the published total_sample_columns definition. In v4.0 its denominator includes refseq_id.
    # Sequence identifiers convert to missing numeric values, so this denominator exceeds the actual sample count by
    # one.
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

    # Each source site gets a row. Keep an empty site for absent genes. Its canonical flag denotes a gene targeted by
    # the audit.
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

        # Match complete residue-position labels to distinguish S465 from S4650. Record site presence and quantification
        # separately.
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

    # Retain all target rows and absent genes. A canonical match requires the target gene, an exact site and an actual
    # row.
    present_count = int(coverage_table["row_present"].sum())
    if present_count != len(target_table_total) or set(coverage_table["gene"]) != set(target_genes):
        raise ValueError("The coverage table does not account for every source target row and missing gene")
    if not coverage_table["nonmissing_samples"].between(0, len(historical_columns)).all():
        raise ValueError("Quantified sample columns exceed the original analysis denominator")
    files.write_table(coverage_table, args.output_dir / "01_phosphosite_coverage.csv", ["gene", "site"])
    is_canonical = coverage_table["gene"].isin(canonical_sites) & coverage_table["canonical_activation_site"]
    present_sites = is_canonical & coverage_table["row_present"]
    detected_table = coverage_table.loc[present_sites, ["gene", "site", "nonmissing_samples"]]

    # The published available flag denotes a matching row. Assessing pathway activation requires separate quantitative
    # evidence.
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
    files.write_json(summary, args.output_dir / "02_phosphosite_summary.json")
    print(
        f"{len(target_genes)} target genes, {len(target_table_total)} observed site rows, "
        f"{len(detected_table)} canonical SMAD sites"
    )


if __name__ == "__main__":
    main()
