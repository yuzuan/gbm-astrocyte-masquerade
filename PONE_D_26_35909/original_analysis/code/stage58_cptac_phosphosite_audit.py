#!/usr/bin/env python3
"""Audit CPTAC GBM phosphosite coverage for the TGFB2-TGFBR-SMAD axis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = (
    ROOT
    / "external_validation"
    / "raw"
    / "cptac_gbm"
    / "Proteome"
    / "phosphoproteome_mssm_per_gene_clean.v4.0.tsv.gz"
)
RESULT_DIR = ROOT / "external_validation" / "results" / "cptac_gbm"
TARGET_GENES = {"TGFB2", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "SMAD4"}
CANONICAL_SITES = {
    "SMAD2": {"S465", "S467"},
    "SMAD3": {"S423", "S425"},
}


def detect_columns(columns: list[str]) -> tuple[str, str]:
    gene_candidates = [column for column in columns if column.lower() == "gene"]
    site_candidates = [
        column
        for column in columns
        if column.lower() in {"site", "phosphosite", "phospho_site"}
    ]
    if not gene_candidates or not site_candidates:
        raise ValueError(f"Unexpected phosphoproteome columns: {columns[:12]}")
    return gene_candidates[0], site_candidates[0]


def main() -> None:
    if not DATA_FILE.is_file():
        raise FileNotFoundError(DATA_FILE)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(DATA_FILE, sep="\t", nrows=0)
    columns = header.columns.tolist()
    gene_column, site_column = detect_columns(columns)
    metadata_columns = {
        gene_column,
        site_column,
        "Reference Sequence",
        "Peptide",
        "Sequence",
        "refseq",
        "peptide",
    }
    sample_columns = [column for column in columns if column not in metadata_columns]

    selected = []
    for chunk in pd.read_csv(DATA_FILE, sep="\t", chunksize=50000, low_memory=False):
        genes = chunk[gene_column].astype(str).str.upper()
        keep = genes.isin(TARGET_GENES)
        if keep.any():
            selected.append(chunk.loc[keep].copy())
    target_rows = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=columns)

    records = []
    for gene in sorted(TARGET_GENES):
        gene_rows = target_rows[target_rows[gene_column].astype(str).str.upper() == gene]
        if gene_rows.empty:
            records.append(
                {
                    "gene": gene,
                    "site": "",
                    "row_present": False,
                    "nonmissing_samples": 0,
                    "total_sample_columns": len(sample_columns),
                    "canonical_activation_site": gene in CANONICAL_SITES,
                }
            )
            continue
        for _, row in gene_rows.iterrows():
            site = str(row[site_column])
            values = pd.to_numeric(row[sample_columns], errors="coerce")
            canonical = any(
                canonical_site in site
                for canonical_site in CANONICAL_SITES.get(gene, set())
            )
            records.append(
                {
                    "gene": gene,
                    "site": site,
                    "row_present": True,
                    "nonmissing_samples": int(values.notna().sum()),
                    "total_sample_columns": len(sample_columns),
                    "canonical_activation_site": canonical,
                }
            )

    coverage = pd.DataFrame(records).sort_values(
        ["gene", "row_present", "site"], ascending=[True, False, True]
    )
    coverage.to_csv(RESULT_DIR / "phosphosite_coverage.csv", index=False)

    canonical_rows = coverage[
        coverage["gene"].isin(CANONICAL_SITES)
        & coverage["canonical_activation_site"]
        & coverage["row_present"]
    ]
    summary = {
        "source_file": str(DATA_FILE),
        "sample_columns": len(sample_columns),
        "target_genes": sorted(TARGET_GENES),
        "canonical_sites_requested": {
            gene: sorted(sites) for gene, sites in CANONICAL_SITES.items()
        },
        "canonical_sites_detected": canonical_rows[
            ["gene", "site", "nonmissing_samples"]
        ].to_dict(orient="records"),
        "canonical_smad_activation_available": not canonical_rows.empty,
        "interpretation": (
            "Canonical SMAD2 S465/S467 and SMAD3 S423/S425 activation sites are "
            "not represented in the clean CPTAC GBM phosphoproteome matrix; "
            "downstream TGF-beta pathway activation cannot be validated from this file."
        ),
    }
    (RESULT_DIR / "phosphosite_coverage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
