> **Current PLOS ONE manuscript release:** [PONE-D-26-35909](PONE_D_26_35909/README.md) contains the patient-level TGFB2 analyses, recoded derived inputs, and October 2026 revision. The repository-root description and stage01-stage26 scripts below document an earlier project version and do not describe the revised manuscript’s conclusions.

# gbm-astrocyte-masquerade

Analysis code for the manuscript:

> **Malignant cells masquerade as astrocytes in glioblastoma and confound immune-microenvironment analysis**
> Zuan Yu, Yuqi Tao, Tao Yu, Risheng Liang
> Correspondence: Risheng Liang (doctoran94@126.com)

## Overview

Single-cell/nucleus RNA-seq of IDH-wildtype glioblastoma routinely resolves an "astrocyte" cluster that is interpreted as a non-malignant support-cell population. Across four independent public cohorts (>470,000 cells/nuclei), this code shows that the cluster is dominated by AC-like malignant cells carrying the GBM chromosome 7 gain / chromosome 10 loss (+7/−10) hallmark, quantifies this "astrocyte masquerade" non-circularly (astrocyte markers to define the cluster, an immune-referenced +7/−10 threshold to call malignancy, independent of the original labels), and demonstrates that the contamination inflates the astrocyte→macrophage TGF-β2 immunosuppressive signal.

## Cohorts

| GEO accession | Cells/nuclei | Patients | Platform |
|---|---|---|---|
| GSE211376 (discovery) | 39,355 | 11 | snRNA-seq |
| GSE162631 + GSE273274 | 135,535 | 7 | scRNA-seq |
| GSE276177 + GSE286419 | 68,920 | 6 | snRNA-seq |
| GSE174554 | 228,319 | 80 | snRNA-seq |

Bulk survival: TCGA-GBM (UCSC Xena, HiSeqV2) and CGGA (mRNAseq_693).

## Repository layout

```
gbm-astrocyte-masquerade/
├── stage01-02 … stage26   analysis pipeline (see below)
├── requirements.txt
├── LICENSE
├── .gitignore
└── data/                  inputs & intermediates (not tracked — see Data)
    ├── intermediate/       per-stage intermediate outputs
    └── external/           externally processed h5ad (scGBM, snGBM)
```

## Data

All primary data are publicly available from NCBI GEO under the accessions above; **raw data are not included in this repository**. Download the matrices from GEO and place them (plus the per-stage intermediate files) under `data/`. A per-sample manifest (accession, tissue, IDH status) is provided with the manuscript as Supplementary Table S1.

> **Note on paths.** The scripts in this repository reflect the analysis exactly as run. Absolute local paths in the original scripts have been replaced with a `data/`-relative layout for release (no analysis logic was changed); adjust the `data/` paths to match your own environment.

## Environment

- Python 3.11
- `pip install -r requirements.txt` (numpy, pandas, scipy, scikit-learn, matplotlib, anndata, h5py, harmonypy, leidenalg)

Several upstream steps referenced in the Methods (inferCNV / copy-number scoring, LIANA cell–cell communication, decoupler/CollecTRI transcription-factor activity) were run with their respective tools; their outputs are consumed by the scripts here.

## Pipeline stages

**CNV validation & clustering (01–08)**
- `stage01-02` — snGBM (GSE276177+GSE286419) non-circular +7/−10 CNV validation
- `stage03` — GSE211376 non-circular CNV validation
- `stage04` — GSE211376 marker-defined astrocyte (non-circular)
- `stage05-06` — GSE174554 Harmony integration, Leiden clustering, CNV scoring
- `stage07` — GSE174554 cluster marker annotation
- `stage08` — cross-cohort threshold matching

**Communication & immune-evasion (09–16)**
- `stage09-10` — malignant→microenvironment communication (CellPhoneDB-style product statistic)
- `stage11` — LIANA cross-method validation
- `stage12` — GSE211376 astrocyte-cluster decontamination
- `stage13` — GSE174554 astrocyte-cluster decontamination
- `stage14` — 41-gene immune-evasion panel, cross-cohort
- `stage15` — Figure 6 cross-cohort concordance
- `stage16` — scGBM AC-masquerade validation

**Figures (20–26)**
- `stage20`–`stage25` — main Figures 1–6
- `stage26` — graphical abstract

## Citation

If you use this code, please cite the manuscript (journal and DOI to be added upon acceptance).

## License

Released under the MIT License (see `LICENSE`).
