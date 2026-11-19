> **Current PLOS ONE manuscript** · [Patient-level TGFB2 analysis, data and reproduction instructions](PONE_D_26_35909/README.md). The numbered scripts at the repository root belong to the earlier astrocyte study described below.

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

## Reproduce the earlier study

Use Python 3.11, install the packages in `requirements.txt`, and install Poppler with `pdfinfo`, `pdftotext`, `pdftoppm` and `pdftocairo` on PATH. Run the numbered scripts from the repository root. Each script header gives its input filenames, output filenames and invocation. New results are grouped by stage under `03_result/`.

| Run order | Purpose | Stage numbers in the earlier release |
|---|---|---|
| 01–03 | Immune-referenced CNV scoring in snGBM and GSE211376, including the marker-defined astrocyte group | 01–02, 03, 04 |
| 04–06 | Recluster GSE174554, reproduce the archived cluster-specific CNV analysis, and compare matched thresholds across cohorts | 07, 05–06, 08 |
| 07–10 | scGBM communication and mixing analysis, snGBM ligand rankings, and sender-cell filtering in GSE211376 and GSE174554 | 09–10, 11, 12, 13 |
| 11–13 | Cross-cohort expression and ranks for the immune panel, then tumor and peritumor AC-like expression in scGBM | 14, 15, 16 |
| 14–19 | Generate Figures 1–6 | 20–25 |
| 20 | Generate the graphical abstract | 26 |

The GEO matrices, processed AnnData objects and upstream annotations are separate inputs. Place the scGBM and snGBM objects under `data/external/`, the GSE174554 matrices under `data/GSE174554/`, and the accession-specific matrices, metadata and intermediate tables listed in each script under `data/intermediate/`. Other earlier-study inputs such as `cell_metadata.csv` and `infercnv_cell_scores.csv` go directly under `data/`. Some stages consume existing inferCNV, LIANA or transcription-factor activity outputs. Downloading an expression matrix alone does not reconstruct those intermediates.

Stage 04 reclusters the GSE174554 cells. For historical reproduction, stage 05 reads the original `gse174554_cluster.npz` and the matching `gse174554_cluster_markers.csv` from `data/intermediate/`. Stage 10 uses the partition and CNV values recorded together by stage 05. The archived partition had 18 clusters. Re-running the clustering with the validated environment produced 16. The numerical cluster labels cannot be interchanged. The archive records sample and tissue order. Stage 05 reconstructs row keys from the RNA libraries and labels them `reconstructed_cell_id`. The archive lacks original barcodes for independently checking the order within a sample and tissue.

The first command below runs a CNV analysis with the source inputs present. The second renders Figure 1. Six small [figure source tables](01_figure_sources/) transcribe values, labels or schematic connections explicitly listed in the earlier Git release. Panels whose values were read from external files require the corresponding analysis output or named local input. The [per-panel input guide](01_figure_sources/12_figure_data_sources.md) lists these dependencies. Each plot is saved as a PDF master and a PNG rendered from that PDF.

```sh
python 01_sngbm_immune_reference_cnv.py
python 14_fig1_cohorts_and_study_design.py
```

The [analysis layout and executable entry points](ARCHITECTURE.md) cover both studies. The [October release](https://github.com/yuzuan/gbm-astrocyte-masquerade/tree/a63d6b46c531a786a266a1303b3b5bd7b1ba0bd3) preserves the original script filenames for earlier citations.

## Citation

If you use this code, please cite the manuscript (journal and DOI to be added upon acceptance).

## License

Released under the MIT License (see `LICENSE`).
