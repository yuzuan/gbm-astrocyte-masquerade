# Revision analysis reproduction

Manuscript PONE-D-26-35909; analyses added in response to peer review, 12 October 2026.

## Lightweight, independently executable route

Open `Revision_analysis.ipynb` from this directory and run all cells. It fits every REML/modified Knapp-Hartung model from `analysis/meta_patient_inputs.csv`, independently verifies the numerical estimates in SciPy, checks the supplied audit summaries and regenerates S15-S17 figures. It requires Python with NumPy, pandas, SciPy, Matplotlib, AnnData and h5py, and R with metafor installed. `Rscript` must be on PATH, or set `RSCRIPT` to its executable. Exact versions used are in validation/. Figures reuse the original submission plottie_figure_style.py unchanged: Arial/Helvetica font preference, 7.1-inch canvas width, shared panel headings and semantic cell-state colors (AC-like coral; other malignant blue). The notebook writes to `reproduced/` and does not overwrite the supplied reference tables. It does not rerun source count extraction or independently verify source files not supplied in this archive.

Patient identifiers are sequential within each cohort; use cohort + patient as the key. No cell barcodes or metadata cache is included. Codes correspond to S9-S14 Data; source-count sample rows use the corresponding Pn-S1 code. GSE274987 is the historical internal label for the joint human GSE274986/GSE274987 source-study cohort, not a separate independent cohort.

## Full count-level workflow and input requirements

`stage63_revision_preprocessing.py` reads full retained count layers from `public_inputs/processed_inputs/{scGBM_GSE162631_GSE273274.h5ad,snGBM_GSE276177_GSE286419.h5ad,regional_GSE274987_GSE241037.h5ad}` and GSE211376 public raw counts/metadata from `public_inputs/raw_inputs/{GSE211376_raw_counts.tsv.gz,GSE211376_metadata.csv.gz}`. Set `GLIOMA_VALIDATION_ROOT` to the parent of public_inputs, and `PLOS_REVISION_DIR` to a new output directory with analysis, validation, figures and supporting_information subdirectories. Its extraction computes all-gene library totals before retaining the 37-gene panel. Archived upstream notebooks and accession mappings are part of the original S8/S13 materials. These large project inputs are not embedded here; the original small 33-gene H5ADs are insufficient substitutes.

`stage63b_revision_gse162631.py` additionally needs the four original tumor 10x directories under the required environment variable `GSE162631_RAW_ROOT` (public audit rows are coded P1-S1 through P4-S1), containing matrix.mtx.gz, barcodes.tsv.gz and genes.tsv.gz or features.tsv.gz. It matches source counts and uses all source genes for library totals. This route audits the retained cell universe and does not reproduce a new cell-calling/doublet/malignancy pipeline from FASTQ.

Execute stage63, then stage63b. `prepare_meta_inputs()` in stage65 merges new patient contrasts with the frozen baseline tables `claude_outputs/public_only_submission/tables/TableS3a_ligand_contrasts.csv` and `TableS7a_uniform_annotation_patient_contrasts.csv` under GLIOMA_VALIDATION_ROOT. Run stage64 with PLOS_REVISION_DIR as its sole argument; then run stage65 to verify, plot and build S14. Use original study-local IDs for this full source-audit route; mapping to sequential public patient codes is applied at export. The supplied lightweight tables already contain the baseline contrasts, so that route does not require the historical project layout.

## Interpretation

REML pools arithmetic patient-level means, whereas the original signed Stouffer statistic uses directions of cohort medians and rank-test P values. GSE162631 has one positive and two negative patient contrasts, a negative median and a near-zero positive mean. Primary and uniform labels are different analysis populations. Strict filtering and 1,000-count uniform reassignment exclude GSE162631; stronger pooled evidence is not correction of the cohort. Fixed-label and pseudobulk sensitivities have distinct estimands/scales. See the manuscript and S14 for full uncertainty and eligibility reporting.

The code is supplied for unrestricted reuse under the MIT license (LICENSE). Source data retain their original repository/study terms. This directory contains the revision analyses; original analyses and recoded inputs are supplied alongside it in original_analysis/ (S13 File).
