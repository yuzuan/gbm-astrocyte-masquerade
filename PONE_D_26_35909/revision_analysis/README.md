# Revision analysis reproduction

The sensitivity analyses for PONE-D-26-35909 are described in the [October 2026 analysis plan](ANALYSIS_PLAN.md).

## Refit the published models

Open [06_复现返修模型.ipynb](06_复现返修模型.ipynb) in this directory and run all cells. Use the Python environment with the dependencies from the repository root, R with `metafor`, and Poppler's `pdftocairo`. `Rscript` must be on PATH, or set `RSCRIPT` to the executable. The archived [software versions](validation/) document the original run.

The notebook takes the supplied [patient contrasts and audit tables](analysis/) as inputs. It fits the 16 REML models with modified Knapp–Hartung intervals, checks the fits independently in SciPy, compares the three model tables against their published references, and regenerates S15–S17. To export the S14 workbook from those results, run `PLOS_REVISION_DIR=03_result/06_公开复现 python code/05c_导出患者补充工作簿.py` with the default paths. The notebook always writes under `06_公开复现/` within `03_result/` or within the output root selected by `PLOS_REVISION_DIR`. The full-count stages use that root directly.

Patient codes apply within each cohort. The source-count summaries use the corresponding Pn-S1 codes. GSE274987 is the historical analysis label for the joint GSE274986/GSE274987 source-study cohort.

## Recompute from source counts

Set `GLIOMA_VALIDATION_ROOT` to the parent of `public_inputs/`. Stage 01a requires these five files:

- `public_inputs/processed_inputs/scGBM_GSE162631_GSE273274.h5ad`
- `public_inputs/processed_inputs/snGBM_GSE276177_GSE286419.h5ad`
- `public_inputs/processed_inputs/regional_GSE274987_GSE241037.h5ad`
- `public_inputs/raw_inputs/GSE211376_raw_counts.tsv.gz`
- `public_inputs/raw_inputs/GSE211376_metadata.csv.gz`

Set `GSE162631_RAW_ROOT` to the directory containing the four original tumor 10x directories. Each contains a Matrix Market count matrix, barcode list and feature or gene list. Stage 03 also reads the two frozen patient tables `TableS3a_ligand_contrasts.csv` and `TableS7a_uniform_annotation_patient_contrasts.csv` from `claude_outputs/public_only_submission/tables/` under `GLIOMA_VALIDATION_ROOT`. These inputs must use consistent source-study identifiers. The public notebook already contains the required baseline contrasts.

From this directory, run the following in order with `PLOS_REVISION_DIR` set to your chosen full-run output directory.

```sh
python code/01a_读取全基因计数.py
python code/01b_比较预处理与注释.py
python code/02_核对GSE162631来源计数.py
python code/03_整理患者模型输入.py
Rscript code/04_拟合随机效应模型.R "$PLOS_REVISION_DIR"
python code/05a_核对患者与模型结果.py
python code/05b_绘制返修补充图.py
python code/05c_导出患者补充工作簿.py
```

The count workflow uses every retained gene to calculate library totals before selecting the marker panel. It audits existing cell inclusion and labels. The four minimal expression files in the original release support the primary-table reproduction and lack the full count denominator needed here.

## Interpretation

REML pools arithmetic patient means. The signed Stouffer analysis combines directions of cohort medians with rank-test P values. GSE162631 has one positive and two negative patient contrasts, a negative median and a near-zero positive mean. The primary- and uniform-label analyses use different populations. Strict filtering and reassignment at 1,000 counts exclude GSE162631. Fixed-label and pseudobulk analyses use different estimands or scales. See the manuscript and S14 for uncertainty and eligibility.
