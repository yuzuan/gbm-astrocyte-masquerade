# PONE-D-26-35909: patient-level TGFB2 analysis

Current manuscript: **Patient-level public multi-omic analysis supports aggregate astrocyte-like TGFB2 association without consistent evidence for myeloid signaling in glioblastoma**. Revision release, 12 October 2026.

This directory is the manuscript-specific release. The repository-root scripts relate to an earlier project and are not the pipeline used for this revised manuscript.

- `original_analysis/`: original analysis code, recoded minimal expression inputs, source tables and upstream workflows. Run `python reproduce_primary.py` in that directory to reproduce eight primary result tables.
- `revision_analysis/`: new random-effects, preprocessing and source-count verification scripts. Run `Revision_analysis.ipynb` to reproduce the 16 effect-size models and supplementary figures from supplied patient-level inputs.
- `supporting_data/`: revised S8-S14 workbooks, using the same study-local patient codes as both code directories.

Reproduction checks passed for the eight primary tables and all 16 revision models. Patient identifiers and cell barcodes are recoded; original identifier crosswalks are not distributed. Full upstream count-level processing requires the separately acquired inputs documented in each directory. Reproduction of supplied patient-level statistical results is distinct from rerunning source cell selection or malignant annotation.

Code is available for unrestricted reuse under the MIT license. Public source datasets retain their original repository terms. No manuscript correspondence or private review files are included.
