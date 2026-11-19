# Upstream processing notebooks

The eight notebooks in this directory are English renditions of the output-cleared notebooks released with the original analysis. They document the local processing and annotation steps used to build the AnnData inputs from public GEO data. The renditions preserve cell order and cleared execution outputs and counts. The release at commit `0758314` provides the original text, including its historical path references. [Original notebook directory at commit 0758314](https://github.com/yuzuan/gbm-astrocyte-masquerade/tree/0758314b38a8db326e5196525c8137810769c2bc/PONE_D_26_35909/original_analysis/code/upstream_processing)

## Recovering historical snapshots

Stage `04_audit_processed_inputs.py` reads the eight original working notebooks from `PUBLIC_INPUT_ROOT/upstream_source/` using the literal source paths in `source_data/TableS9c_upstream_notebook_snapshots.csv`. It preserves each source cell and its order, clears code-cell outputs and execution counts, and writes the recovered notebooks and a README under `code/upstream_processing/` in the output package. Its `data/upstream_notebook_snapshots.csv` records the historical source paths, source sizes, saved snapshot sizes and snapshot digests. The published CSV inventories and `supporting_data/S8_Data.xlsx` retain the original source metadata. Those sizes and digests describe the recovered historical snapshots. The English renditions are a separate release artifact.

For an isolated external working package, set `PUBLIC_ONLY_PACKAGE` to its directory. When the destination resolves to the shipped analysis package, including through a symbolic link, the stage writes under `01_cohort_reanalysis/` inside that package. The shipped English notebook files remain available in their original directory.

## Processed inputs and cohort membership

The annotation objects evolved over several project runs. Rebuilding the exact processed objects requires the original inputs and annotation history. `TableS9a_processed_input_objects.csv` identifies the objects by SHA-256. These objects require a permanent deposit with the final code release before the package can support end-to-end reproduction from GEO.

The statistical analysis separates cohorts by the `dataset` metadata field before calculating patient-level summaries. Several accessions share storage or preprocessing. The processed objects are common upstream dependencies, and the patient-level calculations retain the accession-specific cohort definitions.

The regional notebook and AnnData object use the legacy value `dataset == "GSE274987"` for the joint Wu et al. source-study cohort. `Platform == "ParseBio"` identifies GSE274987 (4 patients, 11 regional profiles). `Platform == "Multiome"` identifies GSE274986 (6 patients, 26 RNA profiles). The analysis counts these 10 patients as one source-study cohort. `TableS9b_processed_cohort_membership.csv` records the mapping to the two source accessions.
