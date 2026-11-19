# Analysis directories and file contracts

| Directory | Entry point or inputs | Generated results |
|---|---|---|
| Repository root | Numbered scripts 01–13 read earlier-study GEO matrices, AnnData objects and annotations under `data/`. Stage 04 recomputes an independent partition. Stages 05 and 10 reproduce the archived partition and use the same cells and assignments. | A numbered subdirectory per stage in `03_result/`. |
| Repository root | Scripts 14–20 use analysis outputs, older intermediate files if available, and the small historical [figure source tables](01_figure_sources/). | PDF masters and derived PNG figures in `03_result/`. |
| [Original PLOS ONE analysis](PONE_D_26_35909/original_analysis/) | [14_reproduce_primary_results.py](PONE_D_26_35909/original_analysis/14_reproduce_primary_results.py) reads the release's minimal expression objects and compares against its original CSVs. | Eight numbered tables and a per-table check in `14_primary_results/`. |
| Original analysis, `code/` | Stages 01–05 process the original full inputs, Stage 11 extracts the minimal panel. Stages 06–10 address separate spatial and protein datasets. Stages 12 and 13 render the earlier and submission figure layouts. | Use a separate `PUBLIC_ONLY_PACKAGE` for a full upstream rebuild. External analyses accept explicit input and output paths. Both figure renderers accept `--output`. |
| [Revision analysis](PONE_D_26_35909/revision_analysis/) | [06_reproduce_revision_models.ipynb](PONE_D_26_35909/revision_analysis/06_reproduce_revision_models.ipynb) reads supplied patient contrasts and audit tables. Stages 01a–05c provide a separate route from full-gene count data. | The full route writes below `03_result/` or `PLOS_REVISION_DIR`. The notebook writes its inputs, fitted models and supplementary figures in a separate `06_public_reproduction/` subdirectory. |

The original and revision scripts import the shared PLOS ONE [figure style and exporter](PONE_D_26_35909/00a_figure_style.py). Reference tables, workbooks and manuscript figure filenames retain their release names so that existing citations, sheet names and table identifiers resolve. The eight [upstream processing notebooks](PONE_D_26_35909/original_analysis/code/upstream_processing/) document the historical workflows in English. Their README gives the source inputs and the archive location for the original notebooks.

The root analyses and the PLOS ONE patient-level analyses have different populations, endpoints and input layouts. Use the entry point for the study you want to reproduce.
