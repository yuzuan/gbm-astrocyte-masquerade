# Data sources for the original study figures

The six CSV files in this directory transcribe values, labels, and edges stated explicitly in files tracked at commit `a63d6b4`. Their numbers identify the original panel sources. They contain no newly calculated expression values, patient records, or cell identifiers. Use `git show a63d6b4:<filename>` to inspect a cited file at that commit.

| Source table | Evidence at `a63d6b4` | What each row represents |
|---|---|---|
| `01_cohort_sizes_and_astrocyte_cnv_rates.csv` | `stage20_figure_1.py`, lines 79–80 and 104–112, and the Cohorts table in `README.md` | The cell or nucleus count for a whole cohort, the patient count and platform reported in the original figure, and the percentage of that cohort’s astrocyte group positive for the CNV signature. Only the percentage uses the astrocyte-group denominator. |
| `02_reference_cell_cnv_rates.csv` | `stage21_figure_2.py`, lines 55–65 | Observed reference-cell rates for the two pooled GEO cohorts. The original figure marks the scGBM oligodendrocyte rate as n/a. Equivalent columns for the discovery cohort and GSE174554 come from their analysis outputs. |
| `03_tumor_peritumor_cnv_rates.csv` | `stage21_figure_2.py`, lines 71–86 | Tumor and peritumoral percentages for each of two cohorts. This section of the original script gives neither the tissue-specific denominators nor a procedure for estimating thresholds separately by tissue. |
| `05_transcription_factor_activity_differences.csv` | `stage22_figure_3.py`, lines 86–93 and 133–139 | Nine differences in TF activity between AC-like cells and the mean of the other malignant subtypes. Fig3b displays the values recorded in the original code. |
| `07_hypothesized_communication_edges.csv` | `stage23_figure_4.py`, lines 104–107 and 147–162 | Candidate connections specified by the original schematic. Fig4c draws these as dashed lines. Node sizes and positions illustrate the model. |
| `11_liana_ligand_ranks.csv` | `stage25_figure_6.py`, lines 56–67 | The five ligands and integer ranks given in the original code. When stage 08 results are available, Fig6c reads `AC_rank` by ligand name. |

## Local inputs for each figure

The figure scripts locate the project root relative to `00a_plotting.py`. They prefer current analysis outputs under `03_result/`. Place files from the **Original archive** column under `data/intermediate/` relative to the same root. A file that cannot be read or lacks required fields raises an error.

| Figure or panel | Current analysis outputs, relative to `03_result/` | Original archive |
|---|---|---|
| Fig1, Fig2a, and CNV panel of the graphical abstract | `01_sngbm_immune_reference_cnv/01_sngbm_immune_reference_cnv.json`<br>`03_gse211376_marker_defined_astrocytes/01_gse211376_marker_defined_astrocyte_cnv.json`<br>`04_gse174554_clustering_and_markers/01_gse174554_clusters.npz`<br>`05_gse174554_cluster_cnv/02_gse174554_cluster_cnv.json` | `noncircular_cnv_validation.json`, `gse211376_marker_noncircular.json`, `gse174554_cluster.npz`, `gse174554_leiden_cnv.json`. Fig1 can also be drawn independently using the cohort sizes and overall rates given in the original code. |
| Fig2b/c | Both stage 03 and stage 05 outputs listed above are required. Stage 01 supplies the snGBM reference rates when available. | The corresponding stage 03 and 05 JSON files listed above. Fig2d uses the tissue rates in source table 03. |
| Fig3a | Supply the original stage22 summary with `category`, `gene`, and the five `*_mean` expression columns. Root stage 13 uses tumor and peritumoral groups and cannot supply the five cell-group means. | `scgbm_ac_validation_summary.csv` |
| Fig4a/b | `07_scgbm_communication_and_mixture_adjustment/01_communication_and_mixture_adjustment.json` | The `tam_scores` field of `communication_analysis_results.json`. |
| Fig5 and the sender-filtering panel of the graphical abstract | `09_gse211376_sender_filtering/01_sender_filtering_scores.json`<br>`10_gse174554_sender_filtering/01_sender_filtering_scores.json` | `gse211376_communication_decontam.json`, `gse174554_communication_decontam.json` |
| Fig6a/b | Run stages 11 and 12 and use `12_gse211376_subtype_expression_and_ranks/02_cross_cohort_ac_ranks.csv`. | `fig6_concordance_gse211376_vs_scgbm.csv` |
| Fig6c | `08_sngbm_ligand_ranks/01_liana_ligand_ranks.json`. Source table 11 supplies the original ranks if the analysis output and archive are absent. | `liana_validation.json` |

To reproduce historical Fig5 or the graphical abstract, use stage 09 for the GSE211376 scores. For GSE174554, use the stage 10 `01_sender_filtering_scores.json` produced from the original frozen partition with primary astrocyte cluster 5 verified in stage 05, or supply the communication archive from that same partition. The partition from a fresh stage 04 clustering run cannot enter this historical 05/10 analysis directly. The plotting scripts take scores from the supplied input. Fig5 reads the upstream `pct_change` field. The abstract requires only TGFB2–TGFBR2 and calculates its label from the two displayed scores, `(after - before) / before * 100`, following lines 58–68 of `stage26_graphical_abstract.py`.

The original scripts refer to locally supplied external statistics in `stage22_figure_3.py` (lines 76–84), `stage23_figure_4.py` (lines 84–100), `stage24_figure_5.py` (lines 21–36 and 43–50), and `stage25_figure_6.py` (lines 19–27 and 39–44).

The figure scripts use Python with Matplotlib, NumPy, pandas, and Pillow, and the Poppler commands `pdfinfo`, `pdftotext`, and `pdftoppm`. Each of the seven entry points writes a single-page PDF under `03_result/<number>_<figure_name>/` and renders the matching PNG from that PDF at the 300 dpi specified in the original plotting code.
