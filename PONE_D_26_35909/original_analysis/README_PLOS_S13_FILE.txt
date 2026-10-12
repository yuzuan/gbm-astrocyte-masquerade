PONE-D-26-35909: original analyses and recoded inputs

Run python reproduce_primary.py from this directory to reproduce the eight primary result tables using the supplied minimal expression files. Results are written to reproduced/ and checked against the supplied reference tables. This does not require the larger original count objects.

Code under code/ includes original analysis and figure routines. source_data/ and external_validation/results/ contain the spatial and protein source tables. The eight upstream notebooks document the original workflows with execution outputs cleared and source-specific clinical dictionaries externalized. Full source preprocessing requires separately acquired inputs, described in RELEASE_NOTES.md.

S8 provenance records describe the original analysis inputs and pre-release notebook snapshots; the current public copies have been recoded and their embedded metadata removed. Numerical scientific results and expression matrices are unchanged. Patient and section codes agree with the revised S9-S14 tables. Code is provided under LICENSE (MIT).

Display-only main-figure revision (12 October 2026)

From this directory, run:
    MPLBACKEND=Agg python code/stage62_plottie_submission_figures.py --main-only --output figures

The renderer requires numpy, pandas, matplotlib and seaborn. It reads frozen results; it does not rerun significance tests, meta-analysis or bootstrap estimation. The original Plottie theme and palette are retained. The output consists of six PNG/PDF figures and canvas-boundary reports. The manuscript upload TIFFs are RGB, 600 dpi and LZW-compressed.

external_validation/results/cptac_gbm/tgfb2_sample_values.csv adds the 109 sample-level TGFB2 values shown in Fig 6a. Values were copied verbatim from the TGFB2 row of the original clean, non-imputed normalized matrix, proteome_mssm_per_gene_clean.v4.0.tsv.gz; no normalization or statistical analysis was repeated. The file contains 99 tumor samples (existing study-local P codes) and 10 normal references (N1-N10). These are distinct groups. Original identifiers and their crosswalk are not included. Figure 4b reads patient differences already present in data/regional_communication_tests.csv. Existing effect sizes, confidence intervals and P/q values are unchanged.
