# PONE-D-26-35909 · Patient-level TGFB2 analysis

**Patient-level public multi-omic analysis supports aggregate astrocyte-like TGFB2 association without consistent evidence for myeloid signaling in glioblastoma**. The revised scientific release dates to 12 October 2026.

## Run from the supplied data

Install the Python dependencies from the repository's requirements file. The revision models also use R with `metafor`. Install Poppler for PDF rendering and inspection. The full Greenwald analysis also uses the R packages `jsonlite` and `pdftools`. Run these commands from the repository root.

```sh
python -m pip install -r requirements.txt
python PONE_D_26_35909/original_analysis/14_复现患者主结果.py
cd PONE_D_26_35909/revision_analysis
jupyter lab 06_复现返修模型.ipynb
```

In Jupyter, select the Python environment used to install the requirements and run all cells. The [primary analysis](original_analysis/README_PLOS_S13_FILE.txt) rebuilds eight patient-level tables from the four supplied expression objects. The [revision notebook and full count workflow](revision_analysis/README.md) fit 16 effect-size models from patient contrasts, check the results independently and regenerate supplementary figures S15–S17. For the plots corresponding to the six [manuscript TIFFs](main_figures/), use the submission renderer listed in the primary-analysis instructions.

The [supporting workbooks](supporting_data/) contain S8, S9, S10, S11, S12 and S14. S13 is the original analysis code package. Patient codes are sequential within each cohort. Match patients by cohort and patient code together. Reconstruction of the original annotation and count processing requires the larger source files documented in each analysis directory. The eight output-cleared upstream notebooks are archived as source records.

Code is available under the MIT license. Public source datasets retain their original repository terms.
