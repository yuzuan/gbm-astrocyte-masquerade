# Upstream processing snapshots

These output-cleared notebooks document the local processing and annotation path used to build the project AnnData inputs from public GEO data.

They are provenance snapshots, not a claim that the current processed objects can be reconstructed byte-for-byte from these notebooks alone. The annotation objects evolved through iterative project runs. The exact processed inputs are identified by SHA-256 in `TableS9a_processed_input_objects.csv` and must be deposited with the final code release. Until a permanent repository DOI is inserted, the submission package is not an end-to-end GEO-to-result archive.

The statistical analysis separates cohorts by the `dataset` metadata field before calculating patient-level summaries. Shared storage or shared preprocessing does not merge patients across accessions, but it is disclosed because the processed objects are common upstream dependencies.

The regional notebook and AnnData object use the legacy value `dataset == "GSE274987"` for a joint Wu et al. cohort assembled from two source accessions. `Platform == "ParseBio"` maps to GSE274987 (4 patients, 11 regional profiles), whereas `Platform == "Multiome"` maps to GSE274986 (6 patients, 26 RNA profiles). The current analysis retains this joint 10-patient source-study cohort and does not count the two accessions as independent replications. The source-accession mapping is audited in `TableS9b_processed_cohort_membership.csv`.
