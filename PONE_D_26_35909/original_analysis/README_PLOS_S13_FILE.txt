PONE-D-26-35909: original analyses and recoded inputs

Run python reproduce_primary.py from this directory to reproduce the eight primary result tables using the supplied minimal expression files. Results are written to reproduced/ and checked against the supplied reference tables. This does not require the larger original count objects.

Code under code/ includes original analysis and figure routines. source_data/ and external_validation/results/ contain the spatial and protein source tables. The eight upstream notebooks document the original workflows with execution outputs cleared and source-specific clinical dictionaries externalized. Full source preprocessing requires separately acquired inputs, described in RELEASE_NOTES.md.

S8 provenance records describe the original analysis inputs and pre-release notebook snapshots; the current public copies have been recoded and their embedded metadata removed. Numerical scientific results and expression matrices are unchanged. Patient and section codes agree with the revised S9-S14 tables. Code is provided under LICENSE (MIT).
