# Recoded PLOS ONE release

Patient identifiers are sequential within cohort (P1, P2, ...), consistent across S9-S14 and both code packages. Spatial sections use patient-linked sequential codes. Cell barcodes have been replaced by C1, C2, ... within each minimal input. Original identifiers and the private crosswalk are not included. Numeric analysis values and minimal expression matrices are unchanged.

The original source-reading scripts require separately acquired public inputs. For the Greenwald input route, provide general/sample_metadata.csv using the schema in input_templates/greenwald_sample_metadata.csv and correspondingly coded matrix filenames. The archived snRNA notebook now reads sample_info.csv and snrna_sample_info.csv from its local input folder instead of embedding clinical dictionaries. These are provenance workflows; the full upstream inputs are not supplied. The original-label tables and figures can be reproduced from the supplied minimal inputs and source tables; full count-level revision analysis requires the larger retained-count objects documented in the revision package.

Patient coding is study-local pseudonymization of derived public data, not a claim that the underlying source repositories have been altered. Anatomical tumor regions and analytical spatial coordinates needed for reproduction are retained.
