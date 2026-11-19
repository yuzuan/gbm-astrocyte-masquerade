# ======================================================================
# 00a_expression_and_clustering.py | Shared functions for the GBM astrocyte analyses
# Root stages 01 through 06 use these functions to reproduce the historical CNV analyses.
#
# Purpose
# Share gene sets, per-cell calculations, and readers and writers across cohorts.
# Each stage defines its cell-selection criteria and cohort-specific denominators.
# The functions can be read alongside the analyses that call them.
# In an editor, Ctrl-click a function name (Cmd-click on macOS) or search for its definition.
# Start at the call in the stage of interest.

import gzip
import json
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import pandas as pd
from scipy.io import mminfo, mmread


def chromosome_gene_sets():
    """Read chr7 and chr10 gene names from the scGBM var category codes. Missing annotations raise an error."""
    with h5py.File("data/external/scGBM/adata2.h5ad", "r") as reference:
        genes = reference["var/_index"].asstr()[:]
        labels = reference["var/chromosome/categories"].asstr()[:].tolist()
        codes = reference["var/chromosome/codes"][:]

    # Missing categories use code -1. Comparing codes directly keeps missing genes out of the last chromosome category.
    gene_sets = {}
    for chromosome in ("chr7", "chr10"):
        gene_sets[chromosome] = set(genes[codes == labels.index(chromosome)])
    if gene_sets["chr7"] & gene_sets["chr10"]:
        raise ValueError("A gene is assigned to both chr7 and chr10 in the scGBM annotations.")
    return gene_sets


# GSE211376 and GSE174554 share these markers. Stage 03 adds two complement genes to the GSE211376 myeloid panel.
lineage_genes = {
    "astro": ["GFAP", "AQP4", "SLC1A3", "S100B", "ALDOC", "CLU"],
    "imm": ["PTPRC", "CD68", "CD14", "AIF1", "ITGAM", "CSF1R"],
    "oligo": ["MOG", "PLP1", "MBP", "MOBP"],
}


def gse211376_expression(gene_sets, count_dtype):
    """Stream gene-by-cell counts into per-cell gene-set scores. Check row widths, barcodes, and gene matches."""
    with gzip.open("data/intermediate/GSE211376_raw_counts.tsv.gz", "rt", encoding="utf-8") as counts:
        header = counts.readline().rstrip("\r\n").split("\t")
        first_gene = next(counts)
        cell_count = first_gene.count("\t")

        # The GEO header has only barcodes. The first count row also identifies TSVs with a gene-column header.
        if len(header) == cell_count + 1:
            header = header[1:]
        if len(header) != cell_count or not cell_count or "" in header or len(set(header)) != cell_count:
            raise ValueError("GSE211376 count barcodes repeat, are empty, or differ in count from the first data row.")
        library_counts = np.zeros(cell_count)
        group_counts = {name: np.zeros(cell_count) for name in gene_sets}
        matched_genes = dict.fromkeys(gene_sets, 0)

        # All genes contribute to each cell's library size. A gene can belong to both a chromosome and a lineage panel.
        for line_number, line in enumerate(chain([first_gene], counts), start=2):
            gene, _, count_text = line.partition("\t")
            values = np.fromstring(count_text, sep="\t", dtype=count_dtype)
            if values.size != cell_count or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"GSE211376 row {line_number} ({gene}) has missing or invalid counts.")
            library_counts += values
            for name, genes in gene_sets.items():
                if gene in genes:
                    group_counts[name] += values
                    matched_genes[name] += 1

    # Scale each cell to 10,000 counts, divide by the number of matched genes, then take log1p.
    # All-zero libraries retain the original small divisor.
    library_scale = library_counts / 1e4
    library_scale[library_scale == 0] = 1e-9
    expression_table = pd.DataFrame(index=pd.Index(header, name="cell_id"))
    for name, values in group_counts.items():
        if matched_genes[name] == 0:
            raise ValueError(f"The GSE211376 count table contains no genes from the {name} set.")
        expression_table[name] = np.log1p(values / library_scale / matched_genes[name])
    return expression_table


def gse211376_metadata(cell_ids):
    """Align the GEO metadata, one row per barcode, to the count matrix. Reject duplicate barcodes or missing labels."""
    metadata_table = pd.read_csv("data/intermediate/GSE211376_metadata.csv.gz", index_col=0)
    if not metadata_table.index.is_unique:
        raise ValueError("The GSE211376 metadata contain duplicate cell barcodes.")
    aligned_table = metadata_table.reindex(cell_ids)
    if aligned_table["predicted.high_hierarchy"].isna().any():
        raise ValueError("Some GSE211376 count barcodes lack an author-assigned cell annotation.")
    return aligned_table


def centered_cnv(chr7, chr10, center_mask, threshold_mask):
    """Return centered per-cell +7/-10 scores, the strict cutoff, and both reference means.

    Empty reference groups and nonfinite scores raise an error.
    """
    if not np.any(center_mask) or not np.any(threshold_mask):
        raise ValueError("The reference for chromosome centering or the CNV threshold contains no cells.")
    center7 = chr7[center_mask].mean()
    center10 = chr10[center_mask].mean()
    scores = (chr7 - center7) - (chr10 - center10)
    if not np.isfinite(scores).all():
        raise ValueError("Per-cell CNV scores contain missing or infinite values.")

    # All three original analyses compare scores with the unrounded 95th percentile of their reference cells.
    threshold = np.percentile(scores[threshold_mask], 95)
    result = scores, threshold, center7, center10
    return result


def positive_summary(positive, mask):
    """Return the selected cell count and percent positive to one decimal place. Empty groups have a JSON null rate."""
    selected = np.asarray(positive)[mask]
    percent = None
    if selected.size:
        percent = round(float(100 * selected.mean()), 1)
    result = {"n": int(selected.size), "cnv_pos_pct": percent}
    return result


def inhouse_cnv_summary(score_path, metadata_path):
    """Read in-house cell scores and conditions. Return both astrocyte rates and the reference-group rates.

    Missing or mismatched cell data raise an error.
    """
    score_table = pd.read_csv(score_path, index_col=0, dtype={"cnv_pos": "boolean"})
    metadata_table = pd.read_csv(metadata_path, index_col=0)
    if not score_table.index.is_unique or not metadata_table.index.is_unique:
        raise ValueError("The in-house score table or metadata contain duplicate cell barcodes.")
    conditions = metadata_table["condition"].reindex(score_table.index)
    if conditions.isna().any() or score_table[["chr7", "chr10", "cell_type", "cnv_pos"]].isna().any().any():
        raise ValueError("In-house cells have unmatched conditions, missing scores, or missing cell types.")

    # Myeloid and T_NK cells across both tissues set the strict cutoff. Upstream cnv_pos defines the loose calls.
    cell_types = score_table["cell_type"]
    immune = cell_types.isin(["Myeloid", "T_NK"])
    scores = score_table["chr7"] - score_table["chr10"]
    if not immune.any() or not np.isfinite(scores).all():
        raise ValueError("The in-house cohort lacks immune reference cells or contains nonfinite CNV scores.")
    threshold = np.quantile(scores[immune], 0.95)
    positive = scores > threshold

    # Only GBM astrocytes enter the primary denominator. The historical peritumoral field uses the Normal label.
    # The metadata do not specify a more precise histological site.
    astrocytes = cell_types == "Astrocyte"
    gbm = astrocytes & conditions.eq("GBM")
    strict = positive_summary(positive, gbm)
    if strict["n"] == 0:
        raise ValueError("The in-house cohort contains no GBM astrocytes.")
    loose = positive_summary(score_table["cnv_pos"].to_numpy(dtype=bool), gbm)
    normal = positive_summary(positive, astrocytes & conditions.eq("Normal"))

    # Each reference denominator counts every cell of its type, including oligodendrocytes assessed after thresholding.
    reference_groups = {}
    for cell_type in ("Myeloid", "T_NK", "Oligodendrocyte"):
        reference_groups[cell_type] = positive_summary(positive, cell_types == cell_type)
    result = {
        "n": strict["n"],
        "strict_matched_pct": strict["cnv_pos_pct"],
        "loose_cnv_pos_pct": loose["cnv_pos_pct"],
        "strict_ref_FPR": {name: group["cnv_pos_pct"] for name, group in reference_groups.items()},
        "peritumoral_astro_strict_pct": normal["cnv_pos_pct"],
        "strict_threshold": float(threshold),
        "reference_n": {name: group["n"] for name, group in reference_groups.items()},
        "normal_astro_n": normal["n"],
    }
    return result


def feature_names(feature_path):
    """Read gene names from the first RNA feature-table column in matrix row order. A missing file raises an error."""
    with gzip.open(feature_path, "rt", encoding="utf-8") as features:
        genes = [line.rstrip("\r\n").split("\t")[0] for line in features]
    return genes


def gse174554_libraries():
    """Return the common gene order, RNA matrix paths, and matching cell table. Check gene order and barcode counts."""
    directory = Path("data/GSE174554")
    references = sorted(directory.glob("*_features.tsv.gz"))
    if not references:
        raise FileNotFoundError(f"No RNA feature table was found in {directory}.")
    genes = feature_names(references[0])
    matrices = []
    library_tables = []

    # Use the original filename order and skip the damaged library. Matrix headers identify files with a different
    # feature count (snATAC in this release).
    for matrix_path in sorted(directory.glob("*_matrix.mtx.gz")):
        prefix = matrix_path.name.removesuffix("_matrix.mtx.gz")
        if "GSM5319529_SF8963" in prefix:
            continue
        barcode_path = directory / f"{prefix}_barcodes.tsv.gz"
        if not barcode_path.is_file():
            continue
        gene_count, cell_count, *_ = mminfo(str(matrix_path))
        if gene_count != len(genes):
            continue

        # Match every gene in order before stacking RNA libraries. Full prefixes distinguish batches of a sample.
        if feature_names(directory / f"{prefix}_features.tsv.gz") != genes:
            raise ValueError(f"{prefix} has a different gene order from the reference RNA feature table.")
        with gzip.open(barcode_path, "rt", encoding="utf-8") as barcodes:
            barcode_ids = [line.strip() for line in barcodes]
        sample = next((part for part in prefix.split("_") if part.startswith("SF")), prefix)
        library_table = pd.DataFrame({"barcode": barcode_ids, "sample": sample})
        library_table["cell_id"] = prefix + ":" + library_table["barcode"]
        library_table["tissue_id"] = sample + "_" + library_table["barcode"].str.split("-").str[0]
        if len(library_table) != cell_count:
            raise ValueError(f"The barcode count for {prefix} differs from its RNA matrix column count.")
        library_tables.append(library_table[["cell_id", "sample", "tissue_id"]])
        matrices.append(matrix_path)

    # Keep library and barcode order. tissue_id links historical tissue labels. The full cell_id identifies each row.
    if not matrices:
        raise ValueError("No GSE174554 RNA libraries meet the historical inclusion criteria.")
    cell_table_total = pd.concat(library_tables, ignore_index=True)
    if not cell_table_total["cell_id"].is_unique:
        raise ValueError("GSE174554 cell identifiers built from library prefixes and full barcodes are duplicated.")
    result = np.asarray(genes), matrices, cell_table_total
    return result


def read_cell_matrix(matrix_path):
    """Convert one Matrix Market gene-by-barcode library to the historical float32 cell-by-gene sparse matrix."""
    gene_matrix = mmread(matrix_path).tocsr()
    cell_matrix = gene_matrix.T.astype(np.float32)
    return cell_matrix


def result_or_frozen(result_path, old_name):
    """Try the new output, the frozen data/intermediate file, then the legacy root file. Raise if all are missing."""
    for path in (result_path, Path("data/intermediate") / old_name, Path(old_name)):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Neither {result_path} nor the legacy file {old_name} was found.")


def write_checked_json(path, result):
    """Write UTF-8 JSON and read back all fields and values. Reject nonfinite numbers or mismatches. Return None."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temporary:
        draft = Path(temporary) / path.name
        draft.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if json.loads(draft.read_text(encoding="utf-8")) != result:
            raise ValueError(f"Fields or values in {path.name} changed when read back.")
        draft.replace(path)


def write_checked_arrays(path, arrays):
    """Compress named arrays. Verify keys, shapes, dtypes, and all values before publishing. Return None."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temporary:
        draft = Path(temporary) / path.name
        np.savez_compressed(draft, **arrays)
        with np.load(draft, allow_pickle=False) as stored:
            if set(stored.files) != set(arrays):
                raise ValueError(f"The arrays read back from {path.name} have missing or unexpected keys.")
            for name, expected in arrays.items():
                observed = stored[name]
                if observed.dtype != expected.dtype or not np.array_equal(observed, expected):
                    raise ValueError(f"In {path.name}, array {name} differs in shape, dtype, or element values.")
        draft.replace(path)
