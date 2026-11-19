# ======================================================================
# 00a_cell_communication.py - shared functions for the GBM astrocyte-like cell analyses
# Used by stages 07 through 13 of the original root analysis.
#
# Purpose
# Share scGBM group definitions, sampling rules, expression matrix access and output checks.
# Review all calling analyses when changing these functions.
# Consult each function at the point where an analysis calls it.
# Ctrl-click a function name (Cmd-click on macOS) to open its definition in the editor.
# Definitions can also be found by searching for the function name.

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy import sparse


def scgbm_state_masks(minor_cluster):
    """Return masks for four scGBM sender states from the subtype labels, pooling the two original NPC labels."""
    masks = {
        "AC": minor_cluster.eq("AC"),
        "MES1": minor_cluster.eq("MES1"),
        "OPC": minor_cluster.eq("OPC"),
        "NPC": minor_cluster.isin(["NPC1", "NPC2"]),
    }
    return masks


def sampled_scgbm_positions(cell_mask):
    """Return expression matrix row indices in the historical sampling order for cells selected by the mask."""
    positions = np.flatnonzero(cell_mask)

    # Sample groups larger than 5,000. Restart MT19937 at seed 42 for each group to retain the cells and means.
    if len(positions) > 5000:
        generator = np.random.RandomState(42)
        positions = generator.choice(positions, 5000, replace=False)

    return positions


def cell_expression_table(dataset, positions, genes):
    """Return cells by genes from X. Empty selections and missing or duplicate gene identifiers raise an error."""
    if not len(positions) or not len(genes):
        raise ValueError("Expression means require at least one cell and one available gene.")
    gene_positions = dataset.var_names.get_indexer(genes)
    if (gene_positions < 0).any():
        raise ValueError("The expression matrix is missing a requested gene.")

    # Select cell rows first, then gene columns, to keep every cell-gene combination in the sampled row order.
    cell_matrix = dataset.X[positions, :]
    selected_matrix = cell_matrix[:, gene_positions]
    if sparse.issparse(selected_matrix):
        selected_matrix = selected_matrix.toarray(order="F")
    else:
        selected_matrix = np.asfortranarray(selected_matrix)

    # Store each gene contiguously so its float32 mean uses the original summation order.
    expression_table = pd.DataFrame(selected_matrix, columns=genes)
    return expression_table


def scgbm_product_scores(dataset, receiver):
    """Return four sender-state scores per ligand-receptor pair for a scGBM receiver. Skip unmeasured pairs."""
    if receiver == "TAM/Macrophage":
        pairs = [("TGFB2", "TGFBR1"), ("TGFB2", "TGFBR2"), ("CCL2", "CCR2"), ("CSF1", "CSF1R"), ("CXCL12", "CXCR4")]
    elif receiver == "T/NK":
        pairs = [("CD274", "PDCD1"), ("NECTIN2", "TIGIT"), ("PVR", "TIGIT"), ("PDCD1LG2", "PDCD1")]
    else:
        raise ValueError(f"The signaling panel does not define a receiver group named {receiver}.")

    # Both genes must be measured. The published results distinguish a missing pair from zero mean expression.
    available_pairs = []
    for ligand, receptor in pairs:
        if ligand in dataset.var_names and receptor in dataset.var_names:
            available_pairs.append((ligand, receptor))

    scores = {}
    if not available_pairs:
        return scores

    # All four sender states share the receiver means. Average each gene's vector at its original precision.
    receptors = list(dict.fromkeys(receptor for ligand, receptor in available_pairs))
    positions = sampled_scgbm_positions(dataset.obs["Major_category"].eq(receiver))
    receiver_table = cell_expression_table(dataset, positions, receptors)
    receptor_means = {gene: float(np.mean(receiver_table[gene].to_numpy())) for gene in receptors}

    # Keep the published JSON's case-sensitive subtype labels. Sample each state separately and multiply Python floats.
    ligands = list(dict.fromkeys(ligand for ligand, receptor in available_pairs))
    for state, mask in scgbm_state_masks(dataset.obs["Minor_cluster"]).items():
        sender_table = cell_expression_table(dataset, sampled_scgbm_positions(mask), ligands)
        for ligand, receptor in available_pairs:
            pair_name = f"{ligand}-{receptor}"
            ligand_mean = float(np.mean(sender_table[ligand].to_numpy()))
            scores.setdefault(pair_name, {})[state] = ligand_mean * receptor_means[receptor]

    return scores


def count_matrix_cells(stream):
    """Read GSE211376 cell IDs from the first line in their original order. Reject duplicate or empty IDs."""
    cells = pd.Index(stream.readline().rstrip("\r\n").split("\t"))
    if not cells.is_unique or "" in cells:
        raise ValueError("The count file header must contain one unique, nonempty cell ID per column.")
    return cells


def read_gene_counts(line, cell_count):
    """Return the gene and its count vector. Reject invalid gene names, column counts and count values."""
    gene, separator, values = line.partition("\t")
    counts = np.fromstring(values, sep="\t")
    if not gene or not separator or len(counts) != cell_count:
        raise ValueError(f"The count row for {gene!r} does not match the {cell_count} cells in the header.")
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError(f"Raw counts for {gene} contain nonfinite or negative values.")
    result = gene, counts
    return result


def library_size_factors(total_counts):
    """Return all-gene library sizes divided by 10,000, with a small historical divisor for zero libraries."""
    factors = total_counts / 1e4
    factors[factors == 0] = 1e-9
    return factors


def write_gene_table(path, result_table, expected_genes):
    """Write one CSV row per gene. Check uniqueness, panel order and all saved columns and values. No return value."""
    if not result_table["gene"].is_unique or result_table["gene"].tolist() != list(expected_genes):
        raise ValueError("Output genes, order or row count differ from the input panel.")

    # Stage the file beside its target and publish it after checking. Preserve the category "NA" and missing numbers.
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_path.parent) as folder:
        pending_path = Path(folder) / output_path.name
        result_table.to_csv(pending_path, index=False, encoding="utf-8-sig")
        saved_table = pd.read_csv(pending_path, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
        pd.testing.assert_frame_equal(
            saved_table, result_table.reset_index(drop=True), check_dtype=False, check_exact=False,
            rtol=1e-7, atol=1e-12,
        )
        pending_path.replace(output_path)
