# ==============================================================================
# 00a_spatial_analysis.py - Shared functions for spatial and protein validation in GBM
# External validation in the original analysis for PONE-D-26-35909.
#
# Purpose
# Calculate within-section correlations, summarize patient results, test ligands and write spatial tables.
# Each cohort specifies its own adjustment. Both CPTAC analyses use the same local-first input lookup.
# Consult these functions when they are called from an analysis script.
# Ctrl-click a function name (Cmd-click on macOS) to jump to its definition in an editor.
# Use the call site to locate the corresponding function.

from __future__ import annotations

import gzip
import hashlib
import subprocess
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

# Keep the published panel order. CPTAC permutations and bootstrap draws share one random stream in this order.
ligands = ["TGFB2", "TGFB1", "ANXA1", "SPP1", "CSF1", "CCL2"]
receptors = ["TGFBR1", "TGFBR2"]
ac_markers = ["GFAP", "S100B", "HOPX", "SLC1A3", "MLC1"]


def cptac_input(local_path, url, cache_dir, required_fields):
    """Return the local table or validated HTTPS gzip cache. Fail on missing input, download or content errors."""
    if local_path.is_file():
        return local_path
    if not url:
        raise FileNotFoundError(f"Supply a local input containing {local_path.name} or its direct HTTPS URL")
    if urlsplit(url).scheme != "https" or not urlsplit(url).hostname:
        raise ValueError("The CPTAC source must be a complete HTTPS URL")

    # Cache each source separately. A hit avoids network access and rewriting. Results omit the source URL.
    source_id = hashlib.sha256(url.encode("utf-8")).hexdigest()
    cached_path = cache_dir / source_id / local_path.name
    if cached_path.is_file():
        return cached_path
    cached_path.parent.mkdir(parents=True, exist_ok=True)

    # Restrict redirect protocols. Read through gzip EOF to check its checksum and header before publishing the cache.
    with TemporaryDirectory(dir=cached_path.parent) as directory:
        partial = Path(directory) / "download.gz"
        command = [
            "curl", "--disable", "--fail", "--silent", "--show-error", "--location", "--proto", "=https",
            "--proto-redir", "=https", "--max-redirs", "5", "--connect-timeout", "15", "--max-time", "600",
            "--output", str(partial), "--", url,
        ]
        subprocess.run(command, check=True)
        with gzip.open(partial, "rb") as stream:
            columns = stream.readline().decode("utf-8-sig").strip().lower().split("\t")
            if not set(required_fields).issubset(columns) or not stream.readline().strip():
                raise ValueError(f"The download for {local_path.name} lacks the expected header or a data row")
            while stream.read(1024 * 1024):
                pass

        # If simultaneous downloads finish, keep the first complete copy that passed validation.
        try:
            cached_path.hardlink_to(partial)
        except FileExistsError:
            if not cached_path.is_file():
                raise

    return cached_path


def bh_adjust(p_values):
    """BH-adjust valid P values for one metric, preserving missing positions and the original index."""
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().astype(float)
    if valid.empty:
        return result

    # Multiply by the number of tests before dividing by rank, then restore adjusted values to the original rows.
    order = np.argsort(valid.to_numpy())
    ranked = valid.to_numpy()[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[valid.index[order]] = np.minimum(adjusted, 1.0)
    return result


def signflip_p(values, gbmspace_rng=None):
    """Return a two-sided sign-flip P value for the patient or plex mean. Return NaN without finite observations."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    count = len(values)
    if not count:
        missing = np.nan
        return missing

    # Only GBM-Space samples signs above 20 patients. Preserve the batch order, comparison threshold and add-one
    # correction.
    observed = abs(values.mean())
    if gbmspace_rng is not None and count > 20:
        extreme = 0
        for start in range(0, 100_000, 5_000):
            signs = gbmspace_rng.choice((-1.0, 1.0), size=(min(5_000, 100_000 - start), count))
            extreme += np.count_nonzero(np.abs((signs * values).mean(axis=1)) >= observed)
        probability = float((1 + extreme) / 100_001)
        return probability

    # Enumerate all sign patterns for the exact test. Chunking bounds memory. Use the published rounding tolerance.
    extreme = 0
    patterns = 2**count
    for start in range(0, patterns, 5_000):
        indices = np.arange(start, min(start + 5_000, patterns))
        signs = 1 - 2 * ((indices[:, None] >> np.arange(count)) & 1)
        null = np.abs((signs * values).mean(axis=1))
        extreme += np.count_nonzero(null >= observed - 1e-15)
    probability = float(extreme / patterns)
    return probability


def patient_metrics(values, gbmspace_rng=None):
    """Summarize the patient count, direction and tests for one metric. Exclude zeros from Wilcoxon and retain NaNs."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    nonzero = values[values != 0]
    wilcoxon_p = np.nan
    if len(nonzero) >= 2:
        method = "auto" if gbmspace_rng is not None and len(nonzero) > 25 else "exact"
        wilcoxon_p = float(wilcoxon(nonzero, alternative="two-sided", method=method).pvalue)

    metrics = {
        "n": len(values), "median": float(np.median(values)) if len(values) else np.nan,
        "mean": float(np.mean(values)) if len(values) else np.nan, "n_positive": int(np.sum(values > 0)),
        "signflip_p": signflip_p(values, gbmspace_rng), "wilcoxon_p": wilcoxon_p,
    }
    return metrics


def spot_spearman(first, second):
    """Return within-section Spearman rho, or NaN for fewer than 100 valid spots or a constant variable."""
    valid = np.isfinite(first) & np.isfinite(second)
    correlation = np.nan
    if valid.sum() >= 100 and np.unique(first[valid]).size >= 2 and np.unique(second[valid]).size >= 2:
        correlation = float(spearmanr(first[valid], second[valid]).statistic)
    return correlation


def rank_residual(values, controls, standardize_controls):
    """Fit ranks with an intercept across complete spots. Align residuals to input spots, with NaN below 100 spots."""
    rank_table = pd.DataFrame({"target": values})
    for i in range(controls.shape[1]):
        rank_table[f"control_{i}"] = controls[:, i]
    valid = np.isfinite(rank_table.to_numpy()).all(axis=1)
    residual = np.full(len(values), np.nan)
    if valid.sum() < 100:
        return residual

    # Rank each column with average ties and divide by the valid count plus one. Only GBM-Space standardizes covariates.
    ranked_table = rank_table.loc[valid].rank(method="average") / (valid.sum() + 1.0)
    outcome = ranked_table.pop("target").to_numpy(dtype=float)
    covariates = ranked_table.to_numpy(dtype=float)
    if standardize_controls:
        means = covariates.mean(axis=0)
        deviations = covariates.std(axis=0)
        keep = deviations > 1e-12
        covariates = (covariates[:, keep] - means[keep]) / deviations[keep]
    else:
        keep = np.nanstd(covariates, axis=0) > 1e-12
        covariates = covariates[:, keep]

    design = np.column_stack([np.ones(len(outcome)), covariates])
    coefficients, _, _, _ = np.linalg.lstsq(design, outcome, rcond=1e-10)
    residual[valid] = outcome - np.einsum("ij,j->i", design, coefficients)
    return residual


def write_spatial_tables(
    output_dir, section_ligand_table, patient_ligand_table, section_table, patient_table, summary_table,
    representative_table
):
    """Check sections, ligands, patients and representative spots against each other, then write and reread each CSV.
    No return value.
    """
    files = import_module("00a_project_paths")
    section_index_table = section_table.set_index("sample", verify_integrity=True)
    counts = section_ligand_table.groupby("sample")["ligand"].nunique()
    if set(counts.index) != set(section_index_table.index) or not counts.eq(len(ligands)).all():
        raise ValueError("At least one section lacks a complete ligand panel")
    if set(section_ligand_table["ligand"]) != set(ligands) or set(summary_table["ligand"]) != set(ligands):
        raise ValueError("Ligands in the section or summary table differ from the predefined panel")

    # Match each section's patient and spot count, then verify patient-ligand pairs against the patient tables.
    samples = section_ligand_table["sample"]
    for column in ["patient", "n_spots"]:
        if not section_ligand_table[column].eq(samples.map(section_index_table[column])).all():
            raise ValueError(f"Section ligand and spatial tables disagree on {column}")
    expected = set(zip(section_ligand_table["patient"], section_ligand_table["ligand"]))
    actual = set(zip(patient_ligand_table["patient"], patient_ligand_table["ligand"]))
    if actual != expected or set(patient_table["patient"]) != set(section_table["patient"]):
        raise ValueError("Patient or ligand membership differs between patient and section tables")

    # Retain all spots from the representative section. GBM-Space applies its eligibility mask only when plotting.
    samples = representative_table["sample"].unique()
    if len(samples) != 1 or len(representative_table) != section_index_table.at[samples[0], "n_spots"]:
        raise ValueError("The representative section's spot count differs from the section summary")
    coordinates = ["x", "y"] if "x" in representative_table else ["pixel_col", "pixel_row"]

    files.write_table(section_ligand_table, output_dir / "01_section_ligand_metrics.csv", ["sample", "ligand"])
    files.write_table(patient_ligand_table, output_dir / "02_patient_ligand_metrics.csv", ["patient", "ligand"])
    files.write_table(section_table, output_dir / "03_section_spatial_metrics.csv", ["sample"])
    files.write_table(patient_table, output_dir / "04_patient_spatial_metrics.csv", ["patient"])
    files.write_table(summary_table, output_dir / "05_ligand_summary.csv", ["ligand"])
    files.write_table(representative_table, output_dir / "06_representative_section_spots.csv", coordinates)
