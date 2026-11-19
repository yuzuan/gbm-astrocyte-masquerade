#!/usr/bin/env Rscript
# Purpose: Use Greenwald and Ravi spot labels to compare AC, other malignant and reactive astrocyte regions
# and estimate the contrast in neighboring myeloid receptor scores.
# Input under --input-dir: general/GBM_samples.txt, general/exp_mats_GBM/*counts.rds,
# MP/mp_assign_124/*.rds (SpotID, spot_type_meta_new) and
# general/GBM_data/SAMPLE/outs/spatial/tissue_positions_list.csv.
# Supply sample, patient, region and source in --metadata or general/sample_metadata.csv.
# Output: external_validation/07_greenwald_spatial_analysis/ contains 01_section_ligand_metrics.csv,
# 02_patient_ligand_metrics.csv, 03_section_spatial_metrics.csv and 04_patient_spatial_metrics.csv,
# 05_ligand_summary.csv, 06_representative_section_spots.csv, 07_analysis_summary.json,
# 08_figure_s6_greenwald_spatial_signaling.pdf and 08_figure_s6_greenwald_spatial_signaling.png.
# Usage: Rscript 07_greenwald_spatial_analysis.R --input-dir RAW_DIR --metadata MATCHING_METADATA.csv [--output-dir DIR]
# RStudio Source uses the default input directory. As the project's converter exception, public pdftools/Poppler
# renders the 600 dpi PNG from the PDF master.
# The local fastR 1.9.1 license is a template. No redistributable source for that version was verified.

library(jsonlite)
library(pdftools)

# RStudio Source records the file name. Rscript provides --file. Resolve default paths from the script's package.
script_file <- NULL
for (frame in sys.frames()) if (!is.null(frame$ofile)) script_file <- frame$ofile


if (is.null(script_file)) {
    file_argument <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    if (length(file_argument) != 1L) stop("Start this script with RStudio Source or Rscript")
    script_file <- sub("^--file=", "", file_argument)
    script_file <- gsub("~+~", " ", script_file, fixed = TRUE)
}
package_dir <- dirname(dirname(normalizePath(script_file, mustWork = TRUE)))
external_dir <- file.path(package_dir, "external_validation")

# Each command-line option takes one path. Report missing arguments, repeated options and unknown options.
arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) %% 2L != 0L) stop("Each option must be followed by a path")
options_by_name <- list()
if (length(arguments)) {
    for (i in seq.int(1L, length(arguments), by = 2L)) {
        option <- arguments[i]
        if (!option %in% c("--input-dir", "--metadata", "--output-dir")) stop("Unknown option ", option)
        if (option %in% names(options_by_name)) stop("Duplicate option ", option)
        options_by_name[[option]] <- path.expand(arguments[i + 1L])
    }
}
input_dir <- options_by_name[["--input-dir"]]
if (is.null(input_dir)) input_dir <- file.path(external_dir, "raw", "greenwald_inputs")
samples <- readLines(file.path(input_dir, "general", "GBM_samples.txt"), warn = FALSE)
samples <- samples[nzchar(samples)]
if (!length(samples) || anyDuplicated(samples)) stop("GBM_samples.txt is empty or contains duplicate sections")

# The public template requires matching input identifiers. Use just the four metadata fields from any local table.
metadata_path <- options_by_name[["--metadata"]]
if (is.null(metadata_path)) {
    metadata_path <- file.path(input_dir, "general", "sample_metadata.csv")
    if (!file.exists(metadata_path)) {
        metadata_path <- file.path(package_dir, "input_templates", "greenwald_sample_metadata.csv")
    }
}
metadata_table <- read.csv(metadata_path, colClasses = "character", check.names = FALSE)
required <- c("sample", "patient", "region", "source")
if (!all(required %in% colnames(metadata_table))) stop("Metadata must include sample, patient, region and source")
metadata_table <- metadata_table[, required, drop = FALSE]
if (anyNA(metadata_table[, c("sample", "patient", "source")])) stop("Missing sample, patient or source in metadata")
if (anyDuplicated(metadata_table$sample) || !setequal(samples, metadata_table$sample)) {
    stop("Section identifiers differ from the metadata. Supply --metadata with identifiers matching the raw files")
}
if (!all(metadata_table$source %in% c("Greenwald2024", "Ravi2022"))) stop("Source must be Greenwald2024 or Ravi2022")

# Read each section's gene-by-spot count matrix, author-assigned spot labels and headerless Visium coordinates.
ligands <- c("TGFB2", "TGFB1", "ANXA1", "SPP1", "CSF1", "CCL2")
receptors <- c("TGFBR1", "TGFBR2")
ac_markers <- c("GFAP", "S100B", "HOPX", "SLC1A3", "MLC1")
target_genes <- unique(c(ligands, receptors, ac_markers))
ligand_tables <- vector("list", length(samples))
spatial_tables <- vector("list", length(samples))
plot_tables <- list()
for (i in seq_along(samples)) {
    sample_id <- samples[i]
    counts_path <- file.path(input_dir, "general", "exp_mats_GBM", paste0(sample_id, "counts.rds"))
    count_matrix <- readRDS(counts_path)
    if (!all(target_genes %in% rownames(count_matrix))) stop(sample_id, " lacks a predefined target gene")
    annotation_path <- file.path(input_dir, "MP", "mp_assign_124", paste0(sample_id, ".rds"))
    annotation_table <- readRDS(annotation_path)
    annotation_table <- as.data.frame(annotation_table)
    if (!all(c("SpotID", "spot_type_meta_new") %in% colnames(annotation_table))) stop(
        sample_id, " lacks author-assigned spot labels"
    )

    # Name the headerless position columns in 10x order. Align all three sources by count-matrix barcode order
    # and retain labeled spots in tissue.
    position_path <- file.path(
        input_dir, "general", "GBM_data", sample_id, "outs", "spatial", "tissue_positions_list.csv"
    )
    position_table <- read.csv(
        position_path,
        header    = FALSE,
        col.names = c("barcode", "in_tissue", "array_row", "array_col", "pixel_row", "pixel_col"),
        stringsAsFactors = FALSE
    )
    common <- Reduce(intersect, list(colnames(count_matrix), annotation_table$SpotID, position_table$barcode))
    annotation_table <- annotation_table[match(common, annotation_table$SpotID), , drop = FALSE]
    position_table <- position_table[match(common, position_table$barcode), , drop = FALSE]
    keep <- position_table$in_tissue == 1L & !is.na(annotation_table$spot_type_meta_new)
    common <- common[keep]
    position_table <- position_table[keep, , drop = FALSE]
    labels <- as.character(annotation_table$spot_type_meta_new[keep])

    # Use each spot's total counts across genes. Densify only target genes for column-wise normalization and transpose.
    libraries <- colSums(count_matrix[, common, drop = FALSE])
    selected_matrix <- as.matrix(count_matrix[target_genes, common, drop = FALSE])
    normalized_matrix <- log1p(sweep(selected_matrix, 2L, pmax(libraries, 1), `/`) * 10000)
    expression_table <- as.data.frame(t(normalized_matrix))
    rm(count_matrix, selected_matrix, normalized_matrix)

    # Keep the authors' AC, other malignant, reactive astrocyte and myeloid labels. Nonmyeloid spots have receptor
    # signal zero.
    ac <- labels == "AC"
    other_malignant <- labels %in% c("OPC", "NPC", "MES", "MES.Hyp", "MES.Ast", "Prolif.Metab", "chromatin.reg")
    reactive <- labels == "Reactive.Ast"
    myeloid <- labels %in% c("Mac", "Inflammatory.Mac")
    ac_score <- rowMeans(expression_table[, ac_markers, drop = FALSE])
    receiver <- ifelse(myeloid, rowMeans(expression_table[, receptors, drop = FALSE]), 0)

    # Each column is one of six Visium neighbor offsets. Exclude the central spot and retain NA if no neighbor is valid.
    coordinates <- paste(position_table$array_row, position_table$array_col, sep = ":")
    if (anyDuplicated(coordinates)) stop(sample_id, " has duplicate Visium coordinates")
    lookup <- setNames(receiver, coordinates)
    offsets <- matrix(c(0, -2, 0, 2, -1, -1, -1, 1, 1, -1, 1, 1), ncol = 2L, byrow = TRUE)
    neighbor_matrix <- matrix(NA_real_, nrow = length(receiver), ncol = nrow(offsets))
    for (j in seq_len(nrow(offsets))) {
        neighbor_ids <- paste(
            position_table$array_row + offsets[j, 1], position_table$array_col + offsets[j, 2], sep = ":"
        )
        neighbor_matrix[, j] <- unname(lookup[neighbor_ids])
    }
    neighbor_receiver <- rowMeans(neighbor_matrix, na.rm = TRUE)
    neighbor_receiver[rowSums(is.finite(neighbor_matrix)) == 0L] <- NA_real_

    # Each contrast requires at least 20 spots per group. Compute differences in mean expression and detection
    # and the rank correlation for each ligand within each section.
    meta_table <- metadata_table[match(sample_id, metadata_table$sample), , drop = FALSE]
    enough_main <- sum(ac) >= 20L && sum(other_malignant) >= 20L
    enough_reactive <- sum(ac) >= 20L && sum(reactive) >= 20L
    ligand_rows <- vector("list", length(ligands))
    for (j in seq_along(ligands)) {
        ligand <- ligands[j]
        values <- expression_table[, ligand]
        other_mean <- NA_real_
        other_detection <- NA_real_
        if (enough_main) {
            other_mean <- mean(values[ac]) - mean(values[other_malignant])
            other_detection <- mean(values[ac] > 0) - mean(values[other_malignant] > 0)
        }
        reactive_mean <- if (enough_reactive) mean(values[ac]) - mean(values[reactive]) else NA_real_
        marker_rho <- NA_real_
        if (length(unique(values)) > 1) marker_rho <- suppressWarnings(cor(values, ac_score, method = "spearman"))

        ligand_rows[[j]] <- data.frame(
            dataset                  = "Greenwald-Ravi",
            source                   = meta_table$source,
            patient                  = meta_table$patient,
            sample                   = sample_id,
            region                   = meta_table$region,
            ligand                   = ligand,
            n_spots                  = nrow(expression_table),
            n_ac_spots               = sum(ac),
            n_other_malignant_spots  = sum(other_malignant),
            n_reactive_astro_spots   = sum(reactive),
            ac_minus_other_mean      = other_mean,
            ac_minus_other_detection = other_detection,
            ac_minus_reactive_mean   = reactive_mean,
            ac_marker_rho            = marker_rho,
            stringsAsFactors         = FALSE
        )
    }
    ligand_tables[[i]] <- do.call(rbind, ligand_rows)

    # Within AC-labeled spots, compare the upper half of positive TGFB2 with zero expression. Estimate a neighbor
    # contrast only if each group has at least ten source spots with neighbors.
    positive <- ac & expression_table$TGFB2 > 0 & is.finite(neighbor_receiver)
    zero <- ac & expression_table$TGFB2 == 0 & is.finite(neighbor_receiver)
    high <- rep(FALSE, nrow(expression_table))
    if (sum(positive) >= 10L) high <- positive & expression_table$TGFB2 >= median(expression_table$TGFB2[positive])
    delta <- NA_real_
    if (sum(high) >= 10L && sum(zero) >= 10L) delta <- mean(neighbor_receiver[high]) - mean(neighbor_receiver[zero])

    spatial_tables[[i]] <- data.frame(
        dataset          = "Greenwald-Ravi",
        source           = meta_table$source,
        patient          = meta_table$patient,
        sample           = sample_id,
        region           = meta_table$region,
        n_spots          = nrow(expression_table),
        n_ac_spots       = sum(ac),
        n_myeloid_spots  = sum(myeloid),
        n_tgfb2_high_ac  = sum(high),
        n_tgfb2_zero_ac  = sum(zero),
        tgfb2_neighbor_receiver_delta = delta,
        stringsAsFactors = FALSE
    )
    plot_tables[[sample_id]] <- data.frame(
        pixel_col = position_table$pixel_col,
        pixel_row = position_table$pixel_row,
        label     = labels,
        tgfb2     = expression_table$TGFB2,
        receiver  = receiver,
        sample    = sample_id,
        stringsAsFactors = FALSE
    )
}
sample_ligand_table_total <- do.call(rbind, ligand_tables)
sample_spatial_table_total <- do.call(rbind, spatial_tables)

# Return the median of one patient's finite section values for a metric, or NA if no section qualifies.
median_finite <- function(values) {
    result <- median(values[is.finite(values)])
    return(result)
}
patient_ligand_table <- aggregate(
    cbind(ac_minus_other_mean, ac_minus_other_detection, ac_minus_reactive_mean, ac_marker_rho) ~ patient + ligand,
    data = sample_ligand_table_total,
    FUN  = median_finite,
    na.action = na.pass
)
patient_spatial_table <- aggregate(
    tgfb2_neighbor_receiver_delta ~ patient,
    data = sample_spatial_table_total,
    FUN  = median_finite,
    na.action = na.pass
)

# Compute the exact two-sided sign-flip P value for the mean patient difference. Enumerate signs in chunks,
# retain the original tolerance and return NA for empty input.
proximity_values <- patient_spatial_table$tgfb2_neighbor_receiver_delta
signflip_p <- function(values) {
    values <- values[is.finite(values)]
    count <- length(values)
    result <- NA_real_
    if (!count) return(result)
    observed <- abs(mean(values))
    pattern_count <- 2^count
    extreme <- 0

    # Each bit represents one patient. Use integer division to preserve the original mapping from 0 to -1 and 1 to +1.
    for (start in seq(0, pattern_count - 1, by = 10000L)) {
        patterns <- seq(start, min(start + 9999L, pattern_count - 1))
        signs <- matrix(-1, nrow = length(patterns), ncol = count)
        for (j in seq_len(count)) signs[, j] <- 2 * (floor(patterns / 2^(j - 1L)) %% 2) - 1
        null_values <- abs(rowMeans(sweep(signs, 2L, values, `*`)))
        extreme <- extreme + sum(null_values >= observed - 1e-15)
    }
    result <- extreme / pattern_count
    return(result)
}
proximity_signflip <- signflip_p(proximity_values)

# Drop zero differences and require two finite nonzero values. Retain R's asymptotic Wilcoxon test and continuity
# correction.
wilcoxon_p <- function(values) {
    values <- values[is.finite(values) & values != 0]
    result <- NA_real_
    if (length(values) >= 2L) result <- suppressWarnings(wilcox.test(values, mu = 0, exact = FALSE)$p.value)
    return(result)
}
proximity_wilcoxon <- wilcoxon_p(proximity_values)

# Summarize AC versus other malignant and AC versus reactive astrocyte differences separately.
# Within each contrast, apply BH adjustment across the six ligands.
summary_rows <- vector("list", length(ligands))
for (i in seq_along(ligands)) {
    group_table <- patient_ligand_table[patient_ligand_table$ligand == ligands[i], , drop = FALSE]
    values <- group_table$ac_minus_other_mean
    reactive <- group_table$ac_minus_reactive_mean
    summary_rows[[i]] <- data.frame(
        ligand                       = ligands[i],
        n_patients                   = sum(is.finite(values)),
        ac_minus_other_median        = median_finite(values),
        ac_minus_other_mean          = mean(values[is.finite(values)]),
        ac_minus_other_n_positive    = sum(values > 0, na.rm = TRUE),
        ac_minus_other_signflip_p    = signflip_p(values),
        ac_minus_other_wilcoxon_p    = wilcoxon_p(values),
        ac_minus_reactive_n          = sum(is.finite(reactive)),
        ac_minus_reactive_median     = median_finite(reactive),
        ac_minus_reactive_n_positive = sum(reactive > 0, na.rm = TRUE),
        ac_minus_reactive_signflip_p = signflip_p(reactive),
        stringsAsFactors             = FALSE
    )
}
summary_table_total <- do.call(rbind, summary_rows)
summary_table_total$ac_minus_other_signflip_q <- p.adjust(summary_table_total$ac_minus_other_signflip_p, method = "BH")
summary_table_total$ac_minus_reactive_signflip_q <- p.adjust(
    summary_table_total$ac_minus_reactive_signflip_p, method = "BH"
)

# Select the section whose primary TGFB2 contrast lies closest to the section median. Break ties in input-list order.
keep <- sample_ligand_table_total$ligand == "TGFB2" & is.finite(sample_ligand_table_total$ac_minus_other_mean)
tgfb2_sample_table <- sample_ligand_table_total[keep, , drop = FALSE]
if (!nrow(tgfb2_sample_table)) stop("No section has an estimable TGFB2 effect for the representative plot")
distance <- abs(tgfb2_sample_table$ac_minus_other_mean - median(tgfb2_sample_table$ac_minus_other_mean))
representative_sample <- tgfb2_sample_table$sample[which.min(distance)]
representative_table <- plot_tables[[representative_sample]]

# Check section-ligand and patient-ligand pairs and representative spot counts. Count unique patients per source.
patient_sources_table <- unique(metadata_table[, c("patient", "source")])
if (anyDuplicated(patient_sources_table$patient)) stop("Metadata assigns a patient to two sources")
if (!setequal(patient_spatial_table$patient, patient_sources_table$patient)) stop("Patients differ from metadata")
if (!all(table(sample_ligand_table_total$sample, sample_ligand_table_total$ligand) == 1L)) stop(
    "Missing or repeated section-ligand pair"
)
if (!all(table(patient_ligand_table$patient, patient_ligand_table$ligand) == 1L)) stop(
    "Missing or repeated patient-ligand pair"
)
if (!setequal(sample_spatial_table_total$sample, samples)) stop("Sections missing from the spatial table")
if (!setequal(summary_table_total$ligand, ligands)) stop("Ligands missing from the summary")
spot_count <- sample_spatial_table_total$n_spots[match(representative_sample, sample_spatial_table_total$sample)]
if (nrow(representative_table) != spot_count) stop("Representative spot count differs from the spatial table")

# Stage the six CSVs in the output directory. Verify column order, rows, text and values before replacing each file.
output_dir <- options_by_name[["--output-dir"]]
if (is.null(output_dir)) output_dir <- file.path(external_dir, "07_greenwald_spatial_analysis")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_tables <- list(
    "01_section_ligand_metrics.csv" = sample_ligand_table_total,
    "02_patient_ligand_metrics.csv" = patient_ligand_table,
    "03_section_spatial_metrics.csv" = sample_spatial_table_total,
    "04_patient_spatial_metrics.csv" = patient_spatial_table,
    "05_ligand_summary.csv"   = summary_table_total,
    "06_representative_section_spots.csv" = representative_table
)
for (file_name in names(output_tables)) {
    result_table <- output_tables[[file_name]]
    csv_path <- tempfile(tmpdir = output_dir, fileext = ".csv")
    write.csv(result_table, csv_path, row.names = FALSE, na = "NA")
    classes <- ifelse(vapply(result_table, is.character, logical(1)), "character", "numeric")
    saved_table <- read.csv(csv_path, colClasses = classes, check.names = FALSE, na.strings = "NA")
    if (!identical(colnames(saved_table), colnames(result_table)) || nrow(saved_table) != nrow(result_table)) {
        stop(file_name, " row count or column order changed on readback")
    }
    for (column in colnames(result_table)) {
        comparison <- all.equal(
            result_table[, column], saved_table[, column], tolerance = 1e-14, check.attributes = FALSE
        )
        if (!isTRUE(comparison)) stop(file_name, " differs on readback in column ", column)
    }
    if (!file.rename(csv_path, file.path(output_dir, file_name))) stop("Could not save ", file_name)
}

# Count patients by source from the input sample-patient-source records. The published JSON uses four decimal
# places. The CSVs retain their own precision.
tgfb2_summary_table <- summary_table_total[summary_table_total$ligand == "TGFB2", , drop = FALSE]
proximity_summary <- list(
    n_patients = sum(is.finite(proximity_values)),
    median     = median_finite(proximity_values),
    mean       = mean(proximity_values[is.finite(proximity_values)]),
    n_positive = sum(proximity_values > 0, na.rm = TRUE),
    signflip_p = proximity_signflip,
    wilcoxon_p = proximity_wilcoxon
)
analysis_summary <- list(
    dataset    = "Greenwald-Ravi",
    n_samples  = length(samples),
    n_patients = nrow(patient_sources_table),
    n_greenwald_patients = sum(patient_sources_table$source == "Greenwald2024"),
    n_ravi_patients = sum(patient_sources_table$source == "Ravi2022"),
    tgfb2      = as.list(tgfb2_summary_table),
    proximity  = proximity_summary
)
json_path <- tempfile(tmpdir = output_dir, fileext = ".json")
write_json(analysis_summary, json_path, pretty = TRUE, auto_unbox = TRUE, na = "null", digits = 4)
saved_summary <- read_json(json_path, simplifyVector = TRUE)
if (!identical(names(saved_summary), names(analysis_summary))) stop("Summary fields changed on readback")
if (saved_summary$n_greenwald_patients + saved_summary$n_ravi_patients != saved_summary$n_patients) stop(
    "Patient counts by source do not sum to the total"
)
if (saved_summary$n_samples != nrow(sample_spatial_table_total)) stop("Summary section count differs")
if (saved_summary$tgfb2$ligand != "TGFB2") stop("Unexpected ligand in summary")
if (!file.rename(json_path, file.path(output_dir, "07_analysis_summary.json"))) stop("Could not save JSON summary")

# The original workflow drew PNG before PDF. Skip the uniforms used by the first plot to reproduce the PDF jitter.
tgfb2_patient_table <- patient_ligand_table[patient_ligand_table$ligand == "TGFB2", , drop = FALSE]
effect_sets <- list(
    tgfb2_patient_table$ac_minus_other_mean, patient_ligand_table$ac_minus_other_mean,
    tgfb2_patient_table$ac_minus_reactive_mean, proximity_values
)
jitter_count <- 0L
for (values in effect_sets) jitter_count <- jitter_count + sum(is.finite(values))
set.seed(20260716)
invisible(runif(jitter_count))

# Record a full base R plot on a separate device. Close it on exit and return the plot for replay on the PDF device.
record_figure <- function() {
    pdf(NULL, width = 12, height = 8, useDingbats = FALSE)
    on.exit(dev.off())
    dev.control(displaylist = "enable")
    par(mfrow = c(2, 4), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

    # Draw panels a, b, c and d in order. Jitter only x coordinates. Statistics use the original values.
    titles <- c(
        "a  TGFB2 in AC spots", "b  Fixed ligand panel", "c  Malignant AC vs reactive astrocyte",
        "d  Neighboring myeloid receptor score"
    )
    axis_labels <- c(
        "AC minus other malignant", "AC minus other malignant", "TGFB2 expression difference",
        "High-minus-zero TGFB2 AC spots"
    )
    for (i in seq_along(effect_sets)) {
        values <- effect_sets[[i]]
        values <- values[is.finite(values)]
        if (i == 2L) {
            plot(
                NA,
                xlim = c(0.5, length(ligands) + 0.5),
                ylim = range(c(values, 0)),
                xaxt = "n",
                xlab = "",
                ylab = axis_labels[i],
                main = titles[i],
                bty  = "l"
            )
            axis(1, at = seq_along(ligands), labels = ligands, las = 2, cex.axis = 0.75)
            abline(h = 0, lty = 2, col = "#777777")
            for (j in seq_along(ligands)) {
                values <- patient_ligand_table$ac_minus_other_mean[patient_ligand_table$ligand == ligands[j]]
                values <- values[is.finite(values)]
                positions <- jitter(rep(j, length(values)), amount = 0.08)
                points(positions, values, pch = 16, cex = 0.7, col = adjustcolor("#6C7A89", alpha.f = 0.65))
                color <- ifelse(ligands[j] == "TGFB2", "#C0392B", "#222222")
                segments(j - 0.22, median_finite(values), j + 0.22, median_finite(values), lwd = 2.5, col = color)
            }
            next
        }
        limits <- if (length(values)) range(values) else c(-1, 1)
        positions <- jitter(rep(1, length(values)), amount = 0.08)
        plot(
            positions, values,
            pch  = 16,
            col  = adjustcolor("#167D8D", alpha.f = 0.8),
            xaxt = "n",
            xlab = "",
            ylab = axis_labels[i],
            main = titles[i],
            xlim = c(0.65, 1.35),
            ylim = limits,
            bty  = "l"
        )
        abline(h = 0, lty = 2, col = "#777777")
        if (length(values)) segments(0.82, median(values), 1.18, median(values), lwd = 3, col = "#C0392B")
    }

    # Color each spatial metric between its 1st and 99th percentiles. For constant values, expand the bin range by 0.5.
    map_values <- list(
        as.numeric(representative_table$label == "AC"), representative_table$tgfb2, representative_table$receiver
    )
    palettes <- list(c("#EEEEEE", "#167D8D"), c("#F3E5AB", "#D95F02", "#7F0000"), c("#E8F1F2", "#2C7FB8", "#253494"))
    map_titles <- c("e  Author-assigned AC spots", "f  TGFB2 expression", "g  Myeloid x TGFBR score")
    for (i in seq_along(map_values)) {
        values <- map_values[[i]]
        finite <- is.finite(values)
        colors <- rep("#D9D9D9", length(values))
        if (any(finite)) {
            limits <- quantile(values[finite], c(0.01, 0.99), names = FALSE)
            if (limits[1] == limits[2]) limits <- range(values[finite])
            if (limits[1] == limits[2]) limits <- limits + c(-0.5, 0.5)
            breaks <- seq(limits[1], limits[2], length.out = 101L)
            bins <- cut(pmin(pmax(values[finite], limits[1]), limits[2]), breaks, include.lowest = TRUE, labels = FALSE)
            palette <- colorRampPalette(palettes[[i]])
            colors[finite] <- palette(100)[bins]
        }
        plot(
            representative_table$pixel_col, -representative_table$pixel_row,
            pch  = 16,
            cex  = 0.28,
            col  = colors,
            asp  = 1,
            axes = FALSE,
            xlab = "",
            ylab = "",
            main = map_titles[i]
        )
    }
    plot.new()
    note <- paste0(
        "Representative section\n", representative_sample, "\n\nClosest to the median TGFB2 effect.",
        "\n\nInference uses ", analysis_summary$n_patients, " patient medians."
    )
    text(0, 1, note, adj = c(0, 1), cex = 0.9)
    mtext("Greenwald-Ravi spatial replication of AC-like TGFB2", outer = TRUE, side = 3, adj = 0, cex = 1.25, font = 2)
    result <- recordPlot()
    return(result)
}
figure <- record_figure()

# Require one PDF page with the spatial panel. Reread the derived PNG header to check its size at 600 dpi.
pdf_path <- tempfile(tmpdir = output_dir, fileext = ".pdf")
pdf(pdf_path, width = 12, height = 8, useDingbats = FALSE)
    print(figure)
invisible(dev.off())
page_info <- pdf_info(pdf_path)
if (page_info$pages != 1L || !grepl("TGFB2 expression", pdf_text(pdf_path), fixed = TRUE)) stop(
    "Check the PDF page count and TGFB2 spatial panel"
)
png_path <- sub("\\.pdf$", ".png", pdf_path)
invisible(pdf_convert(pdf_path, format = "png", pages = 1L, filenames = png_path, dpi = 600, verbose = FALSE))
png_header <- readBin(png_path, "raw", n = 24L)
if (length(png_header) != 24L) stop("Incomplete PNG header")
if (!identical(png_header[1:8], as.raw(c(137, 80, 78, 71, 13, 10, 26, 10)))) stop("Invalid PNG signature")
pixels <- readBin(png_header[17:24], "integer", n = 2L, size = 4L, endian = "big")
page_table <- pdf_pagesize(pdf_path)
expected_pixels <- round(c(page_table$width, page_table$height) * 600 / 72)
if (!all(pixels == expected_pixels)) stop("PNG pixel dimensions differ from the PDF page size at 600 dpi")
if (!file.rename(pdf_path, file.path(output_dir, "08_figure_s6_greenwald_spatial_signaling.pdf"))) stop(
    "Could not save the Greenwald PDF"
)
if (!file.rename(png_path, file.path(output_dir, "08_figure_s6_greenwald_spatial_signaling.png"))) stop(
    "Could not save the Greenwald PNG"
)
cat("Sections: ", nrow(sample_spatial_table_total), ", patients: ", nrow(patient_spatial_table), "\n", sep = "")
