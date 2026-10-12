#!/usr/bin/env Rscript

args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_all, value = TRUE)
script_file <- sub("^--file=", "", file_arg[[1]])
root <- normalizePath(file.path(dirname(script_file), ".."), mustWork = TRUE)
input_dir <- file.path(root, "raw", "greenwald_inputs")
result_dir <- file.path(root, "results", "greenwald")
figure_dir <- file.path(root, "figures")
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

ligands <- c("TGFB2", "TGFB1", "ANXA1", "SPP1", "CSF1", "CCL2")
receptors <- c("TGFBR1", "TGFBR2")
ac_markers <- c("GFAP", "S100B", "HOPX", "SLC1A3", "MLC1")
target_genes <- unique(c(ligands, receptors, ac_markers))
malignant_labels <- c(
  "AC", "OPC", "NPC", "MES", "MES.Hyp", "MES.Ast",
  "Prolif.Metab", "chromatin.reg"
)
myeloid_labels <- c("Mac", "Inflammatory.Mac")
min_group_spots <- 20L
min_source_spots <- 10L
set.seed(20260716)

samples <- readLines(file.path(input_dir, "general", "GBM_samples.txt"))
samples <- samples[nzchar(samples)]

sample_meta <- read.csv(file.path(input_dir, "general", "sample_metadata.csv"), stringsAsFactors = FALSE)
required <- c("sample", "patient", "region", "source")
if (!all(required %in% names(sample_meta))) stop("sample_metadata.csv requires sample, patient, region, source")
if (!setequal(samples, sample_meta$sample)) {
  stop("Sample metadata does not exactly match GBM_samples.txt")
}

median_na <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(NA_real_)
  median(x)
}

exact_signflip_p <- function(values) {
  values <- values[is.finite(values)]
  n <- length(values)
  if (n == 0) return(NA_real_)
  observed <- abs(mean(values))
  n_patterns <- 2^n
  count_extreme <- 0
  chunk_size <- 10000L
  bit_positions <- 0:(n - 1L)
  for (start in seq.int(0, n_patterns - 1L, by = chunk_size)) {
    stop_at <- min(start + chunk_size - 1L, n_patterns - 1L)
    patterns <- start:stop_at
    bits <- outer(
      patterns,
      bit_positions,
      FUN = function(pattern, bit) {
        bitwAnd(pattern, bitwShiftL(1L, bit)) != 0L
      }
    )
    signs <- ifelse(bits, 1, -1)
    null_values <- abs(rowMeans(sweep(signs, 2, values, `*`)))
    count_extreme <- count_extreme + sum(null_values >= observed - 1e-15)
  }
  count_extreme / n_patterns
}

safe_wilcoxon <- function(values) {
  values <- values[is.finite(values) & values != 0]
  if (length(values) < 2) return(NA_real_)
  suppressWarnings(wilcox.test(values, mu = 0, exact = FALSE)$p.value)
}

neighbor_mean <- function(values, array_row, array_col) {
  keys <- paste(array_row, array_col, sep = ":")
  lookup <- setNames(values, keys)
  offsets <- matrix(
    c(0, -2, 0, 2, -1, -1, -1, 1, 1, -1, 1, 1),
    ncol = 2,
    byrow = TRUE
  )
  neighbor_values <- sapply(seq_len(nrow(offsets)), function(i) {
    neighbor_keys <- paste(
      array_row + offsets[i, 1],
      array_col + offsets[i, 2],
      sep = ":"
    )
    unname(lookup[neighbor_keys])
  })
  n_present <- rowSums(is.finite(neighbor_values))
  result <- rowMeans(neighbor_values, na.rm = TRUE)
  result[n_present == 0] <- NA_real_
  result
}

read_sample <- function(sample, include_plot = FALSE) {
  counts_file <- file.path(
    input_dir, "general", "exp_mats_GBM", paste0(sample, "counts.rds")
  )
  annotation_file <- file.path(
    input_dir, "MP", "mp_assign_124", paste0(sample, ".rds")
  )
  position_file <- file.path(
    input_dir, "general", "GBM_data", sample, "outs", "spatial",
    "tissue_positions_list.csv"
  )
  if (!all(file.exists(c(counts_file, annotation_file, position_file)))) {
    stop("Missing input for ", sample)
  }

  counts <- readRDS(counts_file)
  if (!all(target_genes %in% rownames(counts))) {
    missing <- setdiff(target_genes, rownames(counts))
    stop(sample, ": missing genes: ", paste(missing, collapse = ", "))
  }
  annotation <- readRDS(annotation_file)
  positions <- read.csv(
    position_file,
    header = FALSE,
    col.names = c(
      "barcode", "in_tissue", "array_row", "array_col",
      "pixel_row", "pixel_col"
    ),
    stringsAsFactors = FALSE
  )

  common <- Reduce(
    intersect,
    list(colnames(counts), annotation$SpotID, positions$barcode)
  )
  annotation <- annotation[match(common, annotation$SpotID), ]
  positions <- positions[match(common, positions$barcode), ]
  keep <- positions$in_tissue == 1L & !is.na(annotation$spot_type_meta_new)
  common <- common[keep]
  annotation <- annotation[keep, ]
  positions <- positions[keep, ]

  library_size <- colSums(counts[, common, drop = FALSE])
  selected <- counts[target_genes, common, drop = FALSE]
  normalized <- log1p(sweep(selected, 2, pmax(library_size, 1), `/`) * 10000)
  expression <- as.data.frame(t(normalized), stringsAsFactors = FALSE)
  expression$barcode <- common
  expression$label <- as.character(annotation$spot_type_meta_new)
  expression$array_row <- positions$array_row
  expression$array_col <- positions$array_col
  expression$pixel_row <- positions$pixel_row
  expression$pixel_col <- positions$pixel_col

  ac <- expression$label == "AC"
  other_malignant <- expression$label %in% setdiff(malignant_labels, "AC")
  reactive_astro <- expression$label == "Reactive.Ast"
  myeloid <- expression$label %in% myeloid_labels
  ac_marker_score <- rowMeans(expression[, ac_markers, drop = FALSE])
  receiver <- ifelse(
    myeloid,
    rowMeans(expression[, receptors, drop = FALSE]),
    0
  )
  receiver_neighbor <- neighbor_mean(
    receiver, expression$array_row, expression$array_col
  )

  meta <- sample_meta[match(sample, sample_meta$sample), ]
  ligand_rows <- lapply(ligands, function(ligand) {
    values <- expression[[ligand]]
    enough_main <- sum(ac) >= min_group_spots &&
      sum(other_malignant) >= min_group_spots
    enough_reactive <- sum(ac) >= min_group_spots &&
      sum(reactive_astro) >= min_group_spots
    data.frame(
      dataset = "Greenwald-Ravi",
      source = meta$source,
      patient = meta$patient,
      sample = sample,
      region = meta$region,
      ligand = ligand,
      n_spots = nrow(expression),
      n_ac_spots = sum(ac),
      n_other_malignant_spots = sum(other_malignant),
      n_reactive_astro_spots = sum(reactive_astro),
      ac_minus_other_mean = if (enough_main) {
        mean(values[ac]) - mean(values[other_malignant])
      } else {
        NA_real_
      },
      ac_minus_other_detection = if (enough_main) {
        mean(values[ac] > 0) - mean(values[other_malignant] > 0)
      } else {
        NA_real_
      },
      ac_minus_reactive_mean = if (enough_reactive) {
        mean(values[ac]) - mean(values[reactive_astro])
      } else {
        NA_real_
      },
      ac_marker_rho = if (length(unique(values)) > 1) {
        suppressWarnings(cor(values, ac_marker_score, method = "spearman"))
      } else {
        NA_real_
      },
      stringsAsFactors = FALSE
    )
  })

  tgfb2 <- expression$TGFB2
  positive_ac <- ac & tgfb2 > 0 & is.finite(receiver_neighbor)
  zero_ac <- ac & tgfb2 == 0 & is.finite(receiver_neighbor)
  high_ac <- rep(FALSE, nrow(expression))
  if (sum(positive_ac) >= min_source_spots) {
    high_cut <- median(tgfb2[positive_ac])
    high_ac <- positive_ac & tgfb2 >= high_cut
  }
  enough_proximity <- sum(high_ac) >= min_source_spots &&
    sum(zero_ac) >= min_source_spots
  spatial_row <- data.frame(
    dataset = "Greenwald-Ravi",
    source = meta$source,
    patient = meta$patient,
    sample = sample,
    region = meta$region,
    n_spots = nrow(expression),
    n_ac_spots = sum(ac),
    n_myeloid_spots = sum(myeloid),
    n_tgfb2_high_ac = sum(high_ac),
    n_tgfb2_zero_ac = sum(zero_ac),
    tgfb2_neighbor_receiver_delta = if (enough_proximity) {
      mean(receiver_neighbor[high_ac]) - mean(receiver_neighbor[zero_ac])
    } else {
      NA_real_
    },
    stringsAsFactors = FALSE
  )

  plot_data <- NULL
  if (include_plot) {
    plot_data <- data.frame(
      pixel_col = expression$pixel_col,
      pixel_row = expression$pixel_row,
      label = expression$label,
      tgfb2 = expression$TGFB2,
      receiver = receiver,
      stringsAsFactors = FALSE
    )
  }
  rm(counts, selected, normalized)
  gc(verbose = FALSE)
  list(
    ligand = do.call(rbind, ligand_rows),
    spatial = spatial_row,
    plot = plot_data
  )
}

ligand_results <- list()
spatial_results <- list()
for (i in seq_along(samples)) {
  cat(sprintf("[%02d/%02d] %s\n", i, length(samples), samples[[i]]))
  result <- read_sample(samples[[i]], include_plot = FALSE)
  ligand_results[[i]] <- result$ligand
  spatial_results[[i]] <- result$spatial
}
sample_ligand <- do.call(rbind, ligand_results)
sample_spatial <- do.call(rbind, spatial_results)

patient_ligand <- aggregate(
  cbind(
    ac_minus_other_mean,
    ac_minus_other_detection,
    ac_minus_reactive_mean,
    ac_marker_rho
  ) ~ patient + ligand,
  data = sample_ligand,
  FUN = median_na,
  na.action = na.pass
)
patient_spatial <- aggregate(
  tgfb2_neighbor_receiver_delta ~ patient,
  data = sample_spatial,
  FUN = median_na,
  na.action = na.pass
)

summary_rows <- lapply(ligands, function(ligand) {
  group <- patient_ligand[patient_ligand$ligand == ligand, ]
  values <- group$ac_minus_other_mean
  reactive <- group$ac_minus_reactive_mean
  data.frame(
    ligand = ligand,
    n_patients = sum(is.finite(values)),
    ac_minus_other_median = median_na(values),
    ac_minus_other_mean = mean(values[is.finite(values)]),
    ac_minus_other_n_positive = sum(values > 0, na.rm = TRUE),
    ac_minus_other_signflip_p = exact_signflip_p(values),
    ac_minus_other_wilcoxon_p = safe_wilcoxon(values),
    ac_minus_reactive_n = sum(is.finite(reactive)),
    ac_minus_reactive_median = median_na(reactive),
    ac_minus_reactive_n_positive = sum(reactive > 0, na.rm = TRUE),
    ac_minus_reactive_signflip_p = exact_signflip_p(reactive),
    stringsAsFactors = FALSE
  )
})
ligand_summary <- do.call(rbind, summary_rows)
ligand_summary$ac_minus_other_signflip_q <- p.adjust(
  ligand_summary$ac_minus_other_signflip_p, method = "BH"
)
ligand_summary$ac_minus_reactive_signflip_q <- p.adjust(
  ligand_summary$ac_minus_reactive_signflip_p, method = "BH"
)

proximity_values <- patient_spatial$tgfb2_neighbor_receiver_delta
proximity_summary <- list(
  n_patients = sum(is.finite(proximity_values)),
  median = median_na(proximity_values),
  mean = mean(proximity_values[is.finite(proximity_values)]),
  n_positive = sum(proximity_values > 0, na.rm = TRUE),
  signflip_p = exact_signflip_p(proximity_values),
  wilcoxon_p = safe_wilcoxon(proximity_values)
)

write.csv(
  sample_ligand,
  file.path(result_dir, "sample_ligand_metrics.csv"),
  row.names = FALSE
)
write.csv(
  patient_ligand,
  file.path(result_dir, "patient_ligand_metrics.csv"),
  row.names = FALSE
)
write.csv(
  sample_spatial,
  file.path(result_dir, "sample_spatial_metrics.csv"),
  row.names = FALSE
)
write.csv(
  patient_spatial,
  file.path(result_dir, "patient_spatial_metrics.csv"),
  row.names = FALSE
)
write.csv(
  ligand_summary,
  file.path(result_dir, "ligand_summary.csv"),
  row.names = FALSE
)

tgfb2_summary <- ligand_summary[ligand_summary$ligand == "TGFB2", ]
analysis_summary <- list(
  dataset = "Greenwald-Ravi",
  n_samples = length(samples),
  n_patients = length(unique(sample_meta$patient)),
  n_greenwald_patients = length(unique(own_meta$patient)),
  n_ravi_patients = length(unique(ravi_meta$patient)),
  tgfb2 = as.list(tgfb2_summary),
  proximity = proximity_summary
)
jsonlite::write_json(
  analysis_summary,
  file.path(result_dir, "analysis_summary.json"),
  pretty = TRUE,
  auto_unbox = TRUE,
  na = "null"
)

tgfb2_samples <- sample_ligand[
  sample_ligand$ligand == "TGFB2" &
    is.finite(sample_ligand$ac_minus_other_mean),
]
sample_median <- median(tgfb2_samples$ac_minus_other_mean)
representative_sample <- tgfb2_samples$sample[
  which.min(abs(tgfb2_samples$ac_minus_other_mean - sample_median))
]
representative <- read_sample(representative_sample, include_plot = TRUE)$plot
representative$sample <- representative_sample
write.csv(
  representative,
  file.path(result_dir, "representative_plot.csv"),
  row.names = FALSE
)

strip_panel <- function(values, title, ylab) {
  values <- values[is.finite(values)]
  plot(
    jitter(rep(1, length(values)), amount = 0.08),
    values,
    pch = 16,
    col = grDevices::adjustcolor("#167D8D", alpha.f = 0.8),
    xaxt = "n",
    xlab = "",
    ylab = ylab,
    main = title,
    xlim = c(0.65, 1.35),
    bty = "l"
  )
  abline(h = 0, lty = 2, col = "#777777")
  if (length(values)) {
    segments(0.82, median(values), 1.18, median(values), lwd = 3, col = "#C0392B")
  }
}

ligand_panel <- function(frame) {
  plot(
    NA,
    xlim = c(0.5, length(ligands) + 0.5),
    ylim = range(c(frame$ac_minus_other_mean, 0), na.rm = TRUE),
    xaxt = "n",
    xlab = "",
    ylab = "AC minus other malignant",
    main = "b  Fixed ligand panel",
    bty = "l"
  )
  axis(1, at = seq_along(ligands), labels = ligands, las = 2, cex.axis = 0.75)
  abline(h = 0, lty = 2, col = "#777777")
  for (i in seq_along(ligands)) {
    values <- frame$ac_minus_other_mean[frame$ligand == ligands[[i]]]
    values <- values[is.finite(values)]
    points(
      jitter(rep(i, length(values)), amount = 0.08),
      values,
      pch = 16,
      cex = 0.7,
      col = grDevices::adjustcolor("#6C7A89", alpha.f = 0.65)
    )
    if (length(values)) {
      segments(
        i - 0.22, median(values), i + 0.22, median(values),
        lwd = 2.5,
        col = if (ligands[[i]] == "TGFB2") "#C0392B" else "#222222"
      )
    }
  }
}

map_panel <- function(x, y, values, title, palette) {
  finite <- is.finite(values)
  colors <- rep("#D9D9D9", length(values))
  if (sum(finite) > 0) {
    limits <- quantile(values[finite], c(0.01, 0.99), na.rm = TRUE)
    if (limits[[1]] == limits[[2]]) limits <- range(values[finite])
    if (limits[[1]] == limits[[2]]) limits <- limits + c(-0.5, 0.5)
    bins <- cut(
      pmin(pmax(values[finite], limits[[1]]), limits[[2]]),
      breaks = seq(limits[[1]], limits[[2]], length.out = 101),
      include.lowest = TRUE,
      labels = FALSE
    )
    colors[finite] <- palette(100)[bins]
  }
  plot(
    x,
    -y,
    pch = 16,
    cex = 0.28,
    col = colors,
    asp = 1,
    axes = FALSE,
    xlab = "",
    ylab = "",
    main = title
  )
}

draw_figure <- function(device) {
  if (device == "png") {
    png(
      file.path(figure_dir, "SupplementaryFigureS6_Greenwald_spatial_validation.png"),
      width = 12,
      height = 8,
      units = "in",
      res = 600,
      bg = "white"
    )
  } else {
    pdf(
      file.path(figure_dir, "SupplementaryFigureS6_Greenwald_spatial_validation.pdf"),
      width = 12,
      height = 8,
      useDingbats = FALSE
    )
  }
  par(mfrow = c(2, 4), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))
  tgfb2_patient <- patient_ligand[patient_ligand$ligand == "TGFB2", ]
  strip_panel(
    tgfb2_patient$ac_minus_other_mean,
    "a  TGFB2 in AC spots",
    "AC minus other malignant"
  )
  ligand_panel(patient_ligand)
  strip_panel(
    tgfb2_patient$ac_minus_reactive_mean,
    "c  Malignant AC vs reactive astrocyte",
    "TGFB2 expression difference"
  )
  strip_panel(
    patient_spatial$tgfb2_neighbor_receiver_delta,
    "d  Neighboring myeloid receptor score",
    "High-minus-zero TGFB2 AC spots"
  )
  map_panel(
    representative$pixel_col,
    representative$pixel_row,
    as.numeric(representative$label == "AC"),
    "e  Author-assigned AC spots",
    colorRampPalette(c("#EEEEEE", "#167D8D"))
  )
  map_panel(
    representative$pixel_col,
    representative$pixel_row,
    representative$tgfb2,
    "f  TGFB2 expression",
    colorRampPalette(c("#F3E5AB", "#D95F02", "#7F0000"))
  )
  map_panel(
    representative$pixel_col,
    representative$pixel_row,
    representative$receiver,
    "g  Myeloid x TGFBR score",
    colorRampPalette(c("#E8F1F2", "#2C7FB8", "#253494"))
  )
  plot.new()
  text(
    0,
    1,
    paste0(
      "Representative section\n", representative_sample,
      "\n\nSelected as the section closest\nto the cohort-median TGFB2 effect.",
      "\n\nInference uses 18 patient medians;\nspots and sections are not treated\nas independent patients."
    ),
    adj = c(0, 1),
    cex = 0.9
  )
  mtext(
    "Greenwald-Ravi spatial replication of AC-like TGFB2",
    outer = TRUE,
    side = 3,
    adj = 0,
    cex = 1.25,
    font = 2
  )
  dev.off()
}

draw_figure("png")
draw_figure("pdf")
png_path <- file.path(
  figure_dir,
  "SupplementaryFigureS6_Greenwald_spatial_validation.png"
)
png_image <- magick::image_read(png_path)
magick::image_write(png_image, png_path, density = "600x600")
cat(jsonlite::toJSON(analysis_summary, pretty = TRUE, auto_unbox = TRUE), "\n")
