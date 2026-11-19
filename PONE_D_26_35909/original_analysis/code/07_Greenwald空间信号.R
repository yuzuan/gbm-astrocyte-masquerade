#!/usr/bin/env Rscript
# 目的：按 Greenwald 与 Ravi 的点位注释比较 AC、其它恶性和反应性星形胶质区域，并估计邻近髓系受体差。
# 输入：--input-dir 下 general/GBM_samples.txt、general/exp_mats_GBM/*counts.rds、
# MP/mp_assign_124/*.rds（SpotID、spot_type_meta_new）、general/GBM_data/样本/outs/spatial/tissue_positions_list.csv。
# 元数据由 --metadata 或 general/sample_metadata.csv 提供，须含 sample、patient、region、source。
# 输出：默认在 external_validation/07_Greenwald空间信号 写 01_切片配体指标.csv、02_患者配体指标.csv、
# 03_切片空间指标.csv、04_患者空间指标.csv、05_配体汇总.csv、06_代表切片点位.csv、07_分析摘要.json、
# 08_FigureS6_Greenwald空间信号.pdf 和 08_FigureS6_Greenwald空间信号.png。
# 运行方式：Rscript 07_Greenwald空间信号.R --input-dir 原始目录 --metadata 同编码元数据.csv [--output-dir 复算目录]
# RStudio 可打开后 Source，使用默认输入目录。按本项目转换器例外，公开的 pdftools/Poppler 从同一 PDF 生成 600 dpi PNG。
# fastR 1.9.1 的本地许可为模板，未核到同版可再分发来源。

library(jsonlite)
library(pdftools)

# RStudio 的 Source 记录源文件名；独立命令行由 --file 提供。所有默认路径均以脚本所在的包为起点。
script_file <- NULL
for (frame in sys.frames()) if (!is.null(frame$ofile)) script_file <- frame$ofile


if (is.null(script_file)) {
    file_argument <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    if (length(file_argument) != 1L) stop("请用 RStudio Source 或 Rscript 启动本脚本")
    script_file <- sub("^--file=", "", file_argument)
    script_file <- gsub("~+~", " ", script_file, fixed = TRUE)
}
package_dir <- dirname(dirname(normalizePath(script_file, mustWork = TRUE)))
external_dir <- file.path(package_dir, "external_validation")

# 命令行路径成对提供；缺参数、重复参数或给错开关时说明具体错误。
arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) %% 2L != 0L) stop("每个选项后应给出一个路径")
options_by_name <- list()
if (length(arguments)) {
    for (i in seq.int(1L, length(arguments), by = 2L)) {
        option <- arguments[i]
        if (!option %in% c("--input-dir", "--metadata", "--output-dir")) stop("不支持的选项 ", option)
        if (option %in% names(options_by_name)) stop("重复的选项 ", option)
        options_by_name[[option]] <- path.expand(arguments[i + 1L])
    }
}
input_dir <- options_by_name[["--input-dir"]]
if (is.null(input_dir)) input_dir <- file.path(external_dir, "raw", "greenwald_inputs")
samples <- readLines(file.path(input_dir, "general", "GBM_samples.txt"), warn = FALSE)
samples <- samples[nzchar(samples)]
if (!length(samples) || anyDuplicated(samples)) stop("GBM_samples.txt 为空或含重复切片")

# 公开模板只适配同编码的原文件。显式提供的本地表可以有其它列，这里只取四列原始元数据。
metadata_path <- options_by_name[["--metadata"]]
if (is.null(metadata_path)) {
    metadata_path <- file.path(input_dir, "general", "sample_metadata.csv")
    if (!file.exists(metadata_path)) {
        metadata_path <- file.path(package_dir, "input_templates", "greenwald_sample_metadata.csv")
    }
}
metadata_table <- read.csv(metadata_path, colClasses = "character", check.names = FALSE)
required <- c("sample", "patient", "region", "source")
if (!all(required %in% colnames(metadata_table))) stop("元数据须含 sample、patient、region、source")
metadata_table <- metadata_table[, required, drop = FALSE]
if (anyNA(metadata_table[, c("sample", "patient", "source")])) stop("元数据缺切片、患者或来源")
if (anyDuplicated(metadata_table$sample) || !setequal(samples, metadata_table$sample)) {
    stop("切片编号与元数据不匹配，请用 --metadata 指定与原始文件同编码的表")
}
if (!all(metadata_table$source %in% c("Greenwald2024", "Ravi2022"))) stop("来源标签须为 Greenwald2024 或 Ravi2022")

# 每张切片分别读取基因×点位的计数矩阵、作者点位标签和无标题的 Visium 坐标文件。
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
    if (!all(target_genes %in% rownames(count_matrix))) stop(sample_id, " 缺少预设目标基因")
    annotation_path <- file.path(input_dir, "MP", "mp_assign_124", paste0(sample_id, ".rds"))
    annotation_table <- readRDS(annotation_path)
    annotation_table <- as.data.frame(annotation_table)
    if (!all(c("SpotID", "spot_type_meta_new") %in% colnames(annotation_table))) stop(sample_id, " 缺少作者点位标签")

    # 位置文件原本无表头，按 10x 原顺序命名。按计数列的条码顺序对齐三份文件，仅保留组织内有标签点。
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

    # 分母为每个点位的全基因文库。仅将小型目标基因矩阵转为密集矩阵做逐列归一化和转置。
    libraries <- colSums(count_matrix[, common, drop = FALSE])
    selected_matrix <- as.matrix(count_matrix[target_genes, common, drop = FALSE])
    normalized_matrix <- log1p(sweep(selected_matrix, 2L, pmax(libraries, 1), `/`) * 10000)
    expression_table <- as.data.frame(t(normalized_matrix))
    rm(count_matrix, selected_matrix, normalized_matrix)

    # AC、其它恶性、反应性星形胶质和髓系沿用作者标签。非髓系点的受体信号按原方案为零。
    ac <- labels == "AC"
    other_malignant <- labels %in% c("OPC", "NPC", "MES", "MES.Hyp", "MES.Ast", "Prolif.Metab", "chromatin.reg")
    reactive <- labels == "Reactive.Ast"
    myeloid <- labels %in% c("Mac", "Inflammatory.Mac")
    ac_score <- rowMeans(expression_table[, ac_markers, drop = FALSE])
    receiver <- ifelse(myeloid, rowMeans(expression_table[, receptors, drop = FALSE]), 0)

    # 一列对应一个 Visium 邻位方向，六个偏移不含原点。没有有效邻点的行保留 NA。
    coordinates <- paste(position_table$array_row, position_table$array_col, sep = ":")
    if (anyDuplicated(coordinates)) stop(sample_id, " 有重复的 Visium 坐标")
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

    # 每个差值的两组各需至少 20 个点。每次取同一切片同一配体的平均表达差、检出率差和秩相关。
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

    # 仅在 AC 标签点比较 TGFB2 阳性上半部和零表达；两组各至少十个有邻点的源点才估计邻位差。
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

# 输入一名患者某指标的各切片结果，返回有限结果的中位数，无有效切片时返回 NA。
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

# 对患者差值的均值做精确双侧符号翻转。分块穷举符号，空输入返回 NA；容差与原检验相同。
proximity_values <- patient_spatial_table$tgfb2_neighbor_receiver_delta
signflip_p <- function(values) {
    values <- values[is.finite(values)]
    count <- length(values)
    result <- NA_real_
    if (!count) return(result)
    observed <- abs(mean(values))
    pattern_count <- 2^count
    extreme <- 0

    # 每个 bit 对应一位患者。用整除提取符号，保持旧序列中零位为 −1、一位为 +1。
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

# Wilcoxon 检验先去零差；至少两个有限的非零值才计算。保留原 R 渐近检验及连续性校正。
wilcoxon_p <- function(values) {
    values <- values[is.finite(values) & values != 0]
    result <- NA_real_
    if (length(values) >= 2L) result <- suppressWarnings(wilcox.test(values, mu = 0, exact = FALSE)$p.value)
    return(result)
}
proximity_wilcoxon <- wilcoxon_p(proximity_values)

# 六种配体分别汇总 AC 对其它恶性和对反应性星形胶质的差值。两种比较各自在配体之间做 BH 校正。
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

# 代表切片为 TGFB2 主差值距切片中位数最近者；并列沿用元数据输入清单的次序。
keep <- sample_ligand_table_total$ligand == "TGFB2" & is.finite(sample_ligand_table_total$ac_minus_other_mean)
tgfb2_sample_table <- sample_ligand_table_total[keep, , drop = FALSE]
if (!nrow(tgfb2_sample_table)) stop("没有可估计的 TGFB2 切片作为代表")
distance <- abs(tgfb2_sample_table$ac_minus_other_mean - median(tgfb2_sample_table$ac_minus_other_mean))
representative_sample <- tgfb2_sample_table$sample[which.min(distance)]
representative_table <- plot_tables[[representative_sample]]

# 核实切片×配体、患者×配体及代表点位数量。元数据中的来源与患者分别计数，不按切片数计患者。
patient_sources_table <- unique(metadata_table[, c("patient", "source")])
if (anyDuplicated(patient_sources_table$patient)) stop("元数据把同一患者分给了两个来源")
if (!setequal(patient_spatial_table$patient, patient_sources_table$patient)) stop("患者空间表与元数据不匹配")
if (!all(table(sample_ligand_table_total$sample, sample_ligand_table_total$ligand) == 1L)) stop("切片配体组合不完整")
if (!all(table(patient_ligand_table$patient, patient_ligand_table$ligand) == 1L)) stop("患者配体组合不完整")
if (!setequal(sample_spatial_table_total$sample, samples)) stop("切片空间表缺切片")
if (!setequal(summary_table_total$ligand, ligands)) stop("汇总表缺配体")
spot_count <- sample_spatial_table_total$n_spots[match(representative_sample, sample_spatial_table_total$sample)]
if (nrow(representative_table) != spot_count) stop("代表切片点位数与空间表不一致")

# 六份表先在同目录写临时 CSV，核原列顺序、行数、全文本及数值后替换目标。文件名对应独立复算目录。
output_dir <- options_by_name[["--output-dir"]]
if (is.null(output_dir)) output_dir <- file.path(external_dir, "07_Greenwald空间信号")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_tables <- list(
    "01_切片配体指标.csv" = sample_ligand_table_total,
    "02_患者配体指标.csv" = patient_ligand_table,
    "03_切片空间指标.csv" = sample_spatial_table_total,
    "04_患者空间指标.csv" = patient_spatial_table,
    "05_配体汇总.csv"   = summary_table_total,
    "06_代表切片点位.csv" = representative_table
)
for (file_name in names(output_tables)) {
    result_table <- output_tables[[file_name]]
    csv_path <- tempfile(tmpdir = output_dir, fileext = ".csv")
    write.csv(result_table, csv_path, row.names = FALSE, na = "NA")
    classes <- ifelse(vapply(result_table, is.character, logical(1)), "character", "numeric")
    saved_table <- read.csv(csv_path, colClasses = classes, check.names = FALSE, na.strings = "NA")
    if (!identical(colnames(saved_table), colnames(result_table)) || nrow(saved_table) != nrow(result_table)) {
        stop(file_name, " 回读时行数或字段顺序不一致")
    }
    for (column in colnames(result_table)) {
        comparison <- all.equal(
            result_table[, column], saved_table[, column], tolerance = 1e-14, check.attributes = FALSE
        )
        if (!isTRUE(comparison)) stop(file_name, " 回读后有差异的列为 ", column)
    }
    if (!file.rename(csv_path, file.path(output_dir, file_name))) stop("无法保存 ", file_name)
}

# 来源计数直接来自输入的 sample/patient/source 关系；历史公开 JSON 以四位小数保存，CSV 另保留精度。
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
if (!identical(names(saved_summary), names(analysis_summary))) stop("摘要回读后字段不全")
if (saved_summary$n_greenwald_patients + saved_summary$n_ravi_patients != saved_summary$n_patients) stop("来源人数不符")
if (saved_summary$n_samples != nrow(sample_spatial_table_total)) stop("摘要切片数不符")
if (saved_summary$tgfb2$ligand != "TGFB2") stop("摘要配体不符")
if (!file.rename(json_path, file.path(output_dir, "07_分析摘要.json"))) stop("无法保存 JSON 摘要")

# 原稿先画 PNG、再画 PDF。先跳过第一套点位置所用的均匀数，使 PDF 保留历史第二次绘图的抖动序列。
tgfb2_patient_table <- patient_ligand_table[patient_ligand_table$ligand == "TGFB2", , drop = FALSE]
effect_sets <- list(
    tgfb2_patient_table$ac_minus_other_mean, patient_ligand_table$ac_minus_other_mean,
    tgfb2_patient_table$ac_minus_reactive_mean, proximity_values
)
jitter_count <- 0L
for (values in effect_sets) jitter_count <- jitter_count + sum(is.finite(values))
set.seed(20260716)
invisible(runif(jitter_count))

# 在独立设备记录整页 base R 图。退出时关闭该设备；返回可在 PDF 中重绘的图对象。
record_figure <- function() {
    pdf(NULL, width = 12, height = 8, useDingbats = FALSE)
    on.exit(dev.off())
    dev.control(displaylist = "enable")
    par(mfrow = c(2, 4), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

    # 按 a、b、c、d 顺序绘制。抖动仅作用于横坐标，统计量取未经抖动的原值。
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

    # 空间面板各用该指标 1%–99% 分位数着色；常量切片以扩展半单位的范围分箱。
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

# PDF 母版必须是一页且包含指定空间面板；PNG 由该页转出，回读文件头核实 600 dpi 对应的像素宽高。
pdf_path <- tempfile(tmpdir = output_dir, fileext = ".pdf")
pdf(pdf_path, width = 12, height = 8, useDingbats = FALSE)
    print(figure)
invisible(dev.off())
page_info <- pdf_info(pdf_path)
if (page_info$pages != 1L || !grepl("TGFB2 expression", pdf_text(pdf_path), fixed = TRUE)) stop("PDF 页数或空间面板不符")
png_path <- sub("\\.pdf$", ".png", pdf_path)
invisible(pdf_convert(pdf_path, format = "png", pages = 1L, filenames = png_path, dpi = 600, verbose = FALSE))
png_header <- readBin(png_path, "raw", n = 24L)
if (length(png_header) != 24L) stop("PNG 文件头不完整")
if (!identical(png_header[1:8], as.raw(c(137, 80, 78, 71, 13, 10, 26, 10)))) stop("PNG 文件格式不符")
pixels <- readBin(png_header[17:24], "integer", n = 2L, size = 4L, endian = "big")
page_table <- pdf_pagesize(pdf_path)
expected_pixels <- round(c(page_table$width, page_table$height) * 600 / 72)
if (!all(pixels == expected_pixels)) stop("PNG 像素数与 PDF 页宽高不符")
if (!file.rename(pdf_path, file.path(output_dir, "08_FigureS6_Greenwald空间信号.pdf"))) stop("无法保存 PDF")
if (!file.rename(png_path, file.path(output_dir, "08_FigureS6_Greenwald空间信号.png"))) stop("无法保存 PNG")
cat("切片 ", nrow(sample_spatial_table_total), " 张，患者 ", nrow(patient_spatial_table), " 名\n", sep = "")
