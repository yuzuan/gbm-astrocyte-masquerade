# 目的　按队列汇总患者内 TGFB2 表达差，再拟合 REML 随机效应模型和逐队列剔除模型。
# 输入　结果目录中的 03_患者模型输入/01_模型患者表达差.csv，含 analysis、cohort、patient 和 difference。
# 输出　04_随机效应模型/01_队列效应量.csv、02_合并效应量.csv、03_逐队列剔除.csv、04_R环境.txt。
# 运行　在 revision_analysis 下运行 Rscript code/04_拟合随机效应模型.R 03_result。

suppressPackageStartupMessages(library(metafor))

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) > 1) stop("只需传入一个患者模型结果目录")
directory <- if (length(arguments)) arguments[1] else Sys.getenv("PLOS_REVISION_DIR", unset = "03_result")
result_dir <- normalizePath(directory, mustWork = TRUE)
input_table <- read.csv(file.path(result_dir, "03_患者模型输入", "01_模型患者表达差.csv"))
if (anyDuplicated(input_table[c("analysis", "cohort", "patient")])) stop("同一模型内有重复的队列及患者组合")

# 每个模型先在各队列内求均值与均值的方差。至少三名患者才进入合并模型，人数较少的队列仍在明细表内。
study_tables <- list()
summary_tables <- list()
omission_tables <- list()
for (scenario in unique(input_table$analysis)) {
    scenario_table <- input_table[input_table$analysis == scenario, ]
    groups <- split(scenario_table, scenario_table$cohort)
    cohort_tables <- list()
    for (cohort in names(groups)) {
        patient_table <- groups[[cohort]]
        values <- patient_table$difference
        if (!all(is.finite(values))) stop("患者对照值缺失或非有限数")
        n_patient <- nrow(patient_table)
        estimate <- mean(values)
        variance <- var(values) / n_patient
        half_width <- qt(0.975, n_patient - 1) * sqrt(variance)
        cohort_tables[[cohort]] <- data.frame(
            analysis  = scenario,
            cohort    = cohort,
            n         = n_patient,
            yi        = estimate,
            vi        = variance,
            median    = median(values),
            positives = sum(values > 0),
            ci_low    = estimate - half_width,
            ci_high   = estimate + half_width
        )
    }

    # 主结果用修正的 Knapp–Hartung，另保留标准 KH 区间。两种拟合共享同一组队列与收敛条件。
    study_table_total <- do.call(rbind, cohort_tables)
    study_table_total$eligible <- study_table_total$n >= 3
    eligible_table <- study_table_total[study_table_total$eligible, ]
    if (nrow(eligible_table) < 3 || !all(eligible_table$vi > 0)) stop("有效队列不足或队列均值方差无效")
    control <- list(threshold = 1e-10, maxiter = 1000)
    fit <- rma(yi, vi, data = eligible_table, method = "REML", test = "adhoc", control = control)
    kh <- rma(yi, vi, data = eligible_table, method = "REML", test = "knha", control = control)
    prediction <- predict(fit)
    limits <- confint(fit)$random
    if (!"tau^2" %in% rownames(limits)) stop("metafor 输出缺少 tau^2 区间")

    # 不足五个队列时按发表口径留空 I²。缺少权重表示该队列未纳入合并。
    study_table_total$weight_pct <- NA_real_
    study_table_total$weight_pct[study_table_total$eligible] <- as.numeric(weights(fit))
    study_tables[[scenario]] <- study_table_total
    i_squared <- if (fit$k >= 5) fit$I2 else NA_real_
    summary_tables[[scenario]] <- data.frame(
        analysis        = scenario,
        k               = fit$k,
        n               = sum(eligible_table$n),
        estimate        = as.numeric(fit$b),
        se              = fit$se,
        ci_low          = fit$ci.lb,
        ci_high         = fit$ci.ub,
        p_value         = fit$pval,
        tau2            = fit$tau2,
        tau2_ci_low     = limits["tau^2", "ci.lb"],
        tau2_ci_high    = limits["tau^2", "ci.ub"],
        Q               = fit$QE,
        Q_p             = fit$QEp,
        I2              = i_squared,
        prediction_low  = prediction$pi.lb,
        prediction_high = prediction$pi.ub,
        KH_ci_low       = kh$ci.lb,
        KH_ci_high      = kh$ci.ub,
        KH_p            = kh$pval,
        method          = "REML + modified Knapp-Hartung (scale >= 1)"
    )

    # 逐一剔除合格队列。每一行保留剔除的队列名，便于与完整模型逐行比较。
    for (omitted in eligible_table$cohort) {
        remaining_table <- eligible_table[eligible_table$cohort != omitted, ]
        fit <- rma(yi, vi, data = remaining_table, method = "REML", test = "adhoc", control = control)
        omission_tables[[paste(scenario, omitted)]] <- data.frame(
            analysis = scenario,
            omitted  = omitted,
            k        = fit$k,
            n        = sum(remaining_table$n),
            estimate = as.numeric(fit$b),
            ci_low   = fit$ci.lb,
            ci_high  = fit$ci.ub,
            p_value  = fit$pval,
            tau2     = fit$tau2
        )
    }
}

# 每张结果表只合并一次，写出后核对表头和行数。
output_dir <- file.path(result_dir, "04_随机效应模型")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_tables <- list(
    "01_队列效应量.csv" = do.call(rbind, study_tables),
    "02_合并效应量.csv" = do.call(rbind, summary_tables),
    "03_逐队列剔除.csv" = do.call(rbind, omission_tables)
)
for (filename in names(output_tables)) {
    output_table <- output_tables[[filename]]
    path <- file.path(output_dir, filename)
    write.csv(output_table, path, row.names = FALSE)
    saved_table <- read.csv(path)
    if (!identical(dim(output_table), dim(saved_table))) stop("写出的表格维数不符")
    if (!identical(colnames(output_table), colnames(saved_table))) stop("写出的结果列名不符")
}
capture.output(sessionInfo(), file = file.path(output_dir, "04_R环境.txt"))
cat("已拟合 ", length(summary_tables), " 种模型，完成 ", length(omission_tables), " 次逐队列剔除。\n", sep = "")
