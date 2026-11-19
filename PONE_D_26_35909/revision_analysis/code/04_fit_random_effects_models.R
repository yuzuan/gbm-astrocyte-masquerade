# Fit cohort-level REML models and leave-one-cohort-out analyses of within-patient TGFB2 differences.
# Input 03_model_inputs/01_model_patient_contrasts.csv contains analysis, cohort, patient and difference.
# Write cohort effects, pooled effects, leave-one-out estimates and R session info in 04_random_effects_models.
# From revision_analysis run Rscript code/04_fit_random_effects_models.R 03_result.

suppressPackageStartupMessages(library(metafor))

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) > 1) stop("Specify a single model results directory")
directory <- if (length(arguments)) arguments[1] else Sys.getenv("PLOS_REVISION_DIR", unset = "03_result")
result_dir <- normalizePath(directory, mustWork = TRUE)
input_table <- read.csv(file.path(result_dir, "03_model_inputs", "01_model_patient_contrasts.csv"))
if (anyDuplicated(input_table[c("analysis", "cohort", "patient")])) stop("A model contains duplicate patient keys")

# Estimate the mean and its variance within each cohort. At least three patients are required for pooling.
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
        if (!all(is.finite(values))) stop("Patient contrasts contain missing or nonfinite values")
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

    # The primary model uses modified Knapp-Hartung intervals. Fit the standard interval on the same cohorts.
    study_table_total <- do.call(rbind, cohort_tables)
    study_table_total$eligible <- study_table_total$n >= 3
    eligible_table <- study_table_total[study_table_total$eligible, ]
    if (nrow(eligible_table) < 3 || !all(eligible_table$vi > 0)) stop("Too few cohorts or invalid sampling variance")
    control <- list(threshold = 1e-10, maxiter = 1000)
    fit <- rma(yi, vi, data = eligible_table, method = "REML", test = "adhoc", control = control)
    kh <- rma(yi, vi, data = eligible_table, method = "REML", test = "knha", control = control)
    prediction <- predict(fit)
    limits <- confint(fit)$random
    if (!"tau^2" %in% rownames(limits)) stop("The metafor output is missing the tau-squared interval")

    # Leave I-squared missing for fewer than five cohorts. Missing weights identify cohorts excluded from the model.
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

    # Retain the name of each omitted eligible cohort for comparison with the full model.
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

# Combine each set of results once, then check the saved table dimensions and column names.
output_dir <- file.path(result_dir, "04_random_effects_models")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_tables <- list(
    "01_cohort_effects.csv" = do.call(rbind, study_tables),
    "02_pooled_effects.csv" = do.call(rbind, summary_tables),
    "03_leave_one_cohort_out.csv" = do.call(rbind, omission_tables)
)
for (filename in names(output_tables)) {
    output_table <- output_tables[[filename]]
    path <- file.path(output_dir, filename)
    write.csv(output_table, path, row.names = FALSE)
    saved_table <- read.csv(path)
    if (!identical(dim(output_table), dim(saved_table))) stop("Output table dimensions changed after writing")
    if (!identical(colnames(output_table), colnames(saved_table))) stop("Output column names changed after writing")
}
capture.output(sessionInfo(), file = file.path(output_dir, "04_r_session_info.txt"))
cat("Fitted ", length(summary_tables), " models and ", length(omission_tables), " leave-one-out models.\n", sep = "")
