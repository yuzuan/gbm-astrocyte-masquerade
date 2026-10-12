# 返修：患者内均值差的 REML 随机效应合并；保留小队列不确定性。
suppressPackageStartupMessages(library(metafor))
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 1)
out <- normalizePath(args[1], mustWork=TRUE)
d <- read.csv(file.path(out, 'analysis', 'meta_patient_inputs.csv'))
studies <- list(); summaries <- list(); omissions <- list()
for (a in unique(d$analysis)) {
  dd <- d[d$analysis == a, ]
  s <- do.call(rbind, lapply(split(dd, dd$cohort), function(x) {
    stopifnot(all(is.finite(x$difference)), !anyDuplicated(x$patient))
    n <- nrow(x); v <- var(x$difference)/n
    data.frame(analysis=a, cohort=x$cohort[1], n=n,
               yi=mean(x$difference), vi=v, median=median(x$difference),
               positives=sum(x$difference > 0),
               ci_low=mean(x$difference)-qt(.975,n-1)*sqrt(v),
               ci_high=mean(x$difference)+qt(.975,n-1)*sqrt(v))
  }))
  s$eligible <- s$n >= 3
  ss <- s[s$eligible, ]
  stopifnot(nrow(ss)>=3, all(ss$vi>0))
  fit <- rma(yi, vi, data=ss, method='REML', test='adhoc', control=list(threshold=1e-10,maxiter=1000))
  kh <- rma(yi, vi, data=ss, method='REML', test='knha', control=list(threshold=1e-10,maxiter=1000))
  pr <- predict(fit)
  het <- confint(fit)$random
  s$weight_pct <- NA_real_
  s$weight_pct[s$eligible] <- as.numeric(weights(fit))
  studies[[a]] <- s
  summaries[[a]] <- data.frame(analysis=a,k=fit$k,n=sum(ss$n),estimate=as.numeric(fit$b),
    se=fit$se,ci_low=fit$ci.lb,ci_high=fit$ci.ub,p_value=fit$pval,
    tau2=fit$tau2,tau2_ci_low=het[1,'ci.lb'],tau2_ci_high=het[1,'ci.ub'],
    Q=fit$QE,Q_p=fit$QEp,I2=if(fit$k>=5)fit$I2 else NA_real_,
    prediction_low=pr$pi.lb,prediction_high=pr$pi.ub,
    KH_ci_low=kh$ci.lb,KH_ci_high=kh$ci.ub,KH_p=kh$pval,
    method='REML + modified Knapp-Hartung (scale >= 1)')
  for (co in ss$cohort) {
    om <- ss[ss$cohort != co,]
    f <- rma(yi,vi,data=om,method='REML',test='adhoc',control=list(threshold=1e-10,maxiter=1000))
    omissions[[paste(a,co)]] <- data.frame(analysis=a,omitted=co,k=f$k,n=sum(om$n),
      estimate=as.numeric(f$b),ci_low=f$ci.lb,ci_high=f$ci.ub,p_value=f$pval,tau2=f$tau2)
  }
}
write.csv(do.call(rbind,studies),file.path(out,'analysis','meta_cohort_effects.csv'),row.names=FALSE)
write.csv(do.call(rbind,summaries),file.path(out,'analysis','meta_summary.csv'),row.names=FALSE)
write.csv(do.call(rbind,omissions),file.path(out,'analysis','meta_leave_one_out.csv'),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(out,'validation','R_session_info.txt'))
print(do.call(rbind,summaries))
