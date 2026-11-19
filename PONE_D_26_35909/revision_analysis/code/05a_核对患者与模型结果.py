"""目的　用患者表核对纳入条件，用 SciPy 独立计算 REML 与修正 Knapp–Hartung。
输入　01、02 的计数与患者表，03、04 的患者、队列和模型结果表。
输出　05_补充图与工作簿/01_模型核对.json。每一项保留实际比较对象和通过情况。
运行　在 revision_analysis 下运行本脚本。轻量笔记本使用随库提供的计数核对表。
"""

import json
from importlib import import_module

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import t

patient = import_module("00a_患者表达")
contrast_table = patient.read_table("preprocessing_patient_contrasts")
input_table = patient.read_table("meta_patient_inputs")
checks = {}
details = {}

# 只有同一队列内的同一患者才能配对。基准情景的差值应复现进入模型前的患者差值。
for method in ["primary", "uniform"]:
    observed_table = contrast_table.query("scenario == 'stored_X' and labels == @method")
    analysis = method + "_labels"
    reference_table = input_table.loc[input_table["analysis"] == analysis]
    paired_table = observed_table.merge(
        reference_table, on=["cohort", "patient"], validate="one_to_one", suffixes=("_observed", "_reference")
    )
    error = np.abs(paired_table["difference_observed"] - paired_table["difference_reference"])
    name = method + " frozen patient contrasts reproduced"
    details[name] = f"max difference {error.max():.3g}"
    checks[name] = (
        len(paired_table) == len(observed_table) == len(reference_table) and error.max() < 1e-6
    )
identifiers = ["scenario", "labels", "cohort", "patient"]
checks["patient keys unique"] = not contrast_table.duplicated(identifiers).any()

# 将来源文件的核对记录与患者纳入记录相对照。公开笔记本读取的是已归档的来源核对记录。
audit_table = patient.read_table("count_input_audit")
expected_objects = {
    "scGBM_GSE162631_GSE273274.h5ad",
    "snGBM_GSE276177_GSE286419.h5ad",
    "regional_GSE274987_GSE241037.h5ad",
    "GSE211376 GEO counts",
}
checks["all four count inputs integer"] = (
    audit_table["object"].is_unique
    and set(audit_table["object"]) == expected_objects
    and audit_table["nonnegative_integer_counts"].eq(True).all()
)

# GEO 的计数表未独立存储 X，其历史误差列以零占位。只在有 X 的 H5AD 上核对逐对象最大误差。
stored_rows = audit_table.loc[audit_table["object"].str.endswith(".h5ad"), "max_abs_X_reconstruction_error"]
checks["three stored X matrices reconstructed"] = (
    np.isfinite(stored_rows).all() and stored_rows.between(0, 1e-6, inclusive="left").all()
)
raw_table = patient.read_table("GSE162631_public_count_audit")
source_eligible_table = patient.read_table("GSE162631_source_eligibility")
eligibility_table = patient.read_table("preprocessing_eligibility")
gse_patients = set(eligibility_table.loc[eligibility_table["cohort"] == "GSE162631", "patient"])
source_patients = {patient.patient_from_tumor_sample(sample) for sample in raw_table["sample"]}
checks["four public 10x tumor count files agree with retained panel"] = (
    raw_table["sample"].is_unique
    and len(raw_table) == len(gse_patients)
    and source_patients == gse_patients
    and set(source_eligible_table["patient"]) == gse_patients
    and raw_table["max_panel_count_disagreement"].eq(0).all()
)


cohort_table = patient.read_table("meta_cohort_effects")
model_table = patient.read_table("meta_summary")

# 独立实现同一 REML 目标函数，在边界零和内部最优点之间选择。返回负的限制对数似然（略去常数）。
def restricted_objective(tau_squared, effect, variance):
    weights = 1 / (variance + tau_squared)
    mean = np.sum(weights * effect) / sum(weights)
    objective = 0.5 * (
        np.log(variance + tau_squared).sum() + np.log(weights.sum()) + np.sum(weights * (effect - mean) ** 2)
    )
    return objective


for row in model_table.itertuples():
    eligible_table = cohort_table.loc[(cohort_table["analysis"] == row.analysis) & cohort_table["eligible"]]
    effect = eligible_table["yi"].to_numpy()
    variance = eligible_table["vi"].to_numpy()
    fit = minimize_scalar(
        restricted_objective,
        args=(effect, variance),
        bounds=(0, max(1, np.var(effect) * 10)),
        method="bounded",
        options={"xatol": 1e-13},
    )
    tau = 0.0 if restricted_objective(0, effect, variance) <= fit.fun else fit.x

    # 区间以队列数减一为自由度，修正 KH 的尺度最低为一。按统计量名称与 R 输出一一对应。
    weights = 1 / (variance + tau)
    estimate = np.sum(weights * effect) / sum(weights)
    degrees = len(effect) - 1
    scale = max(1.0, np.sum(weights * (effect - estimate) ** 2) / degrees)
    error = np.sqrt(scale / sum(weights))
    low, high = estimate + np.array([-1, 1]) * t.ppf(0.975, degrees) * error
    probability = 2 * t.sf(abs(estimate / error), degrees)
    calculated = np.array([estimate, low, high, probability, tau])
    published = np.array([row.estimate, row.ci_low, row.ci_high, row.p_value, row.tau2])
    checks["independent REML and modified-KH: " + row.analysis] = (
        fit.success and np.abs(calculated - published).max() < 5e-6
    )

checks["all meta CI ordered and finite"] = (
    np.isfinite(model_table[["estimate", "ci_low", "ci_high"]]).all().all()
    and model_table["ci_low"].lt(model_table["estimate"]).all()
    and model_table["ci_high"].gt(model_table["estimate"]).all()
)
checks["cohort weights sum to 100"] = (
    np.abs(cohort_table.groupby("analysis")["weight_pct"].sum() - 100).lt(1e-8).all()
)
checks["joint regional source study not split"] = not input_table["cohort"].isin(["GSE274986"]).any()

# 来源计数场景与基准的方向逐患者核对，显示纳入患者的两个细胞组是否都达到20个。
source_table = patient.read_table("GSE162631_source_sensitivity")
baseline_table = contrast_table.query(
    "cohort == 'GSE162631' and labels == 'uniform' and scenario == 'stored_X'"
)
baseline_direction = np.sign(baseline_table.set_index("patient")["difference"]).sort_index()
directions_match = []
for _, scenario_table in source_table.groupby("scenario"):
    observed_direction = np.sign(scenario_table.set_index("patient")["difference"]).sort_index()
    directions_match.append(observed_direction.equals(baseline_direction))
expected_scenarios = {"source_full_gene_10k", "source_original_tumor_concordant"}
checks["source count reprocessing retains discordance"] = (
    set(source_table["scenario"]) == expected_scenarios
    and (baseline_direction > 0).any()
    and (baseline_direction < 0).any()
    and all(directions_match)
)
meets_eligibility = (eligibility_table["n_ac"] >= 20) & (eligibility_table["n_other"] >= 20)
checks["20-cell eligibility consistent"] = eligibility_table["eligible"].eq(meets_eligibility).all()
checks["all displayed eligible contrasts have >=20 cells"] = (contrast_table[["n_ac", "n_other"]] >= 20).all().all()

# 笔记本和全计数路线共用这一份科学核对。发生差异时列出具体比较名称。
checks = {name: bool(value) for name, value in checks.items()}
failed = [name for name, value in checks.items() if not value]
records = []
for name, passed in checks.items():
    records.append({"check": name, "passed": passed, "detail": details.get(name, "")})
report = {
    "passed": sum(checks.values()),
    "failed": len(failed),
    "checks": records,
    "scope": "Scientific numeric/input checks, not journal acceptance or server-upload validation",
}
path = patient.result_root() / "05_补充图与工作簿/01_模型核对.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
if json.loads(path.read_text(encoding="utf-8")) != report or failed:
    raise ValueError(f"模型或输入核对失败：{failed}")
print("患者与模型核对", report["passed"], "项通过")
