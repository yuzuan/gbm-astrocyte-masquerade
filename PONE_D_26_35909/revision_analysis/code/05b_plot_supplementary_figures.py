"""Plot the S15 forest plot, S16 preprocessing comparisons and S17 source-count comparisons.
Inputs are the eligibility, patient, cohort and REML tables from stages 01 through 04.
Write S15, S16 and S17 as PDF and PNG in 05_supplementary_outputs.
From revision_analysis run python code/05b_plot_supplementary_figures.py or execute the reproduction notebook.
"""

import sys
from importlib import import_module
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

patient = import_module("00a_patient_expression")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
style = import_module("00a_figure_style")
style.configure_plottie_style()
plt.rcParams["text.color"] = style.ink

# S15 has one row per included cohort. Point areas reflect REML weights. the diamond shows the pooled estimate and CI.
cohort_table = patient.read_table("meta_cohort_effects")
model_table = patient.read_table("meta_summary").set_index("analysis")
figure, axes = plt.subplots(2, 1, figsize=(7.1, 6.8))
figure.subplots_adjust(hspace=0.95, left=0.225, right=0.70, bottom=0.17, top=0.94)
forest_panels = [
    ("primary_labels", "Original labels", style.teal, "a"),
    ("uniform_labels", "Uniform labels", style.coral, "b"),
]
for axis, (key, title, color, letter) in zip(axes, forest_panels):
    eligible_table = cohort_table.loc[(cohort_table["analysis"] == key) & cohort_table["eligible"]]
    model = model_table.loc[key]
    positions = np.arange(len(eligible_table), 0, -1)
    axis.axvline(0, color=style.gray_dark, lw=0.65, ls="--")
    labels = []
    for y, row in zip(positions, eligible_table.itertuples()):
        axis.plot([row.ci_low, row.ci_high], [y, y], color=color, lw=1)
        axis.scatter(row.yi, y, s=row.weight_pct * 1.6, marker="s", color=color, zorder=3)
        interval = f"{row.yi:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]"
        axis.text(1.04, y, interval, transform=axis.get_yaxis_transform(), va="center", fontsize=6.8)
        accession = "GSE274986/7" if row.cohort == "GSE274987" else row.cohort
        labels.append(f"{accession}  (n={row.n})")

    vertices = [[model.ci_low, 0], [model.estimate, 0.20], [model.ci_high, 0], [model.estimate, -0.20]]
    axis.add_patch(Polygon(vertices, color=color))
    interval = f"{model.estimate:.3f} [{model.ci_low:.3f}, {model.ci_high:.3f}]"
    axis.text(
        1.04,
        0,
        interval,
        transform=axis.get_yaxis_transform(),
        va="center",
        fontsize=6.8,
        fontweight="semibold",
    )
    axis.set_yticks([*positions, 0], [*labels, "Random effects"])
    axis.set_ylim(-0.7, len(eligible_table) + 0.8)
    axis.set_xlim(-0.22, 0.53)
    axis.set_title(f"{title}: {int(model.n)} patients, {int(model.k)} cohorts", loc="left")
    axis.set_xlabel("Mean within-patient difference (log1p / 10,000)")
    axis.text(
        1.04,
        len(eligible_table) + 0.55,
        "Estimate [95% CI]",
        transform=axis.get_yaxis_transform(),
        fontweight="semibold",
        fontsize=7,
    )

    # Show the tau-squared interval for the small-cohort panel and I-squared and prediction intervals where available.
    detail = f"REML; modified KH; P={model.p_value:.4f}; τ²={model.tau2:.4f}; Q P={model.Q_p:.3g}"
    if key == "uniform_labels":
        detail += f"\nI²={model.I2:.1f}%; prediction interval [{model.prediction_low:.3f}, {model.prediction_high:.3f}]"
    else:
        detail += f"\nτ² 95% CI [{model.tau2_ci_low:.4f}, {model.tau2_ci_high:.4f}]"
        detail += "; heterogeneity is imprecisely estimated"
    axis.text(0, -0.40, detail, transform=axis.transAxes, fontsize=6.8, color=style.muted)
    style.open_axes(axis)
    style.panel_label(axis, letter)
output_dir = patient.result_root() / "05_supplementary_outputs"
style.save_figure(figure, output_dir, "02_s15_random_effects_forest_plot", panels="ab")

# Each S16 panel uses its own expression units. Read cohort counts, patient counts, points and intervals from the fits.
panels = [
    (
        "Log1p / 10,000",
        [
            ("Original labels", "primary_labels"),
            ("Rebuilt counts", "preproc__primary__counts_10k"),
            ("Stricter QC", "preproc__primary__counts_10k_strictQC"),
            ("Uniform labels", "uniform_labels"),
            ("Uniform + stricter QC", "preproc__uniform__counts_10k_strictQC"),
        ],
    ),
    (
        "Log1p / 1,000",
        [
            ("Original labels", "preproc__primary__counts_1k"),
            ("Uniform: labels frozen", "preproc__uniform_frozen__counts_1k"),
            ("Uniform: reassigned", "preproc__uniform__counts_1k"),
        ],
    ),
    (
        "Pseudobulk log2 CPM",
        [
            ("Original labels", "preproc__primary__pseudobulk"),
            ("Uniform labels", "preproc__uniform__pseudobulk"),
        ],
    ),
]
figure, axes = plt.subplots(3, 1, figsize=(7.1, 7.4), gridspec_kw={"height_ratios": [5, 3, 2]})
figure.subplots_adjust(hspace=0.95, left=0.27, right=0.95, bottom=0.16, top=0.95)
for axis, (title, entries), letter in zip(axes, panels, "abc"):
    labels = []
    for y, (label, key) in enumerate(reversed(entries)):
        model = model_table.loc[key]
        color = style.coral if "uniform" in key else style.teal
        distances = [[model.estimate - model.ci_low], [model.ci_high - model.estimate]]
        axis.errorbar(model.estimate, y, xerr=distances, fmt="o", color=color, capsize=2, ms=4.2, lw=1)
        labels.append(f"{label}\nn={int(model.n)}, k={int(model.k)}")
    axis.axvline(0, color=style.gray_dark, ls="--", lw=0.65)
    axis.set_yticks(range(len(entries)), labels)
    axis.set_ylim(-0.65, len(entries) - 0.35)
    axis.set_title(title, loc="left")
    axis.set_xlabel("Pooled difference [95% CI]")
    style.open_axes(axis)
    style.panel_label(axis, letter)

# Report whether GSE162631 remained eligible under reassignment and strict filtering.
baseline_members_table = cohort_table.loc[cohort_table["analysis"] == "uniform_labels"]
base_cohorts = set(baseline_members_table.loc[baseline_members_table["eligible"], "cohort"])
changes = []
for analysis, label in [
    ("preproc__uniform__counts_1k", "1,000-count reassignment"),
    ("preproc__uniform__counts_10k_strictQC", "stricter QC"),
]:
    selected = (cohort_table["analysis"] == analysis) & cohort_table["eligible"]
    excluded = sorted(base_cohorts - set(cohort_table.loc[selected, "cohort"]))
    if excluded:
        changes.append(label + " excludes " + ", ".join(excluded))
eligibility_note = ". ".join(changes) + "." if changes else "Baseline cohorts remain eligible."
figure.text(
    0.08,
    0.045,
    "Each panel uses its indicated expression scale.\n" + eligibility_note,
    fontsize=6.8,
    color=style.muted,
)
style.save_figure(figure, output_dir, "03_s16_preprocessing_comparison", panels="abc")

# S17 uses within-cohort patient codes. The cell-count panel includes all patients. other panels use baseline
# eligibility.
contrast_table = patient.read_table("preprocessing_patient_contrasts")
baseline_table = contrast_table.query(
    "cohort == 'GSE162631' and scenario == 'stored_X' and labels == 'uniform'"
)
baseline_table = baseline_table.set_index("patient")
eligible_table = patient.read_table("preprocessing_eligibility").query(
    "cohort == 'GSE162631' and labels == 'uniform'"
)
source_table = patient.read_table("GSE162631_source_sensitivity")
subjects = sorted(eligible_table["patient"].unique())
subject_codes = {}
for i, subject in enumerate(subjects, 1):
    is_public_code = subject.startswith("P") and subject[1:].isdigit()
    subject_codes[subject] = subject if is_public_code else f"P{i}"
eligible_subjects = [subject for subject in subjects if subject in baseline_table.index]
eligible_codes = [subject_codes[subject] for subject in eligible_subjects]
figure, axes = plt.subplots(1, 3, figsize=(7.1, 3.45))
figure.subplots_adjust(wspace=0.60, left=0.075, right=0.975, bottom=0.35, top=0.85)
scenarios = [
    ("stored_X", "Submitted", style.teal),
    ("counts_1k", "1,000 + reassignment", style.coral),
    ("counts_10k_strictQC", "Stricter QC", style.gray_dark),
]
for scenario, label, color in scenarios:
    scenario_table = eligible_table.loc[eligible_table["scenario"] == scenario].set_index("patient")
    axes[0].plot(
        range(len(subjects)),
        scenario_table.loc[subjects, "n_ac"],
        "o-",
        label=label,
        color=color,
        lw=1,
        ms=3.2,
    )
axes[0].axhline(20, color=style.gray_dark, ls="--", lw=0.65)
axes[0].set_xticks(range(len(subjects)), [subject_codes[subject] for subject in subjects])
axes[0].set_ylabel("AC-like cells")
axes[0].set_ylim(0, max(55, eligible_table["n_ac"].max() + 1))

# Connect each patient's two states and label them with detected and total AC-like cell counts.
for i, subject in enumerate(eligible_subjects):
    row = baseline_table.loc[subject]
    ac_percentage = row.ac_detected_fraction * 100
    other_percentage = row.other_detected_fraction * 100
    axes[1].plot([i - 0.1, i + 0.1], [ac_percentage, other_percentage], color=style.grid, lw=0.8)
    axes[1].scatter(
        i - 0.1, ac_percentage, s=18, color=style.state_colors["AC-like"], label="AC-like" if i == 0 else None
    )
    axes[1].scatter(
        i + 0.1,
        other_percentage,
        s=18,
        color=style.state_colors["Other malignant"],
        label="Other malignant" if i == 0 else None,
    )
    fraction = f"{round(row.ac_detected_fraction * row.n_ac)}/{int(row.n_ac)}"
    axes[1].text(i - 0.1, ac_percentage + 0.7, fraction, ha="center", fontsize=6.8)
axes[1].set_xlim(-0.4, len(eligible_subjects) - 0.6)
axes[1].set_xticks(range(len(eligible_codes)), eligible_codes)
axes[1].set_ylabel("TGFB2-detected cells (%)")
axes[1].set_ylim(-0.8, 15)

sources = [
    ("stored_X", "Submitted", style.teal),
    ("source_full_gene_10k", "Source counts", style.coral),
    ("source_original_tumor_concordant", "Concordant labels", style.gray_dark),
]
for j, (scenario, label, color) in enumerate(sources):
    scenario_table = baseline_table
    if scenario != "stored_X":
        scenario_table = source_table.loc[source_table["scenario"] == scenario].set_index("patient")
    values = scenario_table["difference"].reindex(eligible_subjects)
    axes[2].plot(
        np.arange(len(eligible_subjects)) + (j - 1) * 0.10, values, "o", label=label, color=color, ms=3.8
    )
axes[2].axhline(0, color=style.gray_dark, ls="--", lw=0.65)
axes[2].set_xticks(range(len(eligible_codes)), eligible_codes)
axes[2].set_ylabel("AC-like minus other\n(log1p / 10,000)")
axes[2].set_ylim(-0.065, 0.13)
for axis, title, letter in zip(axes, ["AC-like cell counts", "TGFB2 detection", "Source-count audit"], "abc"):
    axis.legend(loc="upper left", bbox_to_anchor=(0, -0.15), borderaxespad=0, fontsize=6.5)
    axis.set_title(title, loc="left")
    style.open_axes(axis)
    style.panel_label(axis, letter)
source_values = source_table.loc[source_table["scenario"] == "source_full_gene_10k", "difference"]
baseline_counts_table = eligible_table.loc[eligible_table["scenario"] == "stored_X"].set_index("patient")
small_groups = []
for subject in subjects:
    if subject not in baseline_table.index:
        n_ac = int(baseline_counts_table.loc[subject, "n_ac"])
        small_groups.append(f"{subject_codes[subject]} has {n_ac} AC-like cells")
small_group_note = " (" + ", ".join(small_groups) + ")" if small_groups else ""
concordant = source_table.loc[source_table["scenario"] == "source_original_tumor_concordant", "difference"]
caption = (
    "Patient codes are local to GSE162631. "
    f"Baseline: {len(eligible_subjects)} of {len(subjects)} patients{small_group_note}.\n"
    f"Source counts: {(source_values > 0).sum()} positive, {(source_values < 0).sum()} negative. "
    f"Concordant labels: {(concordant > 0).sum()} positive, {(concordant < 0).sum()} negative."
)
figure.text(0.075, 0.055, caption, fontsize=6.8, color=style.muted)
style.save_figure(figure, output_dir, "04_s17_source_count_audit", panels="abc")
