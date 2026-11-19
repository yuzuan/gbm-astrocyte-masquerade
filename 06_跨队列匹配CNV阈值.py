# 目的：用各自免疫参照的第 95 百分位比较自有 GBM 与 snGBM 星形细胞的 +7/−10 阈上比例。
# 输入：data/infercnv_cell_scores.csv（首列条码、cell_type、chr7、chr10、cnv_pos），
#       data/cell_metadata.csv（首列条码、condition），以及
#       03_result/01_snGBM免疫参照CNV验证/01_snGBM免疫参照CNV结果.json 中的 snGBM_independent。
#       旧运行可读取 data/intermediate 或根目录的 noncircular_cnv_validation.json。
# 输出：03_result/06_跨队列匹配CNV阈值/01_跨队列匹配CNV阈值结果.json。
# 运行方式：在仓库根目录运行 python 06_跨队列匹配CNV阈值.py，沿用原 Python 阈值计算。

import json
from importlib import import_module
from pathlib import Path

common = import_module("00a_基因表达与聚类")

# 自有队列从逐细胞评分重算。三个参照组覆盖全部组织；主分母只取 condition=GBM 的 Astrocyte。
own = common.inhouse_cnv_summary(Path("data/infercnv_cell_scores.csv"), Path("data/cell_metadata.csv"))

# snGBM 使用 01 对其自身免疫参照计算的结果。旧冻结档只有阳性比例和分母，阈值可留空。
sn_path = common.result_or_frozen(
    Path("03_result/01_snGBM免疫参照CNV验证/01_snGBM免疫参照CNV结果.json"), "noncircular_cnv_validation.json"
)
sn_result = json.loads(sn_path.read_text(encoding="utf-8"))
sn_group = sn_result["snGBM_independent"]
sn_count = sn_group["n"]
sn_percent = sn_group["strict_CNVpos_pct"]
if type(sn_count) is not int or sn_count <= 0 or type(sn_percent) not in (int, float) or not 0 <= sn_percent <= 100:
    raise ValueError("snGBM 输入未给出有效的星形细胞分母和严格阈上百分比。")

# 两队列匹配的是参照分位数规则。自有宽松结果来自上游 cnv_pos；参照 FPR 为各类型的阈上比例。
# 历史 peritumoral 字段按原计算单元取自有 Normal 星形细胞，该列没有更细的取材位置。
result = {
    "own_strict_pct": own["strict_matched_pct"],
    "own_loose_pct": own["loose_cnv_pos_pct"],
    "own_n": own["n"],
    "snGBM_strict_pct": sn_percent,
    "snGBM_n": sn_count,
    "own_ref_FPR_strict": own["strict_ref_FPR"],
    "peritumoral_astro_strict_pct": own["peritumoral_astro_strict_pct"],
    "own_strict_threshold": own["strict_threshold"],
    "snGBM_strict_threshold": sn_group.get("strict_threshold"),
    "own_ref_n": own["reference_n"],
    "own_normal_astro_n": own["normal_astro_n"],
    "sn_summary_source": sn_path.as_posix(),
    "note": (
        f"In-house GBM astrocytes (n={own['n']}): {own['loose_cnv_pos_pct']:.1f}% positive by upstream cnv_pos, "
        f"{own['strict_matched_pct']:.1f}% above the in-house immune 95th percentile. "
        f"snGBM Leiden cluster 0 (n={sn_count}): {sn_percent:.1f}% above the snGBM immune 95th percentile."
    ),
}

# 核对参照分母和百分比一一对应；无该类细胞时应写 null。回读完整结果后发布。
for label, count in result["own_ref_n"].items():
    percent = result["own_ref_FPR_strict"][label]
    if (count == 0) != (percent is None):
        raise ValueError(f"自有队列 {label} 的分母与假阳性率缺失状态不符。")
common.write_checked_json(Path("03_result/06_跨队列匹配CNV阈值/01_跨队列匹配CNV阈值结果.json"), result)
print(f"免疫参照阈上比例：自有 GBM {result['own_strict_pct']}%（n={own['n']}），snGBM {sn_percent}%（n={sn_count}）。")
