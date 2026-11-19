#!/usr/bin/env python3
# 目的：仅用公开的重编码精简表达对象重建八张患者主结果表，逐表与同包的 data/ 参考结果比较。
# 输入：minimal_cell_data/minimal_cell_data_manifest.csv 及其四个 .h5ad（X、患者、区域、原状态标签）；
# data/ 下八张参考 CSV，原名与新名的逐一对应见 main 中的 result_names。
# 输出：14_患者主结果复现/ 下 01_患者状态表达.csv、02_患者状态组成.csv、03_患者配体差值.csv、
# 04_跨队列配体合并.csv、05_患者配体受体共可用指数.csv、06_区域共可用指数检验.csv、
# 07_患者AC样比例.csv、08_区域AC样比例检验.csv、09_复现核对.json。
# 运行方式：在 original_analysis 目录运行 python 14_复现患者主结果.py。数据从脚本所在目录读取。

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import numpy as np
import pandas as pd


def main() -> None:
    """以发布文件为参考，核对列、患者标识、数值与嵌套记录，记录每张表的实测结果。无返回值。"""
    package = Path(__file__).resolve().parent
    code_directory = str(package / "code")
    if code_directory not in sys.path:
        sys.path.insert(0, code_directory)
    primary = import_module("01_公共队列患者表达分析")
    paths = import_module("00a_项目路径")

    # 同一份 snGBM 精简文件供两个队列读取。历史 manifest 记录重编码前的字节数，只核对可比较的字段。
    directory = package / "minimal_cell_data"
    manifest_table = pd.read_csv(directory / "minimal_cell_data_manifest.csv")
    expected_files = {
        "GSE211376_minimal.h5ad",
        "scGBM_minimal.h5ad",
        "snGBM_minimal.h5ad",
        "regional_GSE274987_minimal.h5ad",
    }
    if manifest_table["file"].duplicated().any() or set(manifest_table["file"]) != expected_files:
        raise ValueError("manifest 的文件名与发表的四份精简输入不符")
    for row in manifest_table.itertuples(index=False):
        source = ad.read_h5ad(directory / row.file, backed="r")
        try:
            if source.shape != (row.n_cells, row.n_genes) or list(source.obs.columns) != row.obs_columns.split(";"):
                raise ValueError(f"{row.file} 的维度或元数据列与 manifest 不一致")
            if not source.obs_names.is_unique or not set(primary.primary_genes).issubset(source.var_names):
                raise ValueError(f"{row.file} 的细胞名重复或主分析基因不全")
        finally:
            source.file.close()

    expression_table_total, composition_table_total = primary.read_primary_cohorts(directory)
    tables = primary.primary_tables(expression_table_total, composition_table_total)
    result_names = {
        "patient_state_expression": "01_患者状态表达.csv",
        "patient_state_composition": "02_患者状态组成.csv",
        "ac_vs_other_ligand_contrasts": "03_患者配体差值.csv",
        "meta_ligand_results": "04_跨队列配体合并.csv",
        "patient_communication_scores": "05_患者配体受体共可用指数.csv",
        "regional_communication_tests": "06_区域共可用指数检验.csv",
        "patient_ac_fraction": "07_患者AC样比例.csv",
        "regional_ac_fraction_tests": "08_区域AC样比例检验.csv",
    }
    if tables.keys() != result_names.keys():
        raise ValueError("重建的结果集与八张发表参考表不一致")

    # 每张表先经写盘回读，再与独立的发表表比对。有效的差异也留下布尔结果和具体字段供复核。
    output = package / "14_患者主结果复现"
    output.mkdir(parents=True, exist_ok=True)
    checks = {}
    differences = []
    with TemporaryDirectory(dir=output) as scratch:
        staging = Path(scratch)
        for name, filename in result_names.items():
            keys = primary.primary_row_columns[name]
            paths.write_table(tables[name], staging / filename, keys)
            actual_table = pd.read_csv(staging / filename, float_precision="round_trip")
            reference_table = pd.read_csv(package / "data" / f"{name}.csv", float_precision="round_trip")
            try:
                compare_table(actual_table, reference_table, keys, name)
                checks[name] = True
            except (AssertionError, ValueError, KeyError, TypeError) as error:
                checks[name] = False
                differences.append(f"{name}：{error}")

        paths.write_json(checks, staging / "09_复现核对.json")
        for filename in [*result_names.values(), "09_复现核对.json"]:
            (staging / filename).replace(output / filename)
    print(f"患者主结果对照通过 {sum(checks.values())} 张，失败 {len(differences)} 张。")
    if differences:
        raise ValueError("\n".join(differences))


def compare_table(actual_table: pd.DataFrame, reference_table: pd.DataFrame, keys: list[str], name: str) -> None:
    """按真实行标识对齐，比较所有发布列；重复标识、缺列、形状或任一字段不符即报错。无返回值。"""
    if list(actual_table.columns) != list(reference_table.columns):
        raise ValueError(f"{name} 列名或列序发生变化")
    actual_table = actual_table.set_index(keys, verify_integrity=True).sort_index()
    reference_table = reference_table.set_index(keys, verify_integrity=True).sort_index()
    pd.testing.assert_index_equal(actual_table.index, reference_table.index, obj=f"{name} 行标识")

    # 浮点沿用原脚本 rtol=1e-5、atol=1e-7；整数、布尔、文字和缺失位置分别比较。
    nested_columns = {"patient_values", "loo_results", "cohort_effects", "patient_differences"}
    for column in reference_table:
        if column in nested_columns:
            for row_id in reference_table.index:
                actual = json.loads(actual_table.at[row_id, column])
                expected = json.loads(reference_table.at[row_id, column])
                compare_records(actual, expected, column, f"{name}/{row_id}/{column}")
            continue
        compare_column(actual_table[column], reference_table[column], f"{name}/{column}")


def compare_records(actual: list, expected: list, column: str, name: str) -> None:
    """有患者或队列标识的记录按名称比较；无编号的区域差值数组按发表顺序逐个比。无返回值。"""
    if len(actual) != len(expected):
        raise ValueError(f"{name} 嵌套记录数不同")
    if column == "patient_differences":
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7, equal_nan=True, err_msg=name)
    elif expected:
        identifier = {"patient_values": "patient", "loo_results": "omitted", "cohort_effects": "cohort"}[column]
        expected_fields = {row[identifier]: set(row) for row in expected}
        actual_fields = {row[identifier]: set(row) for row in actual}
        if actual_fields != expected_fields:
            raise ValueError(f"{name} 的标识或逐条记录字段不同")
        actual_table = pd.DataFrame(actual).set_index(identifier, verify_integrity=True).sort_index()
        expected_table = pd.DataFrame(expected).set_index(identifier, verify_integrity=True).sort_index()
        pd.testing.assert_index_equal(actual_table.index, expected_table.index, obj=name)
        if set(actual_table.columns) != set(expected_table.columns):
            raise ValueError(f"{name} 的嵌套字段不同")
        for field in expected_table:
            compare_column(actual_table[field], expected_table[field], f"{name}/{field}")


def compare_column(actual: pd.Series, expected: pd.Series, name: str) -> None:
    """使用与参考字段对应的比较方式，防止缺失、计数或文字差异在数值转换时丢失。无返回值。"""
    if pd.api.types.is_float_dtype(expected):
        np.testing.assert_allclose(
            actual.to_numpy(float), expected.to_numpy(float), rtol=1e-5, atol=1e-7, equal_nan=True, err_msg=name
        )
    else:
        pd.testing.assert_series_equal(actual, expected, check_exact=True, obj=name)


if __name__ == "__main__":
    main()
