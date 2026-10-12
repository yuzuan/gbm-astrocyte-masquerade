#!/usr/bin/env python3
"""为公共数据投稿脚本提供可迁移的项目、输出和输入路径。"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """优先使用环境变量，否则取代码目录的上一级。"""
    configured = os.environ.get("GLIOMA_VALIDATION_ROOT")
    return Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[1]


def submission_package(root: Path) -> Path:
    """兼容源项目目录和冻结后的独立投稿包。"""
    configured = os.environ.get("PUBLIC_ONLY_PACKAGE")
    if configured:
        return Path(configured).expanduser().resolve()
    if (root / "manuscript_public_only.md").is_file():
        return root
    if (root / "data").is_dir() and (root / "source_data").is_dir():
        return root
    return root / "claude_outputs" / "public_only_submission"


def public_input_root(root: Path) -> Path:
    """返回仓库相对公共输入目录，可由环境变量覆盖。"""
    configured = os.environ.get("PUBLIC_INPUT_ROOT")
    return Path(configured).expanduser().resolve() if configured else root / "public_inputs"
