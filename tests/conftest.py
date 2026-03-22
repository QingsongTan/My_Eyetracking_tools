from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pytest

TMP_ROOT = Path("test_tmp_root").resolve()


@pytest.fixture
def tmp_path() -> Path:
    """在工作区内创建可写临时目录，避免系统 tmp 权限干扰测试。"""
    TMP_ROOT.mkdir(exist_ok=True)
    path = (TMP_ROOT / f"pytest-case-{uuid4().hex}").resolve()
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """清理工作区内的测试临时目录根路径。"""
    del session, exitstatus
    shutil.rmtree(TMP_ROOT, ignore_errors=True)
