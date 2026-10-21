from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from sample_pdf import build_sample_pdf

requires_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="Tesseract is not installed"
)


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("library")
    return build_sample_pdf(folder / "manuale-cucina.pdf")
