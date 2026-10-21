from pathlib import Path

import pytest

from mcp_book.config import load_config


def test_defaults_resolve_relative_to_config_file(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text('[library]\npdf_dir = "books"\n[chunking]\ntarget_tokens = 500\n')

    config = load_config(config_file)

    assert config.library.pdf_dir == tmp_path / "books"
    assert config.library.index_dir == tmp_path / "data"
    assert config.database_path == tmp_path / "data" / "library.sqlite"
    assert config.chunking.target_tokens == 500
    assert config.embedding.model == "BAAI/bge-m3"
    assert config.embedding.device == "auto"


def test_repository_config_is_valid() -> None:
    config = load_config(Path(__file__).parents[1] / "config.toml")
    assert config.search.top_k == 8


@pytest.mark.parametrize(
    "content",
    [
        "[chunking]\ntarget_tokens = 900\nmax_tokens = 800\n",
        "[chunking]\noverlap_tokens = 700\n",
        '[embedding]\nbackend = "openai"\n',
        "[library]\ntypo = 1\n",
        "[unknown]\nkey = 1\n",
    ],
)
def test_invalid_config_is_rejected(tmp_path: Path, content: str) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(content)
    with pytest.raises(ValueError):
        load_config(config_file)


def test_missing_explicit_config_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.toml")
