from mcp_book.citation import format_citation


def test_full_citation() -> None:
    assert format_citation("La cucina", "Le salse", 12) == "La cucina, Le salse, p. 12"


def test_page_range_and_missing_chapter() -> None:
    assert format_citation("La cucina", None, 12, 15) == "La cucina, pp. 12–15"
    assert format_citation("La cucina", "Brodi", "xii", "xii") == "La cucina, Brodi, p. xii"


def test_title_only() -> None:
    assert format_citation("La cucina") == "La cucina"
