from mcp_book.chunking import approximate_tokens, chunk_book
from mcp_book.config import ChunkingConfig
from mcp_book.models import PageText, TocEntry

SMALL = ChunkingConfig(target_tokens=40, max_tokens=60, overlap_tokens=15)


def sentences(count: int, word: str = "parola") -> str:
    return " ".join(f"Frase {index} con {word} di prova." for index in range(count))


def test_sections_follow_toc_inside_a_page() -> None:
    pages = [
        PageText(page=1, text="Le salse\n\nMaionese\n\nTesto sulla maionese."),
        PageText(page=2, text="Fine maionese.\n\nBesciamella\n\nTesto sulla besciamella."),
    ]
    toc = [TocEntry(1, "Le salse", 1), TocEntry(2, "Maionese", 1), TocEntry(2, "Besciamella", 2)]

    chunks = chunk_book(pages, toc, ChunkingConfig())

    assert [(c.chapter, c.section, c.page_start, c.page_end) for c in chunks] == [
        ("Le salse", "Maionese", 1, 2),
        ("Le salse", "Besciamella", 2, 2),
    ]
    assert chunks[0].text.startswith("Le salse\n\nMaionese")
    assert "Fine maionese." in chunks[0].text
    assert "Fine maionese." not in chunks[1].text


def test_unmatched_heading_applies_from_page_start() -> None:
    pages = [PageText(page=1, text="Intro."), PageText(page=2, text="Testo senza titolo.")]
    toc = [TocEntry(1, "Primo", 1), TocEntry(1, "Secondo capitolo", 2)]

    chunks = chunk_book(pages, toc, ChunkingConfig())

    assert [(c.chapter, c.page_start) for c in chunks] == [("Primo", 1), ("Secondo capitolo", 2)]


def test_nested_sections_are_joined() -> None:
    pages = [PageText(page=1, text="Testo.")]
    toc = [TocEntry(1, "Carni", 1), TocEntry(2, "Manzo", 1), TocEntry(3, "Brasati", 1)]

    [chunk] = chunk_book(pages, toc, ChunkingConfig())

    assert (chunk.chapter, chunk.section) == ("Carni", "Manzo › Brasati")


def test_chunks_respect_size_and_overlap() -> None:
    pages = [PageText(page=1, text=sentences(30)), PageText(page=2, text=sentences(30, "pagina"))]

    chunks = chunk_book(pages, [], SMALL)

    assert len(chunks) > 5
    assert all(c.token_count <= SMALL.max_tokens for c in chunks)
    assert all(approximate_tokens(c.text) <= SMALL.max_tokens + 5 for c in chunks)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        last_sentence = previous.text.split(". ")[-1]
        assert last_sentence in current.text
    assert any(c.page_start == 1 and c.page_end == 2 for c in chunks)
    assert [c.sequence for c in chunks] == list(range(len(chunks)))


def test_oversized_sentence_is_split() -> None:
    pages = [PageText(page=1, text="parola " * 400)]

    chunks = chunk_book(pages, [], SMALL)

    assert all(c.token_count <= SMALL.max_tokens for c in chunks)
    assert sum(c.text.count("parola") for c in chunks) >= 400


def test_no_toc_and_empty_pages() -> None:
    pages = [PageText(page=1, text=""), PageText(page=2, text="Solo testo.")]

    [chunk] = chunk_book(pages, [], ChunkingConfig())

    assert (chunk.chapter, chunk.section, chunk.page_start) == (None, None, 2)


def test_back_of_book_index_pages_are_not_chunked() -> None:
    entries = [
        f"voce {letter}, {index}·{index * 3}-{index * 3 + 2}"
        for index, letter in enumerate("abcdefghijkl", 1)
    ]
    pages = [
        PageText(page=1, text="La maionese, 3 tuorli, 1 cucchiaio di senape, 500 g di olio."),
        PageText(page=2, text="aceto\n" + "\n".join(entries)),
    ]

    chunks = chunk_book(pages, [], ChunkingConfig())

    assert [(chunk.page_start, chunk.page_end) for chunk in chunks] == [(1, 1)]


def test_custom_token_counter_is_used() -> None:
    pages = [PageText(page=1, text=sentences(10))]

    chunks = chunk_book(pages, [], SMALL, count_tokens=lambda text: len(text.split()))

    assert all(c.token_count == len(c.text.split()) for c in chunks)
