from __future__ import annotations


def format_pages(start: str | int, end: str | int | None = None) -> str:
    if end is None or str(end) == str(start):
        return f"p. {start}"
    return f"pp. {start}–{end}"


def format_citation(
    title: str,
    chapter: str | None = None,
    page_start: str | int | None = None,
    page_end: str | int | None = None,
) -> str:
    """Build "Title, Chapter, p. N"; missing parts are left out."""
    parts = [title]
    if chapter:
        parts.append(chapter)
    if page_start is not None:
        parts.append(format_pages(page_start, page_end))
    return ", ".join(parts)
