from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher

from mcp_book.models import PageText, TocEntry

NON_LETTERS = re.compile(r"[\W\d_]+")
MINIMUM_KEY_LETTERS = 3
MINIMUM_REPEATS = 3
SAME_LINE_RATIO = 0.85


@dataclass
class PageParts:
    page: PageText
    header: list[str] = field(default_factory=list)
    footer: list[str] = field(default_factory=list)


def running_key(line: str) -> str:
    return NON_LETTERS.sub("", line.casefold())


def join_parts(parts: PageParts, keep: set[str] | None = None) -> PageText:
    def kept(lines: list[str]) -> list[str]:
        return [line for line in lines if keep is None or line in keep]

    paragraphs = ["\n".join(kept(parts.header)), parts.page.text, "\n".join(kept(parts.footer))]
    return replace(parts.page, text="\n\n".join(paragraph for paragraph in paragraphs if paragraph))


def merge_variants(pages_by_key: dict[str, set[int]]) -> dict[str, str]:
    canonical: dict[str, str] = {}
    heads: list[str] = []
    for key in sorted(pages_by_key, key=lambda key: (-len(pages_by_key[key]), key)):
        match = next(
            (head for head in heads if SequenceMatcher(None, key, head).ratio() >= SAME_LINE_RATIO),
            None,
        )
        canonical[key] = match or key
        if match is None and len(pages_by_key[key]) > 1:
            heads.append(key)
    return canonical


def chapter_title(variants: Counter[str]) -> str:
    def broken_words(variant: str) -> int:
        return sum(len(word) == 1 for word in variant.split())

    title = " ".join(
        min(variants, key=lambda variant: (broken_words(variant), -variants[variant])).split()
    )
    upper = sum(character.isupper() for character in title)
    lower = sum(character.islower() for character in title)
    return title.capitalize() if upper > lower else title


def remove_running_lines(parts: list[PageParts]) -> tuple[list[PageText], list[TocEntry]]:
    pages_by_key: dict[str, set[int]] = defaultdict(set)
    for part in parts:
        for line in part.header + part.footer:
            key = running_key(line)
            if len(key) >= MINIMUM_KEY_LETTERS:
                pages_by_key[key].add(part.page.page)

    canonical = merge_variants(pages_by_key)
    merged: dict[str, set[int]] = defaultdict(set)
    variants: dict[str, Counter[str]] = defaultdict(Counter)
    for key, pages in pages_by_key.items():
        merged[canonical[key]] |= pages
    repeated = {key for key, pages in merged.items() if len(pages) >= MINIMUM_REPEATS}

    cleaned = []
    for part in parts:
        keep = set()
        for line in part.header + part.footer:
            key = running_key(line)
            label = running_key(part.page.label or "")
            if len(key) < MINIMUM_KEY_LETTERS or key == label:
                continue
            if canonical[key] in repeated:
                variants[canonical[key]][line] += 1
                continue
            keep.add(line)
        cleaned.append(join_parts(part, keep))

    spans = {key: (min(merged[key]), max(merged[key])) for key in repeated}
    book_level: set[str] = set()
    for key in sorted(spans, key=lambda key: spans[key][0] - spans[key][1]):
        start, end = spans[key]
        overlapping = [
            other
            for other, (other_start, other_end) in spans.items()
            if other != key
            and other not in book_level
            and other_start <= end
            and start <= other_end
        ]
        if len(overlapping) >= 2:
            book_level.add(key)

    chapter_keys = sorted(repeated - book_level, key=spans.get)
    if len(chapter_keys) < 2:
        return cleaned, []
    chapters = [
        TocEntry(level=1, title=chapter_title(variants[key]), page=spans[key][0])
        for key in chapter_keys
    ]
    return cleaned, chapters
