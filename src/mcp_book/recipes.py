from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from mcp_book.layout import Line

HEADER_NAMES = ("ingredient", "quantity", "scaling", "procedure")
HEADER_MATCH = 0.75
COLUMN_MARGIN = 3.0
CONTINUATION_GAP = 1.2
PROCEDURE_GAP = 2.5
ROW_OVERLAP = 0.3
UNIT = (
    r"(?:k?g|mg|m?l|cm|mm|oz|lbs?|tsp|tbsp|cups?|each|pieces?|whole|sheets?|leaves|sprigs?"
    r"|cloves?|drops?|slices?|large|small|medium)"
)
QUANTITY = re.compile(
    rf"^(?:\d[\d.,/]*\s?(?:{UNIT}\b|\(|$)|[\dlIOSJ][\dlIOSJ.,]*\s?k?g\b"
    r"|to taste|as needed|optional|pinch|dash|a few)",
    re.IGNORECASE,
)
PERCENT = re.compile(r"^\d[\d.,]*\s?%")
NON_LETTERS = re.compile(r"[^a-z]")


@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class CellLine:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 1.0)


@dataclass
class Row:
    top: float
    cells: dict[str, list[CellLine]] = field(default_factory=dict)


@dataclass
class Recipe:
    columns: list[str]
    rows: list[list[str]] = field(default_factory=list)
    procedure: list[CellLine] = field(default_factory=list)
    used: list[CellLine] = field(default_factory=list)

    @property
    def top(self) -> float:
        return min(line.y0 for line in self.used)


def header_name(text: str) -> str | None:
    key = NON_LETTERS.sub("", text.casefold())
    if len(key) < 5 or not any(character.isupper() for character in text):
        return None
    best = max(HEADER_NAMES, key=lambda name: SequenceMatcher(None, key, name).ratio())
    return best if SequenceMatcher(None, key, best).ratio() >= HEADER_MATCH else None


def cell_lines(words: list[Word]) -> list[CellLine]:
    lines: list[list[Word]] = []
    for word in sorted(words, key=lambda word: (word.y0, word.x0)):
        if lines and lines[-1][0].y0 <= word.center_y <= lines[-1][0].y1:
            lines[-1].append(word)
        else:
            lines.append([word])
    return [
        CellLine(
            min(word.x0 for word in line),
            min(word.y0 for word in line),
            max(word.x1 for word in line),
            max(word.y1 for word in line),
            " ".join(word.text for word in sorted(line, key=lambda word: word.x0)),
        )
        for line in lines
    ]


def _overlap(first: CellLine, second: CellLine) -> float:
    return max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))


def _is_label(line: CellLine) -> bool:
    letters = [character for character in line.text if character.isalpha()]
    upper = sum(character.isupper() for character in letters)
    return len(letters) >= 4 and upper > 0.8 * len(letters)


def find_headers(words: list[Word]) -> list[dict[str, Word]]:
    named = [(header_name(word.text), word) for word in words]
    headers = []
    for name, anchor in named:
        if name != "ingredient":
            continue
        row = {
            other_name: word
            for other_name, word in named
            if other_name and abs(word.center_y - anchor.center_y) < anchor.y1 - anchor.y0
        }
        if "quantity" in row or "scaling" in row:
            headers.append(row)
    return sorted(headers, key=lambda header: header["ingredient"].y0)


def _anchors(cells: dict[str, list[CellLine]], first_column: str) -> list[tuple[str, CellLine]]:
    anchors: list[tuple[str, CellLine]] = []
    for name, pattern in (("quantity", QUANTITY), ("scaling", PERCENT)):
        for line in cells.get(name, []):
            has_ingredient = any(_overlap(line, other) for other in cells[first_column])
            if (
                pattern.match(line.text)
                and (has_ingredient or name == "scaling")
                and not any(_overlap(line, anchor) for _, anchor in anchors)
            ):
                anchors.append((name, line))
    return sorted(anchors, key=lambda anchor: anchor[1].y0)


def _spans_columns(line: CellLine, others: list[CellLine], anchors: list[CellLine]) -> bool:
    return any(
        _overlap(line, other) >= 0.5 * line.height and other not in anchors for other in others
    )


def _build_rows(
    cells: dict[str, list[CellLine]], columns: list[str], named_anchors: list[tuple[str, CellLine]]
) -> list[Row]:
    rows = [Row(anchor.y0, {name: [anchor]}) for name, anchor in named_anchors]
    anchors = [anchor for _, anchor in named_anchors]
    others = [line for name in columns[1:] for line in cells.get(name, [])]
    for name in columns:
        for line in cells[name]:
            if line in anchors:
                continue
            overlaps = [_overlap(line, anchor) for anchor in anchors]
            best = max(range(len(anchors)), key=overlaps.__getitem__)
            if overlaps[best] >= ROW_OVERLAP * line.height:
                rows[best].cells.setdefault(name, []).append(line)
                continue
            if name == columns[0] and _is_label(line) and line.y0 < anchors[-1].y0:
                rows.append(Row(line.y0, {name: [line]}))
                continue
            earlier = [row for row in rows if row.top <= line.y0]
            if not earlier:
                continue
            row = max(earlier, key=lambda row: row.top)
            previous = row.cells.get(name) or [
                line for lines in row.cells.values() for line in lines
            ]
            end = max(previous_line.y1 for previous_line in previous)
            spanning = name == columns[0] and _spans_columns(line, others, anchors)
            if line.y0 - end <= CONTINUATION_GAP * line.height and not spanning:
                row.cells.setdefault(name, []).append(line)
    return sorted(rows, key=lambda row: row.top)


def _procedure(lines: list[CellLine]) -> list[CellLine]:
    kept: list[CellLine] = []
    for line in lines:
        if kept and line.y0 - kept[-1].y1 > PROCEDURE_GAP * line.height:
            break
        kept.append(line)
    return kept


def find_recipes(words: list[Word], page_width: float) -> list[Recipe]:
    recipes = []
    headers = find_headers(words)
    for index, header in enumerate(headers):
        names = [name for name in HEADER_NAMES if name in header]
        starts = [header[name].x0 - COLUMN_MARGIN for name in names]
        bounds = dict(zip(names, zip(starts, [*starts[1:], page_width], strict=True), strict=True))
        columns = [name for name in names if name != "procedure"]
        left, right = bounds[columns[0]][0], bounds[columns[-1]][1]
        header_words = list(header.values())
        top = max(word.y1 for word in header_words)
        limit = min(
            (
                min(word.y0 for word in other.values())
                for other in headers[index + 1 :]
                if left - 50 <= other["ingredient"].x0 < right
            ),
            default=float("inf"),
        )
        cells = {
            name: cell_lines(
                [
                    word
                    for word in words
                    if start <= word.center_x < end and top <= word.center_y < limit
                ]
            )
            for name, (start, end) in bounds.items()
        }
        anchors = _anchors(cells, columns[0])
        if not anchors:
            continue
        rows = _build_rows(cells, columns, anchors)
        recipe = Recipe(columns=columns)
        recipe.rows = [
            [" ".join(line.text for line in row.cells.get(name, [])) for name in columns]
            for row in rows
        ]
        recipe.procedure = _procedure(cells.get("procedure", []))
        recipe.used = [
            CellLine(word.x0, word.y0, word.x1, word.y1, word.text) for word in header_words
        ]
        recipe.used += [line for row in rows for lines in row.cells.values() for line in lines]
        recipe.used += recipe.procedure
        recipes.append(recipe)
    return recipes


def _covered(line: Line, cells: list[CellLine]) -> bool:
    height = max(line.y1 - line.y0, 1.0)
    covered = sum(
        max(0.0, min(line.x1, cell.x1) - max(line.x0, cell.x0))
        for cell in cells
        if min(line.y1, cell.y1) - max(line.y0, cell.y0) >= 0.5 * height
    )
    return covered >= 0.5 * max(line.x1 - line.x0, 1.0)


def remove_recipe_lines(groups: list[list[Line]], recipes: list[Recipe]) -> list[list[Line]]:
    used = [cell for recipe in recipes for cell in recipe.used]
    cleaned = [[line for line in group if not _covered(line, used)] for group in groups]
    return [group for group in cleaned if group]


def place_recipes(
    paragraphs: list[list[Line]], recipes: list[tuple[Recipe, str]]
) -> list[list[Line]]:
    placed = list(paragraphs)
    for recipe, text in sorted(recipes, key=lambda item: item[0].top):
        x0 = min(cell.x0 for cell in recipe.used)
        x1 = max(cell.x1 for cell in recipe.used)
        y1 = max(cell.y1 for cell in recipe.used)
        position = next(
            (
                index
                for index, paragraph in enumerate(placed)
                if paragraph[0].y0 >= recipe.top and paragraph[0].x0 < x1 and paragraph[0].x1 > x0
            ),
            len(placed),
        )
        placed.insert(position, [Line(x0, recipe.top, x1, y1, text)])
    return placed
