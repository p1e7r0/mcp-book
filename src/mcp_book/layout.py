from __future__ import annotations

import contextlib
import json
import logging
import sys
from dataclasses import dataclass, field
from itertools import pairwise

import pymupdf

logger = logging.getLogger(__name__)

LAYOUT_BATCH_PAGES = 50
BOX_TOLERANCE = 2.0
PARAGRAPH_GAP = 0.8
UNSPLIT_BOX_CLASSES = {"table"}


@dataclass
class Line:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    kind: str = "text"


@dataclass
class Region:
    lines: list[Line]
    splittable: bool = True
    columns: list[tuple[float, float]] = field(default_factory=list)


def text_lines(
    page: pymupdf.Page, flags: int, textpage: pymupdf.TextPage | None = None
) -> list[list[Line]]:
    groups = []
    for block in page.get_text("dict", flags=flags, textpage=textpage)["blocks"]:
        if block.get("type") != 0:
            continue
        lines = [
            Line(*line["bbox"], "".join(span["text"] for span in line["spans"]).rstrip())
            for line in block["lines"]
        ]
        lines = [line for line in lines if line.text.strip()]
        if lines:
            groups.append(lines)
    return groups


def layout_boxes(document: pymupdf.Document) -> dict[int, list[Box]]:
    boxes: dict[int, list[Box]] = {}
    for start in range(0, document.page_count, LAYOUT_BATCH_PAGES):
        pages = list(range(start, min(start + LAYOUT_BATCH_PAGES, document.page_count)))
        try:
            with contextlib.redirect_stdout(sys.stderr):
                data = json.loads(pymupdf.pymupdf4llm.to_json(document, pages=pages, use_ocr=False))
        except Exception as error:
            logger.warning(
                "Layout analysis failed on pages %d-%d (%s): keeping the PDF block order",
                pages[0] + 1,
                pages[-1] + 1,
                error,
            )
            continue
        for page in data["pages"]:
            boxes[page["page_number"]] = [
                Box(box["x0"], box["y0"], box["x1"], box["y1"], box["boxclass"])
                for box in page["boxes"]
            ]
    return boxes


def _inside(line: Line, box: Box) -> bool:
    center_x, center_y = (line.x0 + line.x1) / 2, (line.y0 + line.y1) / 2
    return (
        box.x0 - BOX_TOLERANCE <= center_x <= box.x1 + BOX_TOLERANCE
        and box.y0 - BOX_TOLERANCE <= center_y <= box.y1 + BOX_TOLERANCE
    )


def group_by_boxes(groups: list[list[Line]], boxes: list[Box]) -> list[Region]:
    assigned: list[list[Line]] = [[] for _ in boxes]
    leftover = []
    for group in groups:
        rest = []
        for line in group:
            index = next((index for index, box in enumerate(boxes) if _inside(line, box)), None)
            if index is None:
                rest.append(line)
            else:
                assigned[index].append(line)
        if rest:
            leftover.append(rest)
    regions = [
        Region(lines, splittable=box.kind not in UNSPLIT_BOX_CLASSES)
        for box, lines in zip(boxes, assigned, strict=True)
        if lines
    ]
    return regions + [Region(lines) for lines in leftover]


def side_by_side_columns(lines: list[Line]) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    for line in sorted(lines, key=lambda line: line.x0):
        if spans and line.x0 <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], line.x1))
        else:
            spans.append((line.x0, line.x1))
    if len(spans) < 2:
        return []
    heights = [
        (
            min(line.y0 for line in lines if start <= line.x0 <= end),
            max(line.y1 for line in lines if start <= line.x0 <= end),
        )
        for start, end in spans
    ]
    stacked = any(
        first[1] <= second[0] or second[1] <= first[0] for first, second in pairwise(heights)
    )
    return [] if stacked else spans


def _same_columns(first: list[tuple[float, float]], second: list[tuple[float, float]]) -> bool:
    return len(first) == len(second) and all(
        a_start <= b_end and b_start <= a_end
        for (a_start, a_end), (b_start, b_end) in zip(first, second, strict=True)
    )


def paragraphs(lines: list[Line]) -> list[list[Line]]:
    result: list[list[Line]] = []
    for line in lines:
        if result:
            previous = result[-1][-1]
            gap = line.y0 - previous.y1
            if gap <= PARAGRAPH_GAP * (previous.y1 - previous.y0) and line.y0 >= previous.y0:
                result[-1].append(line)
                continue
        result.append([line])
    return result


def reading_order(groups: list[list[Line]], boxes: list[Box] | None = None) -> list[list[Line]]:
    regions = group_by_boxes(groups, boxes) if boxes else [Region(group) for group in groups]
    merged: list[Region] = []
    for region in regions:
        region.columns = side_by_side_columns(region.lines) if region.splittable else []
        previous = merged[-1] if merged else None
        if previous and region.columns and _same_columns(previous.columns, region.columns):
            previous.lines += region.lines
            previous.columns = side_by_side_columns(previous.lines) or previous.columns
            continue
        merged.append(region)

    ordered: list[list[Line]] = []
    for region in merged:
        if not region.columns:
            ordered += paragraphs(region.lines)
            continue
        for start, end in region.columns:
            column = [line for line in region.lines if start <= line.x0 <= end]
            ordered += paragraphs(sorted(column, key=lambda line: line.y0))
    return ordered
