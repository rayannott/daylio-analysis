import pathlib
import datetime
import re
import warnings
from enum import Enum, auto
from difflib import SequenceMatcher
from dataclasses import dataclass
from collections import defaultdict


CLIPPINGS_TXT = pathlib.Path("data", "clippings.txt")


LOCATION_RANGE_RE = re.compile(r"location (\d+)-(\d+)")
LOCATION_RE = re.compile(r"location (\d+)")
PAGE_RE = re.compile(r"on page (\d+)")
DATETIME_RE = re.compile(r"Added on (.+)$")
TITLE_AUTHOR_RE = re.compile(r"([\w\s',-]+) \(([\w\s',-]+)\)")

RUSSIAN_ALPH = set(chr(i) for i in range(ord("а"), ord("я") + 1))


class ClipType(Enum):
    HIGHLIGHT = auto()
    NOTE = auto()
    BOOKMARK = auto()

    @classmethod
    def detect_type(cls, line: str) -> "ClipType":
        for field in cls:
            if field.name.lower() in line.lower():
                return field
        raise ValueError(line)


def extract_location(line: str) -> tuple[int] | tuple[int, int]:
    if m := LOCATION_RANGE_RE.search(line):
        return int(m.group(1)), int(m.group(2))
    if m := LOCATION_RE.search(line):
        return (int(m.group(1)),)
    raise ValueError(line)


def extract_datetime_page(line: str):
    page_match = PAGE_RE.search(line)
    page = int(page_match.group(1)) if page_match else None

    dt_match = DATETIME_RE.search(line)
    if not dt_match:
        raise ValueError("Datetime information not found.")
    dt_str = dt_match.group(1)

    dt = datetime.datetime.strptime(dt_str, "%A, %d %B %Y %H:%M:%S")

    return dt, page


def extract_title_author(line: str) -> tuple[str, str]:
    title_author_match = TITLE_AUTHOR_RE.match(line)
    if title_author_match is None:
        raise ValueError()
    title, author = title_author_match.groups()
    return title, author


def preprocess(text: str) -> list[str]:
    return [el for el in text.strip("\n\ufeff").split("\n") if el]


def are_similar(s1: str, s2: str) -> bool:
    return s1 in s2 or s2 in s1 or SequenceMatcher(None, s1, s2).ratio() > 0.8


@dataclass
class Highlight:
    title: str
    author: str
    page: int | None
    location: tuple[int, int]
    added_on: datetime.datetime
    text: str
    note: str = ""

    def __post_init__(self):
        self.title = self.title.split("_")[0]
        self.title = self.title.replace("-", " ")
        if not RUSSIAN_ALPH & set(self.title):
            self.title = self.title.title()


def clean_highlights(hightlights: list[Highlight]) -> list[Highlight]:
    res = [hightlights[0]]
    for hl in hightlights:
        if are_similar(hl.text, res[-1].text):
            hl.note = hl.note or res[-1].note
            res.pop()
        res.append(hl)
    return res


GroupedHighlightsType = defaultdict[str, list[Highlight]]


def compile_highlights(lines: list[str]) -> list[Highlight]:
    res: list[Highlight] = []
    for line in lines:
        line = line.strip()
        sections = preprocess(line)
        if not sections:
            continue
        loc = extract_location(sections[1])
        match ClipType.detect_type(line):
            case ClipType.HIGHLIGHT:
                assert len(loc) == 2
                title, author = extract_title_author(sections[0])
                dt, page = extract_datetime_page(sections[1])
                res.append(Highlight(title, author, page, loc, dt, sections[2]))
            case ClipType.NOTE:
                assert len(loc) == 1
                for early_highlight in reversed(res):
                    if (
                        early_highlight.location[0]
                        <= loc[0]
                        <= early_highlight.location[1]
                    ):
                        early_highlight.note = "\n".join(sections[2:])
                        break
                else:
                    warnings.warn(f"Unmatched note: {line} at {loc=}")
            case ClipType.BOOKMARK:
                ...
    return res


def group_titles(highlights: list[Highlight]) -> GroupedHighlightsType:
    groups = defaultdict(list)
    for hl in highlights:
        groups[hl.title].append(hl)
    return groups


def get_all_grouped_highlights(
    surpress_warnings: bool = True,
) -> GroupedHighlightsType:
    with warnings.catch_warnings():
        if surpress_warnings:
            warnings.simplefilter("ignore")
        return group_titles(
            clean_highlights(
                compile_highlights(
                    CLIPPINGS_TXT.read_text(encoding="utf-8").split("==========")
                )
            )
        )
