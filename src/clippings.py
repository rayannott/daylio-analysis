import pathlib
import datetime
import re
from dataclasses import dataclass


CLIPPINGS_TXT = pathlib.Path("data", "clippings.txt")


# Pattern for location: captures two groups of digits separated by a hyphen.
LOCATION_RE = re.compile(r"location (\d+)-(\d+)")
# Pattern for page: captures the page number following "on page ".
PAGE_RE = re.compile(r"on page (\d+)")
# Pattern for datetime: captures the date and time following "Added on ".
DATETIME_RE = re.compile(r"Added on (.+)$")


def extract_info(line: str):
    # Find location using the LOCATION_RE pattern.
    loc_match = LOCATION_RE.search(line)
    if not loc_match:
        raise ValueError("Location information not found.")
    location = (int(loc_match.group(1)), int(loc_match.group(2)))

    # Find page number if available using the PAGE_RE pattern.
    page_match = PAGE_RE.search(line)
    page = int(page_match.group(1)) if page_match else None

    # Find the datetime string using the DATETIME_RE pattern.
    dt_match = DATETIME_RE.search(line)
    if not dt_match:
        raise ValueError("Datetime information not found.")
    dt_str = dt_match.group(1)

    # Convert the datetime string into a Python datetime object.
    dt = datetime.datetime.strptime(dt_str, "%A, %d %B %Y %H:%M:%S")

    return location, dt, page


TITLE_AUTHOR_RE = re.compile(r"([\w\s',-]+) \(([\w\s',-]+)\)")


def preprocess(text: str) -> list[str]:
    return [el for el in text.strip("\n\ufeff").split("\n") if el]


@dataclass
class Highlight:
    title: str
    author: str
    page: int | None
    location: tuple[int, int]
    added_on: datetime.datetime
    text: str
    note: str

    @staticmethod
    def extract_location_page_datetime(
        section: str,
    ) -> tuple[tuple[int, int], datetime.datetime, int | None]:
        return extract_info(section)

    @classmethod
    def from_highlight_section(cls, section: str) -> "Highlight":
        sections = preprocess(section)
        title_author_match = TITLE_AUTHOR_RE.match(sections[0])
        if title_author_match is None:
            raise ValueError()

        title, author = title_author_match.groups()
        location, dt, page = cls.extract_location_page_datetime(sections[1])

        print(title, author, location, dt, page)


highlights = CLIPPINGS_TXT.read_text(encoding="utf-8").split("==========")
