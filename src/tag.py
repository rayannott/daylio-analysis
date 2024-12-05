import re
import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.entry import Entry

TAG_RE = re.compile(r"#([\w-]+)(?:\(([^)]+)\))?")
BODY_TITLE_RE = re.compile(r"([^;]+;)?(.*)")


@dataclass(frozen=True)
class Tag:
    """
    A class to represent a tag in a note.

    A tag is a string starting with a hash (#). It has an optional body and a title.
    The body is the text inside parentheses. The title (optional) the part before the semicolon.

    Examples:
        #tag -> Tag(tag='tag', body='', title='')
        #tag(body) -> Tag(tag='tag', body='body', title='')
        #tag(title; body) -> Tag(tag='tag', body='body', title='title')
        #tag(title;) -> Tag(tag='tag', body='', title='title')

    Example usage:
        when finished reading a book and have some thoughts
            "#book(book title; I liked the book)"
        when suddenly have a good idea
            "#idea(eureka!; some good idea here)"
        when making a prediction to be checked later
            "#prediction(us-elections-2024; pretty sure Trump will lose)"
            "#prediction(us-elections-2024; false: Trump actually won, holy shit)"
    """

    tag: str
    title: str
    body: str
    _linked_entry: "Entry"

    @classmethod
    def pull_tags(cls, entry: "Entry"):
        for m in TAG_RE.finditer(entry.note):
            tag, body_all = m.groups()
            if body_all and (body_match := BODY_TITLE_RE.match(body_all)):
                title, body = body_match.groups()
                title = title.strip("; ") if title is not None else ""
                body = body.strip()
            else:
                title = ""
                body = body_all
            yield cls(tag, title, body, entry)

    @property
    def full_date(self) -> datetime.datetime:
        return self._linked_entry.full_date


RATING_RE = re.compile(r"(\d+(\.\d+)?)\/10")
NUM_PAGES_RE = re.compile(r"(\d+)[p|с]")


class BookTag(Tag):
    @property
    def rating(self) -> float | None:
        rating = RATING_RE.search(self.body)
        return float(rating.group(1)) if rating else None

    @property
    def number_of_pages(self) -> int | None:
        num_pages = NUM_PAGES_RE.search(self.body)
        return int(num_pages.group(1)) if num_pages else None
