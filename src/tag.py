import re
import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from src.clippings import Highlight, GroupedHighlightsType

if TYPE_CHECKING:
    from src.entry import Entry

TAG_RE = re.compile(r"#([\w-]+)(?:\(([^)]+)\))?")
BODY_TITLE_RE = re.compile(r"([^;]+;)?(.*)")


@dataclass
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
            "#книга(book title; I liked the book)"
        when suddenly have a good idea
            "#idea(eureka!; some good idea here)"
    """

    tag: str
    title: str
    body: str
    _linked_entry: "Entry"

    @classmethod
    def pull_tags(cls, entry: "Entry") -> Iterable["Tag"]:
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
AUTHOR_RE = re.compile(r"\[(.+)\]")


class BookTag(Tag):
    highlights: list[Highlight] = []  # bad

    @classmethod
    def from_tag(cls, tag: Tag) -> "BookTag":
        assert tag.tag == "книга", "Only 'книга' tags are supported"
        assert tag.title, "Book title is required"
        return cls(tag.tag, tag.title, tag.body, tag._linked_entry)

    def __repr__(self) -> str:
        author_str = f" [{self.author}]" if self.author else ""
        rating_str = f" {self.rating}/10" if self.rating else ""
        pages_str = f" {self.number_of_pages}p" if self.number_of_pages else ""
        return f"Book({self.title}{author_str}{rating_str}{pages_str}: {self.body_clean}; entry={self._linked_entry})"

    @property
    def rating(self) -> float | None:
        rating = RATING_RE.search(self.body)
        return float(rating.group(1)) if rating else None

    @property
    def number_of_pages(self) -> int | None:
        num_pages = NUM_PAGES_RE.search(self.body)
        return int(num_pages.group(1)) if num_pages else None

    @property
    def author(self) -> str | None:
        author = AUTHOR_RE.search(self.body)
        return author.group(1) if author else None

    @property
    def body_clean(self) -> str:
        cleaned = RATING_RE.sub("", self.body)
        cleaned = NUM_PAGES_RE.sub("", cleaned)
        if m := AUTHOR_RE.search(cleaned):
            if m.end() == len(cleaned):
                cleaned = cleaned[: m.start()]
            else:
                cleaned = AUTHOR_RE.sub(m.group(1), cleaned)
        cleaned = cleaned.strip(" ,")
        return cleaned

    def try_assign_highlights(self, groups: GroupedHighlightsType):
        for k, v in groups.items():
            if k.lower() == self.title.lower():
                self.highlights = v
