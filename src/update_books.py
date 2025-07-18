import os
from typing import NamedTuple
from datetime import datetime
from time import perf_counter as pc

from supabase import Client, create_client
import dotenv

from src.tag import BookTag


dotenv.load_dotenv()


SUPABASE_API_KEY = os.environ.get("SUPABASE_API_KEY") or ""
SUPABASE_PROJECT_ID = os.environ.get("SUPABASE_PROJECT_ID") or ""


class Book(NamedTuple):
    """A class for comparing."""

    dt_read: datetime  # this is the id, primary key
    title: str  # this is the id, primary key
    author: str | None
    rating: float | None
    n_pages: int | None
    body: str

    def __repr__(self) -> str:
        return f"{self.title} from {self.dt_read:%d.%m.%Y}"

    @classmethod
    def from_book_tag(cls, book_tag: BookTag) -> "Book":
        return Book(
            title=book_tag.title,
            author=book_tag.author,
            rating=book_tag.rating,
            n_pages=book_tag.number_of_pages,
            body=book_tag.body_clean,
            dt_read=book_tag._linked_entry.full_date,
        )

    @classmethod
    def from_sql_row(cls, row: dict) -> "Book":
        return cls(
            dt_read=datetime.fromisoformat(row["dt_read"]),
            title=row["title"],
            author=row.get("author"),
            rating=row.get("rating"),
            n_pages=row.get("n_pages"),
            body=row["body"],
        )

    def to_row(self) -> dict:
        return {
            "dt_read": self.dt_read.isoformat(),
            "title": self.title,
            "author": self.author,
            "rating": self.rating,
            "n_pages": self.n_pages,
            "body": self.body,
        }

    def insert(self, client: Client):
        client.table("books").insert(self.to_row()).execute()

    def update(self, client: Client):
        _ = (
            client.table("books")
            .update(self.to_row())
            .match({"title": self.title, "dt_read": self.dt_read.isoformat()})
            .execute()
        )


def update_books(book_tags: list[BookTag]):
    client = create_client(
        f"https://{SUPABASE_PROJECT_ID}.supabase.co",
        SUPABASE_API_KEY,
    )

    existing_rows = client.table("books").select("*").execute().data
    existing_books = {
        (b.title, b.dt_read): b for b in map(Book.from_sql_row, existing_rows)
    }

    new_books = [Book.from_book_tag(bt) for bt in book_tags]

    for book in new_books:
        key = (book.title, book.dt_read)
        existing = existing_books.get(key)
        if existing is None:
            t0 = pc()
            book.insert(client)
            print(f"Inserted {book} ({pc() - t0:.3f} sec)")
        elif book != existing:
            t0 = pc()
            book.update(client)
            print(f"Updated {book} ({pc() - t0:.3f} sec)")
