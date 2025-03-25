from __future__ import annotations
import datetime
from itertools import dropwhile
from dataclasses import dataclass
from typing import Iterable, NamedTuple, TYPE_CHECKING, Literal, Callable

if TYPE_CHECKING:
    from src.entry import Entry

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

MOOD_VALUES = {
    "bad": 1.0,
    "meh": 2.0,
    "less ok": 2.5,
    "ok": 3.0,
    "alright": 3.5,
    "fine": 4.0,
    "good": 4.3,
    "great": 4.7,
    "amazing": 5.3,
    "awesome": 6.0,
}

DT_FORMAT_READ = r"%Y-%m-%d %H:%M"
DT_FORMAT_SHOW = r"%d.%m.%Y %H:%M"
DATE_FORMAT_SHOW = r"%d.%m.%Y"

NoteCondition = str | Iterable[str]
IncludeExcludeActivities = str | set[str]
GroupByTypes = Literal["day", "week", "month"]
EntryPredicate = Callable[["Entry"], bool]

FMTS = ["%d.%m.%Y", "%d %b %Y", "%d %B %Y"]


def parse_date(date: str) -> datetime.datetime:
    for fmt in FMTS:
        try:
            return datetime.datetime.strptime(date, fmt)
        except ValueError:
            pass
    raise ValueError(f"Could not parse date: {date}")


class MoodStd(NamedTuple):
    mood: float
    std: float

    def __repr__(self) -> str:
        return f"{self.mood:.3f} ± {self.std:.3f}"


class MoodWithWithout(NamedTuple):
    with_: MoodStd
    without: MoodStd
    n_entries_with_: int
    n_entries_without: int

    def calc_change(self) -> float:
        return (self.with_[0] - self.without[0]) / self.without[0]

    def __str__(self) -> str:
        return f"""with:    {self.with_} (n={self.n_entries_with_:,})
without: {self.without} (n={self.n_entries_without:,})
change:  {self.calc_change():.2%}"""


class CompleteAnalysis(NamedTuple):
    activity: str
    mood_with_without: MoodWithWithout
    num_of_occurances: int


@dataclass
class StatsResult:
    mood: MoodStd
    note_length: tuple[float, float]
    entries_frequency: float | None
    number_of_activities: int

    def __rshift__(self, other: StatsResult) -> MoodWithWithout:
        mood_change = MoodWithWithout(
            with_=other.mood,
            without=self.mood,
            n_entries_with_=0,
            n_entries_without=0,
        )
        return mood_change

    def __repr__(self) -> str:
        FORMAT = "{}: {:.3f} ± {:.3f}{}"
        dat = [
            FORMAT.format("- mood", *self.mood, ""),
            FORMAT.format("- note length", *self.note_length, " symbols"),
        ]
        dat.append(f"- number of activities: {self.number_of_activities:,}")
        if self.entries_frequency:
            dat.append(
                f"- entries frequency: {self.entries_frequency:.3f} entries per day (once every {timedelta_for_humans(datetime.timedelta(days=1 / self.entries_frequency))})"
            )

        res = f"Stats(\n    {'\n    '.join(dat)}\n)"
        return res


def timedelta_for_humans(timedelta: datetime.timedelta) -> str:
    """Returns a string saying how long ago the datetime object is."""
    total_seconds = int(timedelta.total_seconds())
    years, remainder = divmod(total_seconds, 31536000)
    months, remainder = divmod(remainder, 2628000)
    days, remainder = divmod(remainder, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    words = ["year", "month", "day", "hour", "minute", "second"]
    values = [years, months, days, hours, minutes, seconds]
    res = ""
    for value, word in dropwhile(lambda x: x[0] == 0, zip(values, words)):
        res += f"{value} {word}{'s' if value > 1 else ''} " if value else ""
    return res.strip()


def datetime_from_now(dt: datetime.datetime) -> str:
    """Returns a string saying how long ago the datetime object is."""
    res = timedelta_for_humans(datetime.datetime.now() - dt)
    return res + " ago" if res else "just now"
