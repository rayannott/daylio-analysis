from __future__ import annotations
import datetime
from itertools import dropwhile
from dataclasses import dataclass
from typing import Iterable, NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.entry import EntryPredicate, Entry

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
    "good": 4.0,
    "better": 4.5,
    "great": 5.0,
    "awesome": 6.0,
}

DT_FORMAT_READ = r"%Y-%m-%d %H:%M"
DT_FORMAT_SHOW = r"%d.%m.%Y %H:%M"
DATE_FORMAT_SHOW = r"%d.%m.%Y"

MoodCondition = float | set[float]
NoteCondition = str | Iterable[str]
IncludeExcludeActivities = str | set[str]


def date_slice_to_entry_predicate(_slice: slice) -> EntryPredicate:
    if not (_slice.start or _slice.stop):
        raise ValueError("At least one of the slice bounds must be given")
    if _slice.step is not None:
        print("[Warning]: step is not supported yet")
    _date_start = (
        datetime.datetime.strptime(_slice.start, DATE_FORMAT_SHOW)
        if _slice.start
        else None
    )
    _date_stop = (
        datetime.datetime.strptime(_slice.stop, DATE_FORMAT_SHOW)
        if _slice.stop
        else None
    )

    def check_date(entry: Entry) -> bool:
        return (True if _date_start is None else entry.full_date >= _date_start) and (
            True if _date_stop is None else entry.full_date < _date_stop
        )

    return check_date


class MoodStd(NamedTuple):
    mood: float
    std: float

    def __repr__(self) -> str:
        return f"{self.mood:.3f} ± {self.std:.3f}"


class MoodWithWithout(NamedTuple):
    with_: MoodStd
    without: MoodStd

    def calc_change(self) -> float:
        return (self.with_[0] - self.without[0]) / self.without[0]

    def __str__(self) -> str:
        return f"""with:    {self.with_}
without: {self.without}
change:  {self.calc_change():.2%}"""


class CompleteAnalysis(NamedTuple):
    activity: str
    mood_with_without: MoodWithWithout
    num_of_occurances: int


@dataclass
class StatsResult:
    mood: MoodStd
    note_length: tuple[float, float]
    entries_frequency: float

    def __repr__(self) -> str:
        FORMAT = "{}: {:.3f} ± {:.3f}{}"
        median_timedelta = datetime.timedelta(days=1 / self.entries_frequency)
        return "\n".join(
            [
                FORMAT.format("Mood", *self.mood, ""),
                FORMAT.format("Note length", *self.note_length, " symbols"),
                f"Entries frequency: {self.entries_frequency:.3f} entries per day (once every {timedelta_for_humans(median_timedelta)})",
            ]
        )


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
        res += f'{value} {word}{"s" if value > 1 else ""} ' if value else ""
    return res.strip()


def datetime_from_now(dt: datetime.datetime) -> str:
    """Returns a string saying how long ago the datetime object is."""
    res = timedelta_for_humans(datetime.datetime.now() - dt)
    return res + " ago" if res else "just now"
