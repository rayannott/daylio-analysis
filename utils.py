import datetime
from itertools import dropwhile
from dataclasses import dataclass
from typing import NamedTuple


WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


class CompleteAnalysisNT(NamedTuple):
    activity: str
    mood_with: float
    mood_without: float
    change: float
    num_of_occurances: int


@dataclass
class StatsResult:
    mood: tuple[float, float]
    note_length: tuple[float, float]
    entries_frequency: float

    def __repr__(self) -> str:
        FORMAT = '{}: {:.3f} ± {:.3f}{}'

        additional = f' (once every {1/self.entries_frequency:.2f} days)' if self.entries_frequency < 1. else ''
        return '\n'.join([
            FORMAT.format('Mood', *self.mood, ''),
            FORMAT.format('Note length', *self.note_length, ' symbols'),
            f'Entries frequency: {self.entries_frequency:.3f} entries per day{additional}'
        ])


def datetime_from_now(dt: datetime.datetime) -> str:
    """Returns a string saying how long ago the datetime object is."""
    now = datetime.datetime.now()
    diff = now - dt
    years = diff.days // 365; diff -= datetime.timedelta(days=years*365)
    months = diff.days // 30; diff -= datetime.timedelta(days=months*30)
    days = diff.days; diff -= datetime.timedelta(days=days)
    hours = diff.seconds // 3600; diff -= datetime.timedelta(hours=hours)
    minutes = diff.seconds // 60; diff -= datetime.timedelta(minutes=minutes)
    words = ['year', 'month', 'day', 'hour', 'minute']
    values = [years, months, days, hours, minutes]
    res = ''
    for value, word in dropwhile(lambda x: x[0] == 0, zip(values, words)):
        res += f'{value} {word}{"s" if value > 1 else ""} '
    return res + 'ago' if res else 'just now'
