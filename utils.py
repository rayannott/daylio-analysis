import datetime
from itertools import dropwhile
from dataclasses import dataclass
from typing import NamedTuple


WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


class MoodStd(NamedTuple):
    mood: float
    std: float

    def __repr__(self) -> str:
        return f'{self.mood:.3f} ± {self.std:.3f}'


class MoodWithWithout(NamedTuple):
    with_: MoodStd
    without: MoodStd

    def calc_change(self) -> float:
        return (self.with_[0] - self.without[0]) / self.without[0]
    
    def __str__(self) -> str:
        return f'''with: {self.with_}
without: {self.without}
change: {self.calc_change():.2%}'''


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
        FORMAT = '{}: {:.3f} ± {:.3f}{}'
        median_timedelta = datetime.timedelta(days=1/self.entries_frequency)
        return '\n'.join([
            FORMAT.format('Mood', *self.mood, ''),
            FORMAT.format('Note length', *self.note_length, ' symbols'),
            f'Entries frequency: {self.entries_frequency:.3f} entries per day (once every {timedelta_for_humans(median_timedelta)})'
        ])


def timedelta_for_humans(timedelta: datetime.timedelta) -> str:
    """Returns a string saying how long ago the datetime object is."""
    total_seconds = int(timedelta.total_seconds())
    years, remainder = divmod(total_seconds, 31536000)
    months, remainder = divmod(remainder, 2628000)
    days, remainder = divmod(remainder, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    words = ['year', 'month', 'day', 'hour', 'minute', 'second']
    values = [years, months, days, hours, minutes, seconds]
    res = ''
    for value, word in dropwhile(lambda x: x[0] == 0, zip(values, words)):
        res += f'{value} {word}{"s" if value > 1 else ""} ' if value else ''
    return res.strip()


def datetime_from_now(dt: datetime.datetime) -> str:
    """Returns a string saying how long ago the datetime object is."""
    res = timedelta_for_humans(datetime.datetime.now() - dt)
    return res + ' ago' if res else 'just now'
