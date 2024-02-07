import datetime
from itertools import dropwhile
from dataclasses import dataclass


WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


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
    total_seconds = int((datetime.datetime.now() - dt).total_seconds())
    years, remainder = divmod(total_seconds, 31536000)
    months, remainder = divmod(remainder, 2628000)
    days, remainder = divmod(remainder, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    words = ['year', 'month', 'day', 'hour', 'minute', 'second']
    values = [years, months, days, hours, minutes, seconds]
    res = ''
    for value, word in dropwhile(lambda x: x[0] == 0, zip(values, words)):
        res += f'{value} {word}{"s" if value > 1 else ""} '
    return res + 'ago' if res else 'just now'
