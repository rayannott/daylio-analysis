import datetime
from itertools import dropwhile


WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


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
