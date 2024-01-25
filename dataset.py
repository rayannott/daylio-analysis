import csv, datetime, pathlib, json
from functools import lru_cache
from statistics import mean
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Callable, Iterator, Literal

import plotly.express as px
import plotly.graph_objs as go


DATA_DIR = pathlib.Path('other', 'daylio-data')
REMOVE: set[str] = set(json.load(open(pathlib.Path('other') / 'daylio-data' / 'to_remove.json', 'r', encoding='utf-8-sig')))

MOOD_VALUES = {
    'bad': 1., 'meh': 2., 'less ok': 2.5, 
    'ok': 3., 'alright': 3.5, 'good': 4., 
    'better': 4.5, 'great': 5., 'awesome': 6.
}

DT_FORMAT_READ = r"%Y-%m-%d %H:%M"
DT_FORMAT_SHOW = r"%d.%m.%Y %H:%M"
DATE_FORMAT_SHOW = r"%d.%m.%Y"

BAD_MOOD = {1., 2., 2.5}
AVERAGE_MOOD = {3., 3.5, 4.}
GOOD_MOOD = {5., 6.}

WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

MoodCondition = float | set[float] | None
NoteCondition = str | Iterator[str] | None
InclExclActivities = str | set[str]
EntryPredicate = Callable[['Entry'], bool]


@dataclass
class Entry:
    full_date: datetime.datetime
    mood: float
    activities: set[str]
    note: str

    @staticmethod
    def from_dict(row: dict[str, str]) -> 'Entry':
        datetime_str = row['full_date'] + ' ' + row['time']
        return Entry(
            full_date=datetime.datetime.strptime(datetime_str, DT_FORMAT_READ),
            mood=MOOD_VALUES[row['mood']],
            activities=set(row['activities'].split(' | ')) if row['activities'] else set(),
            note=row['note'].replace('<br>', '\n')
        )

    def __repr__(self) -> str:
        return f'[{self.full_date.strftime(DT_FORMAT_SHOW)}] {self.mood} {", ".join(self.activities)}'

    def check_condition(self, 
            incl_act: InclExclActivities,
            excl_act: InclExclActivities, 
            when: datetime.date | str | None, 
            mood: MoodCondition,
            note_contains: NoteCondition,
            predicate: EntryPredicate | None
            ) -> bool:
        """
        Checks if an entry (self) fulfils all of the following conditions:
            has an activity from incl_act
            does not have an activity from excl_act
            is recorded on a particular day
            matches the mood (an exact value or a container of values).
        
        incl_act: a string or a set of strings
        excl_act: a string or a set of strings
        when: a datetime.date object or a string in the format dd.mm.yyyy
        mood: a float or a container of floats
        note_contains: a string or a container of strings
        """
        if predicate is not None and not predicate(self): return False
        if isinstance(incl_act, str): incl_act = {incl_act}
        if isinstance(excl_act, str): excl_act = {excl_act}
        if incl_act & excl_act:
            raise ValueError(f'Some activities are included and excluded at the same time:\n{incl_act=}\n{excl_act=}')
        note_condition_result = (
            True 
            if note_contains is None else
            note_contains in self.note.lower() 
            if isinstance(note_contains, str) else 
            any(el in self.note.lower() for el in note_contains)
        )
        if isinstance(when, str):
            when = datetime.datetime.strptime(when, DATE_FORMAT_SHOW).date()
        return (
            (True if not incl_act else bool(incl_act & self.activities)) and
            (not excl_act & self.activities) and
            (True if when is None else self.full_date.date() == when) and
            (True if mood is None else (self.mood in mood if isinstance(mood, set) else self.mood == mood)) and
            note_condition_result and
            (True if predicate is None else predicate(self))
        )


class Dataset:
    def _from_csv_file(self, csv_file_path: str | pathlib.Path):
        self.entries: list[Entry] = []
        with open(csv_file_path, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.entries.append(Entry.from_dict(row))

    def __init__(self, 
            *, 
            csv_file_path: str | pathlib.Path | None = None, 
            entries: list[Entry] | None = None, 
            remove: bool = True
        ) -> None:
        if entries is not None:
            self.entries = entries
        elif csv_file_path is not None:
            self._from_csv_file(csv_file_path)
            if remove:
                for entr in self.entries:
                    entr.activities -= REMOVE
            print(self)
        else:
            self.entries = []
    
    def __repr__(self) -> str:
        # TODO: change this
        return f'Dataset({len(self.entries)} entries)'

    def __str__(self) -> str:
        # TODO: implement
        ...

    def __getitem__(self, _date_str: str) -> list[Entry]:
        return self.group_by_day().get(datetime.datetime.strptime(_date_str, DATE_FORMAT_SHOW).date(), [])
    
    def __iter__(self) -> Iterator[Entry]:
        return iter(self.entries)

    def __call__(self, arg):
        # TODO: implement??
        pass
    
    def __len__(self) -> int:
        return len(self.entries)

    @lru_cache(maxsize=None)
    def group_by_day(self) -> defaultdict[datetime.date, list[Entry]]:
        """
        Returns a defaultdict of entries grouped by day with 
        the keys as datetime.date objects, the values are lists of Entry objects.
        The entries are sorted by date in ascending order.
        """
        dd = defaultdict(list)
        for e in reversed(self.entries):
            dd[e.full_date.date()].append(e)
        return dd
    
    def sub(self, 
            incl_act: InclExclActivities = set(),
            excl_act: InclExclActivities = set(), 
            when: datetime.date | None = None, 
            mood: MoodCondition = None,
            note_contains: NoteCondition = None,
            predicate: EntryPredicate | None = None
        ) -> 'Dataset':
        """
        Returns a new Dataset object which is a subset of self
        with the entries filtered according to the arguments
        """
        return Dataset(
            entries=[e for e in self if e.check_condition(incl_act, excl_act, when, mood, note_contains, predicate)]
        )

    def mood(self) -> float:
        """
        Get the average mood among all entries
        """
        return sum(e.mood for e in self)/len(self.entries)
    
    def activities(self) -> Counter[str]:
        """
        Returns a Counter object for all activities in the dataset.
        Use `self.activities().keys()` to get only the set of all activities.
        """
        c = Counter()
        for e in self:
            c.update(e.activities)
        return c
    
    def get_datetimes(self) -> list[datetime.datetime]:
        return [e.full_date for e in self]

    def head(self, n: int = 5) -> None:
        """
        Prints the last n entries;
        if n is not given, prints the last 5 entries;
        if n == -1, prints all entries.
        """
        # TODO: change this?
        print(self)
        if n == -1:
            n = len(self.entries)
        for e in self.entries[:n]:
            print(e)
        if len(self.entries) > n:
            print('...')
    
    def mood_with_without(self, activity: str) -> tuple[float, float]:
        df_with = self.sub(incl_act={activity})
        df_without = self.sub(excl_act={activity})
        return df_with.mood(), df_without.mood()
    
    def complete_analysis(self) -> list[tuple[str, float, float, float, int]]:
        """
        Analyse all activities that occur at least 10 times.
        Return a list of tuples (activity, mood_with, mood_without, change, num_of_occurances), 
            where `change` is the mood change.
        """
        cnt = self.activities()
        res = []
        for act, num in cnt.items():
            if num < 10: continue
            mood_with, mood_without = self.mood_with_without(act)
            res.append((act, mood_with, mood_without, (mood_with - mood_without)/mood_without, num))
        res.sort(key=lambda x: x[3], reverse=True)
        return res

    def mood_plot(self) -> go.Figure:
        dd = self.group_by_day()
        days = list(dd.keys())
        avg_moods, max_moods, min_moods = [], [], []
        for day_entries in dd.values():
            this_day_moods = [e.mood for e in day_entries]
            avg_moods.append(sum(this_day_moods)/len(this_day_moods))
            max_moods.append(max(this_day_moods))
            min_moods.append(min(this_day_moods))

        fig = go.Figure([
            go.Scatter(
                name='avg',
                x=days,
                y=avg_moods,
                mode='lines+markers',
                line=dict(color='rgb(31, 119, 180)'),
            ),
            go.Scatter(
                name='max',
                x=days,
                y=max_moods,
                mode='lines',
                marker=dict(color='#444'),
                line=dict(width=0),
                showlegend=False
            ),
            go.Scatter(
                name='min',
                x=days,
                y=min_moods,
                marker=dict(color='#444'),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(130, 130, 130, 0.45)',
                fill='tonexty',
                showlegend=False
            )
        ])
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Mood',
            hovermode='x',
            showlegend=False, 
            template='plotly_dark'
        )
        fig.update_yaxes(
            rangemode='tozero', 
        )
        return fig

    def by_time_bar_plot(self,
            what: Literal['weekday', 'hour', 'day', 'month'],
            swap_freq_mood: bool = False
        ) -> go.Figure:
        """
        Groups entries by weekday, hour or day and plots a bar chart.
        The color of the bars represents the number of entries,
        the height of the bars represents the average mood.

        what: 'weekday', 'hour', 'day' or 'month' - what to group by
        swap_freq_mood: if True, the frequency and mood will be swapped in the bar chart
        """
        FUNC_MAP: dict[str, Callable[[datetime.datetime], int]] = {
            'weekday': datetime.datetime.weekday, 
            'hour': lambda x: x.hour, 
            'day': lambda x: x.day,
            'month': lambda x: x.month
        }

        mood_by: defaultdict[int, list[float]] = defaultdict(list)
        for entry in self:
            mood_by[FUNC_MAP[what](entry.full_date)].append(entry.mood)
        AVG_MOOD = list(map(mean, mood_by.values()))
        FREQ = list(map(len, mood_by.values()))
        fig = px.bar(
            x=mood_by.keys(),
            y=AVG_MOOD if not swap_freq_mood else FREQ,
            color=FREQ if not swap_freq_mood else AVG_MOOD,
            color_continuous_scale='viridis',
            labels={
                'x': what.title(), 
                'y': 'Mood' if not swap_freq_mood else 'Number of entries', 
                'color': 'Number of entries' if not swap_freq_mood else 'Mood'
            },
            title=f'Mood by {what}'
        )
        fig.update_layout(
            template='plotly_dark', 
            xaxis={'dtick': 1},
        )

        WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        if what in {'weekday', 'month'}:
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(7)) if what == 'weekday' else list(range(1, 13)),
                ticktext=WEEKDAYS if what == 'weekday' else MONTHS
            )
        return fig

    def note_length_plot(self, cap_length: int = -1) -> go.Figure:
        """
        Line plot of note lengths vs date.
        cap_length: if not -1, then the length of each note is capped at this value,
            i.e. if a note is longer than cap_length, its length is set to cap_length.
        """
        day_to_total_note_len = defaultdict(int)
        for day, entries in self.group_by_day().items():
            for entry in entries:
                day_to_total_note_len[day] += len(entry.note) if cap_length == -1 else min(len(entry.note), cap_length)
        
        fig = px.line(
            x=day_to_total_note_len.keys(),
            y=day_to_total_note_len.values(),
            labels={'x': 'Date', 'y': 'Total note length'},
            title='Total note length'
        )
        fig.update_layout(
            template='plotly_dark'
        )
        return fig
