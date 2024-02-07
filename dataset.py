import csv, datetime, pathlib, json, re
from io import TextIOWrapper
from functools import lru_cache
from statistics import mean, stdev, median
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Callable, Iterator, Literal

import plotly.express as px
import plotly.graph_objs as go

from utils import datetime_from_now, WEEKDAYS, MONTHS, StatsResult, CompleteAnalysisNT, MoodWithWithoutNT

REMOVE: set[str] = set(json.load(open(pathlib.Path('data') / 'to_remove.json', 'r', encoding='utf-8-sig')))

MOOD_VALUES = {
    'bad': 1., 'meh': 2., 'less ok': 2.5, 
    'ok': 3., 'alright': 3.5, 'good': 4., 
    'better': 4.5, 'great': 5., 'awesome': 6.
}

BAD_MOOD = {1., 2., 2.5}
AVERAGE_MOOD = {3., 3.5, 4.}
GOOD_MOOD = {5., 6.}

MoodCondition = float | set[float] | None
NoteCondition = str | Iterator[str] | None
InclExclActivities = str | set[str]
EntryPredicate = Callable[['Entry'], bool]

DT_FORMAT_READ = r"%Y-%m-%d %H:%M"
DT_FORMAT_SHOW = r"%d.%m.%Y %H:%M"
DATE_FORMAT_SHOW = r"%d.%m.%Y"

DATE_PATTERN = re.compile(r'\d{2}\.\d{2}\.\d{4}')
DATETIME_PATTERN = re.compile(r'\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}')


@dataclass
class Entry:
    full_date: datetime.datetime
    mood: float
    activities: set[str]
    note: str

    @staticmethod
    def from_dict(row: dict[str, str]) -> 'Entry':
        """Construct an Entry object from a dictionary with the keys as in the CSV file."""
        datetime_str = row['full_date'] + ' ' + row['time']
        return Entry(
            full_date=datetime.datetime.strptime(datetime_str, DT_FORMAT_READ),
            mood=MOOD_VALUES[row['mood']],
            activities=set(row['activities'].split(' | ')) if row['activities'] else set(),
            note=row['note'].replace('<br>', '\n')
        )

    def __repr__(self) -> str:
        return f'[{self.full_date.strftime(DT_FORMAT_SHOW)}] {self.mood} {", ".join(self.activities)}'
    
    def verbose(self) -> str:
        return f'{repr(self)}\n\t{self.note}'

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
        predicate: a function that takes an Entry object and returns a bool
        """
        if predicate is not None and not predicate(self): return False
        if isinstance(incl_act, str): incl_act = {incl_act}
        if isinstance(excl_act, str): excl_act = {excl_act}
        if incl_act & excl_act:
            raise ValueError(f'Some activities are included and excluded at the same time: {incl_act=}; {excl_act=}')
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
            remove: bool = True,
            _entries: list[Entry] | None = None, # note: entries might as well have been a tuple
        ) -> None:
        """
        Construct a Dataset object from a CSV file.
        Construction using a list of entries is used within the class.
        """
        if _entries is not None:
            self.entries = _entries
        elif csv_file_path is not None:
            self._from_csv_file(csv_file_path)
            if remove:
                for entr in self.entries:
                    entr.activities -= REMOVE
            print(self)
        else:
            raise ValueError('Either a CSV file path or a list of entries must be provided')
    
    def __repr__(self) -> str:
        if not self.entries: return 'Dataset(0 entries)'
        latest_entry = self.entries[0].full_date
        return f'Dataset({len(self.entries)} entries; last [{datetime_from_now(latest_entry)}]; average mood: {self.mood():.3f})'

    def __getitem__(self, _date_str: str) -> list[Entry] | Entry:
        """
        Return a list of entries for a particular day or an entry for a particular datetime.
        Thus, _date_str can be either a string in the format dd.mm.yyyy or dd.mm.yyyy HH:MM.
        """
        if DATE_PATTERN.fullmatch(_date_str):
            return self.group_by_day().get(datetime.datetime.strptime(_date_str, DATE_FORMAT_SHOW).date(), [])
        elif DATETIME_PATTERN.fullmatch(_date_str):
            datetime_ = datetime.datetime.strptime(_date_str, DT_FORMAT_SHOW)
            for entry in self.entries:
                if entry.full_date == datetime_:
                    return entry
            raise ValueError(f'No entry for {_date_str}')
        else:
            raise ValueError(f'Invalid date string: {_date_str}; expected format: dd.mm.yyyy or dd.mm.yyyy HH:MM')

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
            when: datetime.date | str | None = None, 
            mood: MoodCondition = None,
            note_contains: NoteCondition = None,
            predicate: EntryPredicate | None = None
        ) -> 'Dataset':
        """
        Returns a new Dataset object which is a subset of self
        with the entries filtered according to the arguments.
        
        incl_act: a string or a set of strings - only entries with at least one of these activities will be included
        excl_act: a string or a set of strings - only entries without any of these activities will be included
        when: a datetime.date object or a string in the format dd.mm.yyyy - only entries on this day will be included
        mood: a float or a set of floats - only entries with these moods will be included
        note_contains: a string or an iterator of strings - only entries with notes containing this string (one of these strings) will be included
        predicate: a function that takes an Entry object and returns a bool - only entries for which this function returns True will be included
        """
        return Dataset(
            _entries=[e for e in self if e.check_condition(incl_act, excl_act, when, mood, note_contains, predicate)]
        )

    @lru_cache
    def mood(self) -> float:
        """
        Get the average mood among all entries
        """
        return sum(e.mood for e in self)/len(self.entries)
    
    @lru_cache
    def activities(self) -> Counter[str]:
        """
        Returns a Counter object for all activities in the dataset.
        Use `self.activities().keys()` to get only the set of all activities.
        """
        c = Counter()
        for e in self:
            c.update(e.activities)
        return c
    
    @lru_cache
    def get_datetimes(self) -> list[datetime.datetime]:
        return [e.full_date for e in self]

    def head(self, 
            n: int = 5, 
            file: TextIOWrapper | None = None,
            verbose: bool = False
        ) -> None:
        """
        Prints the last n entries (from newest to oldest);
        if n is not given, prints the last 5 entries;
        if n == -1, prints all entries.

        n: number of the last entries (from newest to oldest) to print; if -1, prints all entries; default: 5
        file: a file-like object to print to; by default, prints to stdout
        verbose: if True, prints the entries in a more verbose format (including the note)
        """
        print(self, file=file)
        if n == -1:
            n = len(self.entries)
        for e in self.entries[:n]:
            print(e if not verbose else e.verbose(), file=file)
        if len(self.entries) > n:
            print('...', file=file)
    
    def mood_with_without(self, activity: str) -> MoodWithWithoutNT:
        df_with = self.sub(incl_act=activity)
        df_without = self.sub(excl_act=activity)
        return MoodWithWithoutNT(df_with.mood(), df_without.mood())
    
    def stats(self) -> StatsResult:
        """
        Returns the following statistics:
            - mood (avg ± std)
            - note length [num symbols] (avg ± std)
            - entries frequency [entries per day] (median)
        as a StatsResult object.
        """
        moods = [e.mood for e in self]
        note_lengths = [len(e.note) for e in self]
        timedeltas_secs = [max(1., (d1 - d2).total_seconds()) for d1, d2 in zip(self.get_datetimes()[:-1], self.get_datetimes()[1:])]
        freqs = [24 * 60 * 60 / td for td in timedeltas_secs]
        return StatsResult(
            mood=(mean(moods), stdev(moods)),
            note_length=(mean(note_lengths), stdev(note_lengths)),
            entries_frequency=median(freqs)
        )
    
    @lru_cache
    def complete_analysis(self) -> list[CompleteAnalysisNT]:
        """
        Analyse all activities that occur at least 10 times.
        Return a list of typed namedtuples
        (activity, mood_with, mood_without, change, num_of_occurances), 
            where `change` is the mood change.
        """
        cnt = self.activities()
        res: list[CompleteAnalysisNT] = []
        for act, num in cnt.items():
            if num < 10: continue
            mood_with, mood_without = self.mood_with_without(act)
            res.append(CompleteAnalysisNT(act, mood_with, mood_without, (mood_with - mood_without)/mood_without, num))
        res.sort(key=lambda x: x.change, reverse=True)
        return res


    # plots:

    def mood_plot(self) -> go.Figure:
        """
        Generates a plot of the average, maximum, and minimum moods over time.
        (the area between the maximum and minimum moods is filled)

        Returns: go.Figure: The plotly figure object representing the mood plot.
        """
        dd = self.group_by_day()
        days = list(dd.keys())
        avg_moods, max_moods, min_moods = [], [], []
        for day_entries in dd.values():
            this_day_moods = [e.mood for e in day_entries]
            avg_moods.append(mean(this_day_moods))
            max_moods.append(max(this_day_moods))
            min_moods.append(min(this_day_moods))

        fig = go.Figure([
            go.Scatter(
                name='avg',
                x=days,
                y=avg_moods,
                # color depends on how many entries there are on that day
                marker=dict(
                    color=[len(day) for day in dd.values()],
                    colorscale='Bluered',
                    showscale=True,
                    colorbar=dict(title='Number of entries')
                ),
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
        Groups entries by weekday, hour or day and returns a bar chart.
        The color of the bars represents the number of entries,
        the height of the bars represents the average mood.

        what: 'weekday', 'hour', 'day' or 'month' - what to group by
        swap_freq_mood: if True, the frequency and mood will be swapped in the bar chart

        Returns: go.Figure: The plotly figure object representing the bar chart.
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

        if what in {'weekday', 'month'}:
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(7)) if what == 'weekday' else list(range(1, 13)),
                ticktext=WEEKDAYS if what == 'weekday' else MONTHS
            )
        return fig

    def note_length_plot(self, cap_length: int = -1) -> go.Figure:
        """
        Generates a line plot showing the average note lengths vs date.

        Args:
            cap_length (int, optional): If not -1, the length of each note is capped at this value.
                If a note is longer than cap_length, its length is set to cap_length. Default is -1.

        Returns:
            go.Figure: A plotly figure object representing the line plot.

        """
        day_to_total_note_len = defaultdict(float)
        for day, entries in self.group_by_day().items():
            tmp = []
            for entry in entries:
                tmp.append(len(entry.note) if cap_length == -1 else min(len(entry.note), cap_length))
            day_to_total_note_len[day] = mean(tmp)
        
        fig = px.line(
            x=day_to_total_note_len.keys(),
            y=day_to_total_note_len.values(),
            labels={'x': 'Date', 'y': 'Average note length'},
            title='Average note length'
        )
        fig.update_layout(
            template='plotly_dark'
        )
        return fig
