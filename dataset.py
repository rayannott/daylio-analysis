import csv, datetime, pathlib, json, re
from io import TextIOWrapper
from itertools import groupby, islice, pairwise
from statistics import mean, stdev, median
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Callable, Iterator, Literal

import plotly.express as px
import plotly.graph_objs as go

from utils import datetime_from_now, WEEKDAYS, MONTHS, StatsResult, CompleteAnalysis, MoodWithWithout

REMOVE: set[str] = set(json.load(open(pathlib.Path('data') / 'to_remove.json', 'r', encoding='utf-8-sig')))

MOOD_VALUES = {
    'bad': 1., 'meh': 2., 'less ok': 2.5, 
    'ok': 3., 'alright': 3.5, 'good': 4., 
    'better': 4.5, 'great': 5., 'awesome': 6.
}

BAD_MOOD = {1., 2., 2.5}
AVERAGE_MOOD = {3., 3.5, 4.}
GOOD_MOOD = {5., 6.}

MoodCondition = float | set[float]
NoteCondition = str | Iterator[str]
InclExclActivities = str | set[str]
EntryPredicate = Callable[['Entry'], bool]

DT_FORMAT_READ = r"%Y-%m-%d %H:%M"
DT_FORMAT_SHOW = r"%d.%m.%Y %H:%M"
DATE_FORMAT_SHOW = r"%d.%m.%Y"

DATE_PATTERN = re.compile(r'\d{2}\.\d{2}\.\d{4}')
DATETIME_PATTERN = re.compile(r'\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}')


def date_slice_to_entry_predicate(_slice: slice) -> EntryPredicate:
    if not (_slice.start or _slice.stop):
        raise ValueError('At least one of the slice bounds must be given')
    if _slice.step is not None: print('[Warning]: step is not supported yet')
    _date_start = datetime.datetime.strptime(_slice.start, DATE_FORMAT_SHOW) if _slice.start else None
    _date_stop = datetime.datetime.strptime(_slice.stop, DATE_FORMAT_SHOW) if _slice.stop else None
    def check_date(entry: Entry) -> bool:
        return (
            (True if _date_start is None else entry.full_date >= _date_start) and
            (True if _date_stop is None else entry.full_date < _date_stop)
        )
    return check_date


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
        P = '{}'
        return f'{self}\n{P[0]}{self.note}{P[1]}'

    def check_condition(self, 
            include: InclExclActivities,
            exclude: InclExclActivities, 
            mood: MoodCondition | None,
            note_pattern: NoteCondition | None,
            predicate: EntryPredicate | None
            ) -> bool:
        """
        Checks if an entry (self) fulfils all of the following conditions:
            has an activity from include
            does not have an activity from exclude
            is recorded on a particular day (or a range of days)
            matches the mood (an exact value or a container of values).
        
        include: a string or a set of strings
        exclude: a string or a set of strings
        mood: a float or a container of floats
        note_contains: a regex pattern or a container of regex patterns
        predicate: a function that takes an Entry object and returns a bool
        """
        if predicate is not None and not predicate(self): return False
        if isinstance(include, str): include = {include}
        if isinstance(exclude, str): exclude = {exclude}
        if include & exclude:
            raise ValueError(f'Some activities are included and excluded at the same time: {include=}; {exclude=}')
        note_condition_result = (
            True 
            if note_pattern is None else
            bool(re.findall(note_pattern, self.note))
            if isinstance(note_pattern, str) else
            any(re.findall(pattern, self.note) for pattern in note_pattern)
        )
        return (
            (True if not include else bool(include & self.activities)) and
            (not exclude & self.activities) and
            # when_condition_result and
            (True if mood is None else (self.mood in mood if isinstance(mood, set) else self.mood == mood)) and
            note_condition_result and
            (True if predicate is None else predicate(self))
        )


class Dataset:
    @staticmethod
    def _from_csv_file(csv_file_path: str | pathlib.Path):
        entries: list[Entry] = []
        with open(csv_file_path, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                entries.append(Entry.from_dict(row))
        return entries

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
            self.entries = Dataset._from_csv_file(csv_file_path)
            if remove:
                for entr in self.entries:
                    entr.activities -= REMOVE
            print(self)
        else:
            raise ValueError('Either a CSV file path or a list of entries must be provided')
    
    def __repr__(self) -> str:
        if not self.entries: return 'Dataset(0 entries)'
        latest_entry_full_date = self.entries[0].full_date
        return f'Dataset({len(self.entries)} entries; last [{datetime_from_now(latest_entry_full_date)}]; mood: {self.mood():.3f} ± {self.mood_std():.3f})'

    def __getitem__(self, _date: str | slice) -> 'Dataset':
        """
        Returns a new Dataset object which is a subset of self
        with the entries filtered according to the date or date range as a slice.
        """
        if isinstance(_date, slice):
            CHECK_FN = date_slice_to_entry_predicate(_date)
        else:
            date = datetime.datetime.strptime(_date, DATE_FORMAT_SHOW).date()
            CHECK_FN: EntryPredicate = lambda e: e.full_date.date() == date
        return Dataset(_entries=[e for e in self if CHECK_FN(e)])

    def __iter__(self) -> Iterator[Entry]:
        return iter(self.entries)

    def __call__(self, _date: str) -> list[Entry]:
        """
        Return a list of entries for a particular day.
        The entries are sorted by time in ascending order.
        Thus, _date is a string in the format dd.mm.yyyy.
        """
        if DATE_PATTERN.fullmatch(_date):
            return self.group_by('day').get(datetime.datetime.strptime(_date, DATE_FORMAT_SHOW).date(), [])
        raise ValueError('Invalid date format: use dd.mm.yyyy')
    
    def __matmul__(self, datetime_like: str | datetime.datetime) -> Entry | None:
        """
        Wrapper for the `at` method.
        """
        return self.at(datetime_like)
    
    def __len__(self) -> int:
        return len(self.entries)

    def people(self) -> dict[str, int]:
        """
        Returns a Counter-like dict of people and the number of times they appear in the dataset.
        """
        return {activity: num for activity, num in self.activities().items() if activity[0].isupper()}
    
    def at(self, datetime_like: str | datetime.datetime) -> Entry | None:
        """
        Returns the entry for a particular datetime or None if there is no such entry.

        datetime_str: a string in the format dd.mm.yyyy HH:MM

        This is used when calling the Dataset object as a function.
        """
        if isinstance(datetime_like, str):
            if DATETIME_PATTERN.fullmatch(datetime_like):
                datetime_ = datetime.datetime.strptime(datetime_like, DT_FORMAT_SHOW)
            else:
                raise ValueError(f'Invalid date string: {datetime_like}; expected format: dd.mm.yyyy or dd.mm.yyyy HH:MM')
        elif isinstance(datetime_like, datetime.datetime):
            datetime_ = datetime_like
        else:
            raise ValueError(f'Invalid type for datetime_like: {type(datetime_like)}; expected str or datetime.datetime')
        for entry in self.entries:
            if entry.full_date == datetime_:
                return entry
        return None

    def group_by(self,
            what: Literal['day', 'month']
        ) -> dict[datetime.date, list[Entry]]:
        """
        Returns a dict of entries grouped by day with the keys as datetime.date objects, the values are lists of Entry objects.
        
        The entries are sorted by date in ascending order.
        """
        KEYMAP: dict[str, Callable[[Entry], datetime.date | datetime.datetime]] = {
            'day': lambda x: x.full_date.date(),
            'month': lambda x: x.full_date.date().replace(day=1)
        }
        if what not in KEYMAP:
            raise ValueError(f'Invalid value for "what": {what}; expected one of {list(KEYMAP.keys())}')
        return {day: list(entries) for day, entries in groupby(reversed(self.entries), key=KEYMAP[what])}
    
    def sub(self, 
            *,
            include: InclExclActivities = set(),
            exclude: InclExclActivities = set(), 
            mood: MoodCondition | None = None,
            note_contains: NoteCondition | None = None,
            predicate: EntryPredicate | None = None
        ) -> 'Dataset':
        """
        Returns a new Dataset object which is a subset of self
        with the entries filtered according to the arguments.
        
        include: a string or a set of strings - only entries with at least one of these activities will be included
        exclude: a string or a set of strings - only entries without any of these activities will be included
        when: a datetime.date object, a string in the format dd.mm.yyyy or a slice with strings of that format - only entries on this day will be included
        mood: a float or a set of floats - only entries with these moods will be included
        note_contains: a string or an iterator of strings - only entries with notes matching this/these pattern(s) will be included
        predicate: a function that takes an Entry object and returns a bool - only entries for which this function returns True will be included
        """
        all_activities_set = self.activities().keys()
        if isinstance(include, str): include = {include}
        if isinstance(exclude, str): exclude = {exclude}
        if ua:=(include - all_activities_set): raise ValueError(f'Unknown activities to include: {ua}')
        if ua:=(exclude - all_activities_set): raise ValueError(f'Unknown activities to exclude: {ua}')
        return Dataset(
            _entries=[e for e in self if e.check_condition(include, exclude, mood, note_contains, predicate)]
        )

    def mood(self) -> float:
        """
        Get the average mood among all entries
        """
        return mean(e.mood for e in self)

    def mood_std(self) -> float:
        """
        Get the standard deviation of the mood among all entries
        """
        return stdev(e.mood for e in self) if len(self) > 1 else 0.
    
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
        for e in islice(self, n):
            print(e if not verbose else e.verbose(), file=file)
        if len(self.entries) > n:
            print('...', file=file)
    
    def mood_with_without(self, activity: str) -> MoodWithWithout:
        df_with = self.sub(include=activity)
        df_without = self.sub(exclude=activity)
        return MoodWithWithout(df_with.mood(), df_without.mood())
    
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
    
    def complete_analysis(self, n_threshold: int = 10) -> list[CompleteAnalysis]:
        """
        Analyse all activities that occur at least (n_threshold) times.
        Return a list of typed namedtuples
        (activity, mood_with, mood_without, change, num_of_occurances), 
            where `change` is the mood change.
        """
        cnt = self.activities()
        res: list[CompleteAnalysis] = []
        for act, num in cnt.items():
            if num < n_threshold: continue
            mood_with, mood_without = self.mood_with_without(act)
            res.append(CompleteAnalysis(act, mood_with, mood_without, (mood_with - mood_without)/mood_without, num))
        res.sort(key=lambda x: x.change, reverse=True)
        return res


    # plots:

    def mood_plot(self,
            by: Literal['day', 'month'] = 'day'
        ) -> go.Figure:
        """
        Generates a plot of the average, maximum, and minimum moods over time.
        (the area between the maximum and minimum moods is filled)

        Returns: go.Figure: The plotly figure object representing the mood plot.
        """
        dd = self.group_by(by)
        groups = list(dd.keys())
        avg_moods, max_moods, min_moods = [], [], []
        for day_entries in dd.values():
            this_day_moods = [e.mood for e in day_entries]
            avg_moods.append(mean(this_day_moods))
            max_moods.append(max(this_day_moods))
            min_moods.append(min(this_day_moods))

        fig = go.Figure([
            go.Scatter(
                name='avg',
                x=groups,
                y=avg_moods,
                # color depends on how many entries there are on that day
                marker=dict(
                    color=[len(day) for day in dd.values()],
                    colorscale='Bluered',
                    showscale=True,
                    colorbar=dict(title='Number of entries'),
                    size=10 if by == 'month' else 5,
                ),
                mode='lines+markers',
                line=dict(color='rgb(31, 119, 180)'),
            ),
            go.Scatter(
                name='max',
                x=groups,
                y=max_moods,
                mode='lines',
                marker=dict(color='#444'),
                line=dict(width=0),
                showlegend=False
            ),
            go.Scatter(
                name='min',
                x=groups,
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
        if not what in FUNC_MAP:
            raise ValueError(f'Invalid value for "what": {what}; expected one of {list(FUNC_MAP.keys())}')
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

    def entries_differences(self) -> go.Figure:
        dts = self.get_datetimes()[::-1]
        diffs = [dt2 - dt1 for dt1, dt2 in pairwise(dts)]
        fig = go.Figure(data=go.Scatter(
            x=dts[1:],
            y=[el.total_seconds() / (3600*24) for el in diffs],
            mode='lines+markers',
            line=dict(dash='dashdot'),
        ))
        if len(dts) > 100:
            window_size = len(dts) // 40
            sliding_average = [mean([el.total_seconds() / (3600*24) for el in diffs[i:i+window_size]]) for i in range(len(diffs) - window_size + 1)]
            fig.add_trace(
                go.Scatter(
                    x=dts[window_size-1:],
                    y=sliding_average,
                    mode='lines',
                    name=f'Sliding average (window size {window_size})',
                    marker=dict(size=0),
                    line=dict(width=2)
                )
            )
            # legend
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
        fig.update_layout(
            yaxis_title='difference, days',
            title='Difference between consecutive entries in days',
            template='plotly_dark'
        )
        return fig

    def note_length_plot(self) -> go.Figure:
        """
        Generates a line plot showing the average note lengths vs date.

        Args:
            cap_length (int, optional): If not -1, the length of each note is capped at this value.
                If a note is longer than cap_length, its length is set to cap_length. Default is -1.

        Returns:
            go.Figure: A plotly figure object representing the line plot.

        """
        day_to_total_note_len = defaultdict(float)
        for day, entries in self.group_by('day').items():
            tmp = []
            for entry in entries:
                tmp.append(len(entry.note))
            day_to_total_note_len[day] = mean(tmp)
        
        window_size = 11
        import numpy as np
        sliding_average = np.convolve(list(day_to_total_note_len.values()), np.ones(window_size) / window_size, mode='valid')
        
        fig = px.scatter(
            x=day_to_total_note_len.keys(),
            y=day_to_total_note_len.values(),
            labels={'x': 'Date', 'y': 'Average note length'},
            title='Average note length'
        )
        fig.add_trace(
            go.Scatter(
            x=list(day_to_total_note_len.keys())[window_size-1:],
            y=sliding_average,
            mode='lines',
            name=f'Sliding average (window size {window_size})',
            marker=dict(size=0),
            line=dict(width=2)
            )
        )
        fig.update_layout(
            template='plotly_dark',
            legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
            )
        )
        return fig


if __name__ == '__main__':
    # run `python -i dataset.py` to use in the terminal
    DATA_DIR = pathlib.Path('data')
    path = next(DATA_DIR.glob('*.csv'))
    print('using file', path.name)

    df = Dataset(csv_file_path=path)
