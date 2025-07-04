import csv
import datetime
import pathlib
import json
import re
from io import TextIOWrapper
from itertools import groupby, islice, pairwise
from statistics import mean, stdev, median
from collections import Counter, defaultdict
from typing import Callable, Iterator, Literal
from functools import singledispatchmethod

import plotly.graph_objs as go

from src.entry import Entry, EntryPredicate
from src.tag import Tag, BookTag
from src.plotting import Plotter
from src.entry_condition import EntryCondition, DateIn
from src.utils import (
    IncludeExcludeActivities,
    NoteCondition,
    datetime_from_now,
    StatsResult,
    CompleteAnalysis,
    MoodWithWithout,
    MoodStd,
    GroupByTypes,
    parse_date,
)

TO_REMOVE_FILE = pathlib.Path("data") / "to_remove.json"

if not TO_REMOVE_FILE.exists():
    REMOVE: set[str] = set()
    print(
        f"Warning: {TO_REMOVE_FILE} not found. No activities will be removed from the dataset.")
else:
    with open(TO_REMOVE_FILE, "r", encoding="utf-8") as f:
        REMOVE = set(json.load(f))


DATETIME_PATTERN = re.compile(r"\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}")


class Dataset:
    @staticmethod
    def read_entries_from_csv(csv_file_path: str | pathlib.Path):
        entries: list[Entry] = []
        with open(csv_file_path, "r", encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                entries.append(Entry.from_dict(row))
        return entries

    def __init__(
        self,
        csv_file: str | pathlib.Path | None = None,
        *,
        remove: bool = True,
        _entries: list[Entry] | None = None,
        _generating_entry_condition: EntryCondition | None = None,
        _parent_stats: StatsResult | None = None,
    ) -> None:
        """
        Construct a Dataset object from a CSV file exported from the app.
        """
        self._generating_entry_condition = _generating_entry_condition
        self._parent_stats = _parent_stats
        if _entries is not None:
            self.entries = _entries
        elif csv_file is not None:
            self.entries = Dataset.read_entries_from_csv(csv_file)
            if remove:
                for entr in self.entries:
                    entr.activities -= REMOVE
            print(self)
        else:
            raise ValueError(
                "Either a CSV file path or a list of entries must be provided"
            )

    def __repr__(self) -> str:
        if not self.entries:
            return "Dataset(0 entries)"
        change_from_parent_info = (
            ""
            if self._parent_stats is None
            else (f" ({(self._parent_stats >> self.stats()).calc_change():+.2%})")
        )
        return (
            f"Dataset({len(self.entries)} entries; last [{datetime_from_now(self.entries[0].full_date)}]; "
            f"mood: {self.mood_std()}{change_from_parent_info})"
        )

    def __getitem__(self, _date: str | slice) -> "Dataset":
        """
        Returns a new Dataset object which is a subset of self
        with the entries filtered according to the date or date range as a slice.
        """
        di = DateIn[_date]
        return self.sub(di)  # type: ignore # TODO: find workaround

    def __iter__(self) -> Iterator[Entry]:
        return iter(self.entries)

    def __matmul__(self, datetime_like: str | datetime.datetime) -> Entry:
        """
        Wrapper for the `at` method.
        """
        return self.at(datetime_like)

    def __len__(self) -> int:
        return len(self.entries)

    def __eq__(self, other: "Dataset") -> bool:
        return self.entries == other.entries

    def people(self) -> Counter[str]:
        """
        Returns a Counter of people and the number of times they appear in the dataset.
        """
        return Counter(
            filter(
                lambda x: x[0].isupper(),
                (activity for entry in self for activity in entry.activities),
            )
        )

    @singledispatchmethod
    def at(self, datetime_like: str | datetime.datetime) -> Entry:
        """
        Returns the entry for a particular datetime-like object.

        datetime_str: a string in the format dd.mm.yyyy HH:MM

        This is used when calling the Dataset object as a function.
        """
        raise NotImplementedError(f"Unsupported type: {type(datetime_like)}")

    def _at(self, datetime_: datetime.datetime) -> Entry:
        for entry in self.entries:
            if entry.full_date == datetime_:
                return entry
        raise ValueError(f"No entry for {datetime_}")

    @at.register
    def _(self, datetime_like: str) -> Entry:
        datetime_ = parse_date(datetime_like)
        return self._at(datetime_)

    @at.register
    def _(self, datetime_: datetime.datetime) -> Entry:
        return self._at(datetime_)

    def group_by(self, what: GroupByTypes) -> dict[datetime.date, list[Entry]]:
        """
        Returns a dict of entries grouped by day with the keys as datetime.date objects, the values are lists of Entry objects.

        The entries are sorted by date in ascending order.
        """
        KEYMAP: dict[str, Callable[[Entry], datetime.date | datetime.datetime]] = {
            "day": lambda x: x.full_date.date(),
            "week": lambda x: (
                x.full_date - datetime.timedelta(days=x.full_date.weekday())
            ).date(),
            "month": lambda x: x.full_date.date().replace(day=1),
        }
        if what not in KEYMAP:
            raise ValueError(
                f'Invalid value for "what": {what}; expected one of {list(KEYMAP.keys())}'
            )
        return {
            day: list(entries)
            for day, entries in groupby(reversed(self.entries), key=KEYMAP[what])
        }

    def monthly_datasets(self) -> dict[datetime.date, "Dataset"]:
        """
        Returns a dict of Dataset objects grouped by month.
        """
        return {
            month: Dataset(_entries=entries[::-1])
            for month, entries in self.group_by("month").items()
        }

    def sub(
        self,
        condition: EntryCondition | None = None,
        *,
        include: IncludeExcludeActivities = set(),
        exclude: IncludeExcludeActivities = set(),
        note_contains: NoteCondition | None = None,
        predicate: EntryPredicate | None = None,
    ) -> "Dataset":
        """
        Returns a new Dataset object which is a subset of self
        with the entries filtered according to the arguments.

        Parameters:
            - condition: an EntryCondition object - only entries that satisfy this condition will be included
            - include: a string or a set of strings - only entries with at least one of these activities will be included
            - exclude: a string or a set of strings - only entries without any of these activities will be included
            - when: a datetime.date object, a string in the format dd.mm.yyyy or a slice with strings of that format - only entries on this day will be included
            - mood: a float or a set of floats - only entries with these moods will be included
            - note_contains: a string or an iterator of strings - only entries with notes matching this/these pattern(s) will be included
            - predicate: a function that takes an Entry object and returns a bool - only entries for which this function returns True will be included

        Returns: Dataset: a new Dataset object with the filtered entries
        """
        all_activities_set = self.activities().keys()
        if isinstance(include, str):
            include = {include}
        if isinstance(exclude, str):
            exclude = {exclude}
        if ua := (include - all_activities_set):
            raise ValueError(f"Unknown activities to include: {ua}")
        if ua := (exclude - all_activities_set):
            raise ValueError(f"Unknown activities to exclude: {ua}")
        return Dataset(
            _entries=[
                e
                for e in self
                if e.check_condition(
                    condition,
                    include,
                    exclude,
                    note_contains,
                    predicate,
                )
            ],
            _generating_entry_condition=condition,
            _parent_stats=self.stats(),
        )

    def mood(self) -> float:
        """
        Get the average mood among all entries
        """
        return mean(e.mood for e in self) if len(self) > 0 else 0.0

    def std(self) -> float:
        """
        Get the standard deviation of the mood among all entries
        """
        return stdev(e.mood for e in self) if len(self) > 1 else 0.0

    def mood_std(self) -> MoodStd:
        """
        Get the average mood and its standard deviation among all entries
        """
        return MoodStd(self.mood(), self.std())

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

    def head(
        self, n: int = 5, *, file: TextIOWrapper | None = None, verbose: bool = False
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
            n = len(self)
        for e in islice(self, n):
            print(e if not verbose else e.verbose(), file=file)
        if len(self) > n:
            print(
                f"    and {len(self) - n} more entries...",
                file=file,
            )

    def mood_with_without(self, activity: str) -> MoodWithWithout:
        df_with = self.sub(include=activity)
        df_without = self.sub(exclude=activity)
        return MoodWithWithout(
            df_with.stats().mood,
            df_without.stats().mood,
            len(df_with),
            len(df_without),
        )

    def stats(self) -> StatsResult:
        """
        Returns the following statistics:
            - mood (avg ± std)
            - median note length [num symbols]
            - entries frequency [entries per day] (median)
            - total number of activities
        as a StatsResult object.
        """
        note_lengths = [len(e.note) for e in self]
        timedeltas_secs = [
            max(1.0, (d1 - d2).total_seconds())
            for d1, d2 in pairwise(self.get_datetimes())
        ]
        return StatsResult(
            mood=self.mood_std(),
            note_length=median(note_lengths),
            entries_frequency=24 * 60 * 60 / median(timedeltas_secs)
            if len(timedeltas_secs) > 1
            else None,
            number_of_activities=sum(len(e.activities) for e in self),
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
            if num < n_threshold:
                continue
            mood_with_without = self.mood_with_without(act)
            res.append(
                CompleteAnalysis(
                    act,
                    mood_with_without,
                    num,
                )
            )
        res.sort(key=lambda x: x.mood_with_without.calc_change(), reverse=True)
        return res

    def build_tags(self) -> defaultdict[str, list[Tag]]:
        """
        Returns all tags grouped by their names.
        """
        tags = defaultdict(list)
        for entry in self:
            for t in entry._tags:
                tags[t.tag].append(t)
        return tags

    def get_book_tags(self) -> list[BookTag]:
        return [BookTag.from_tag(tag) for tag in self.build_tags().get("книга", [])]

    # plots:

    def mood_plot(self, by: GroupByTypes = "day") -> go.Figure:
        """
        Generates a plot of the average, maximum, and minimum moods over time.
        (the area between the maximum and minimum moods is filled)

        Returns: go.Figure: The plotly figure object representing the mood plot.
        """
        return Plotter.mood_plot(self, by)

    def by_time_bar_plot(
        self,
        what: Literal["weekday", "hour", "day", "month"],
        swap_freq_mood: bool = False,
    ) -> go.Figure:
        """
        Groups entries by weekday, hour or day and returns a bar chart.
        The color of the bars represents the number of entries,
        the height of the bars represents the average mood.

        what: 'weekday', 'hour', 'day' or 'month' - what to group by
        swap_freq_mood: if True, the frequency and mood will be swapped in the bar chart

        Returns: go.Figure: The plotly figure object representing the bar chart.
        """
        return Plotter.by_time_bar_plot(self, what, swap_freq_mood)

    def mood_change_activity(self, *activity: str) -> go.Figure:
        return Plotter.mood_change_activity(self, *activity)

    def activities_effect_on_mood(self, n_threshold: int = 10) -> go.Figure:
        return Plotter.activities_effect_on_mood(self, n_threshold)

    def entries_differences(self) -> go.Figure:
        return Plotter.entries_differences(self)

    def note_length_plot(self, groupby: GroupByTypes = "week") -> go.Figure:
        """
        Generates a line plot showing the average note lengths vs date.
        
        Returns:
            go.Figure: A plotly figure object representing the line plot.
        """
        return Plotter.note_length_plot(self, groupby)

    def books_read_plot(
        self,
        *,
        full: bool = False,
        groupby: Literal["date", "month"] = "date",
    ) -> go.Figure:
        return Plotter.books_read(self, full, groupby)

    def show_calendar_plot(
        self,
        years: list[int] | None = None,
        cmap: str = "RdYlGn",
        colorbar: bool = False,
    ) -> None:
        """
        Displays a calendar heatmap of daily average mood values.

        This function generates a heatmap using `calplot` to visualize daily mood data over time.
        It supports filtering by specific years and allows customization of the colormap and colorbar.

        Args:
            df : Dataset
                A dataset containing mood entries, where each entry has a `mood` attribute and is grouped by day.
            years : list[int] | None, optional
                A list of years to include in the plot. If None (default), all years are included.
            cmap : str, optional
                The colormap used for the heatmap. Defaults to `"RdYlGn"` (Yellow-Green).
                Also consider: `["afmhot", "YlGn"]`
            colorbar : bool, optional
                Whether to display the colorbar. Defaults to False.

        Notes:
            - Missing days are correctly handled (i.e., not plotted as zero).
            - Mood values are expected to range between `vmin=1.0` and `vmax=6.0`.
            - Days with a mood value of exactly zero are dropped from the plot.
        """
        return Plotter.show_calendar_plot(self, years, cmap, colorbar)

    def generate_activity_correlation_matrix(self) -> None:
        """
        Generates a correlation matrix for the activities in the dataset.

        Saves the matrix as an html interactive plot.
        """
        return Plotter.generate_activity_correlation_matrix(self)

    def people_frequency(self) -> go.Figure:
        """
        Generates a bar plot of the frequency of people in the dataset.
        """
        return Plotter.people_frequency(self)
    
    def plot_wordcloud(self, additional_stopwords: set[str] = set(), n_threshold: int = 3) -> None:
        return Plotter.plot_wordcloud(self, additional_stopwords, n_threshold)


if __name__ == "__main__":
    # run `python -i dataset.py` to use in the terminal
    DATA_DIR = pathlib.Path("data")
    path = next(DATA_DIR.glob("*.csv"))
    print("using file", path.name)

    df = Dataset(csv_file=path)
