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

import plotly.graph_objs as go

from src.entry import Entry, EntryPredicate
from src.tag import Tag, BookTag
from src.plotting import Plotter
from src.utils import (
    DT_FORMAT_SHOW,
    DATE_FORMAT_SHOW,
    IncludeExcludeActivities,
    MoodCondition,
    NoteCondition,
    datetime_from_now,
    date_slice_to_entry_predicate,
    StatsResult,
    CompleteAnalysis,
    MoodWithWithout,
    MoodStd,
)

REMOVE: set[str] = set(
    json.load(open(pathlib.Path("data") / "to_remove.json", "r", encoding="utf-8-sig"))
)

BAD_MOOD = {1.0, 2.0, 2.5}
AVERAGE_MOOD = {3.0, 3.5, 4.0}
GOOD_MOOD = {5.0, 6.0}

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
    ) -> None:
        """
        Construct a Dataset object from a CSV file exported from the app.
        """
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
        latest_entry_full_date = self.entries[0].full_date
        return f"Dataset({len(self.entries)} entries; last [{datetime_from_now(latest_entry_full_date)}]; mood: {self.mood_std()})"

    def __getitem__(self, _date: str | slice) -> "Dataset":
        """
        Returns a new Dataset object which is a subset of self
        with the entries filtered according to the date or date range as a slice.
        """
        if isinstance(_date, slice):
            CHECK_FN = date_slice_to_entry_predicate(_date)
        else:
            date = datetime.datetime.strptime(_date, DATE_FORMAT_SHOW).date()
            CHECK_FN: EntryPredicate = lambda e: e.full_date.date() == date  # noqa: E731
        return Dataset(_entries=[e for e in self if CHECK_FN(e)])

    def __iter__(self) -> Iterator[Entry]:
        return iter(self.entries)

    def __matmul__(self, datetime_like: str | datetime.datetime) -> Entry:
        """
        Wrapper for the `at` method.
        """
        return self.at(datetime_like)

    def __len__(self) -> int:
        return len(self.entries)

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

    def at(self, datetime_like: str | datetime.datetime) -> Entry:
        """
        Returns the entry for a particular datetime-like object.

        datetime_str: a string in the format dd.mm.yyyy HH:MM

        This is used when calling the Dataset object as a function.
        """
        if isinstance(datetime_like, str):
            if DATETIME_PATTERN.fullmatch(datetime_like):
                datetime_ = datetime.datetime.strptime(datetime_like, DT_FORMAT_SHOW)
            else:
                raise ValueError(
                    f"Invalid date string: {datetime_like}; expected format: dd.mm.yyyy or dd.mm.yyyy HH:MM"
                )
        elif isinstance(datetime_like, datetime.datetime):
            datetime_ = datetime_like
        else:
            raise ValueError(
                f"Invalid type for datetime_like: {type(datetime_like)}; expected str or datetime.datetime"
            )
        for entry in self.entries:
            if entry.full_date == datetime_:
                return entry
        raise ValueError(f"No entry for {datetime_}")

    def group_by(
        self, what: Literal["day", "week", "month"]
    ) -> dict[datetime.date, list[Entry]]:
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
        *,
        include: IncludeExcludeActivities = set(),
        exclude: IncludeExcludeActivities = set(),
        mood: MoodCondition | None = None,
        note_contains: NoteCondition | None = None,
        predicate: EntryPredicate | None = None,
    ) -> "Dataset":
        """
        Returns a new Dataset object which is a subset of self
        with the entries filtered according to the arguments.

        Parameters:
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
                if e.check_condition(include, exclude, mood, note_contains, predicate)
            ]
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
        return MoodWithWithout(df_with.stats().mood, df_without.stats().mood)

    def stats(self) -> StatsResult:
        """
        Returns the following statistics:
            - mood (avg ± std)
            - note length [num symbols] (avg ± std)
            - entries frequency [entries per day] (median)
        as a StatsResult object.
        """
        note_lengths = [len(e.note) for e in self]
        timedeltas_secs = [
            max(1.0, (d1 - d2).total_seconds())
            for d1, d2 in pairwise(self.get_datetimes())
        ]
        median_timedelta = median(timedeltas_secs)
        return StatsResult(
            mood=self.mood_std(),
            note_length=(mean(note_lengths), stdev(note_lengths)),
            entries_frequency=24 * 60 * 60 / median_timedelta,
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
        return [BookTag(**tag.__dict__) for tag in self.build_tags().get("книга", [])]

    def show_tags_stats(self):
        # TODO
        raise NotImplementedError

    def _tag_prediction(self, requested_tags: list[Tag]) -> None:
        prediction_pairs: defaultdict[str, list[Tag]] = defaultdict(list)
        for t in requested_tags:
            prediction_pairs[t.title].append(t)
        predictions: list[tuple[Tag, Tag | None]] = []
        for title, pair in prediction_pairs.items():
            if len(pair) == 1:
                predictions.append((pair[0], None))
                continue
            if len(pair) > 2:
                print(f"warning: more than 2 prediction-tags for {title!r}")
            predictions.append((pair[0], pair[1]))

        for pred1, pred2 in predictions:
            print(
                f"{pred1.body} -> {pred2.body if pred2 else '?'} ({pred2 and 'true' in pred2.body})"
            )

    # plots:

    def mood_plot(self, by: Literal["day", "week", "month"] = "day") -> go.Figure:
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

    def mood_change_activity(self, activity: str) -> go.Figure:
        return Plotter.mood_change_activity(self, activity)

    def entries_differences(self) -> go.Figure:
        return Plotter.entries_differences(self)

    def note_length_plot(self) -> go.Figure:
        """
        Generates a line plot showing the average note lengths vs date.

        Args:
            cap_length (int, optional): If not -1, the length of each note is capped at this value.
                If a note is longer than cap_length, its length is set to cap_length. Default is -1.

        Returns:
            go.Figure: A plotly figure object representing the line plot.

        """
        return Plotter.note_length_plot(self)
    
    def books_read_plot(self) -> go.Figure:
        return Plotter.books_read(self)

    def generate_activity_correlation_matrix(self) -> None:
        """
        Generates a correlation matrix for the activities in the dataset.

        Saves the matrix as an html interactive plot.
        """
        return Plotter.generate_activity_correlation_matrix(self)


if __name__ == "__main__":
    # run `python -i dataset.py` to use in the terminal
    DATA_DIR = pathlib.Path("data")
    path = next(DATA_DIR.glob("*.csv"))
    print("using file", path.name)

    df = Dataset(csv_file=path)
