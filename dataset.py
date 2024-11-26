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

import plotly.express as px
import plotly.graph_objs as go
import numpy as np

from entry import Entry, EntryPredicate
from tag import Tag
from utils import (
    WEEKDAYS,
    MONTHS,
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
            print("...", file=file)

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

    def _tag_book(self, requested_tags: list[Tag]) -> None:
        _PS = r"{}"
        for t in requested_tags:
            print(f"[{t.title}]:\n{_PS[0]}{t.body}{_PS[1]}\n")

    def analyse_special_tag(self, tag: Literal["prediction", "книга"]):
        tags = self.build_tags()
        requested_tags = tags.get(tag)
        if requested_tags is None or not requested_tags:
            raise ValueError(f"No tags of type {tag!r}")
        if tag == "prediction":
            # group prediction-pairs by title (id)
            self._tag_prediction(requested_tags)
        elif tag == "книга":
            self._tag_book(requested_tags)

    # plots:

    def mood_plot(self, by: Literal["day", "week", "month"] = "day") -> go.Figure:
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
            this_day_moods_mean = mean(this_day_moods)
            this_day_moods_std = stdev(this_day_moods) if len(this_day_moods) > 1 else 0
            avg_moods.append(this_day_moods_mean)
            max_moods.append(this_day_moods_mean + this_day_moods_std)
            min_moods.append(this_day_moods_mean - this_day_moods_std)

        fig = go.Figure(
            [
                go.Scatter(
                    name="avg",
                    x=groups,
                    y=avg_moods,
                    # color depends on how many entries there are on that day
                    marker=dict(
                        color=[len(day) for day in dd.values()],
                        colorscale="Bluered",
                        showscale=True,
                        colorbar=dict(title="Number of entries"),
                        size=10 if by == "month" else 5,
                    ),
                    mode="lines+markers",
                    line=dict(color="rgb(31, 119, 180)", shape="spline"),
                ),
                go.Scatter(
                    name="avg+std",
                    x=groups,
                    y=max_moods,
                    mode="lines",
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=False,
                ),
                go.Scatter(
                    name="avg-std",
                    x=groups,
                    y=min_moods,
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    mode="lines",
                    fillcolor="rgba(130, 130, 130, 0.45)",
                    fill="tonexty",
                    showlegend=False,
                ),
            ]
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Mood",
            hovermode="x",
            showlegend=False,
            template="plotly_dark",
        )
        fig.update_yaxes(
            rangemode="tozero",
        )
        return fig

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
        FUNC_MAP: dict[str, Callable[[datetime.datetime], int]] = {
            "weekday": datetime.datetime.weekday,
            "hour": lambda x: x.hour,
            "day": lambda x: x.day,
            "month": lambda x: x.month,
        }
        if what not in FUNC_MAP:
            raise ValueError(
                f'Invalid value for "what": {what}; expected one of {list(FUNC_MAP.keys())}'
            )
        mood_by: defaultdict[int, list[float]] = defaultdict(list)
        for entry in self:
            mood_by[FUNC_MAP[what](entry.full_date)].append(entry.mood)
        AVG_MOOD = list(map(mean, mood_by.values()))
        FREQ = list(map(len, mood_by.values()))
        fig = px.bar(
            x=mood_by.keys(),
            y=AVG_MOOD if not swap_freq_mood else FREQ,
            color=FREQ if not swap_freq_mood else AVG_MOOD,
            color_continuous_scale="viridis",
            labels={
                "x": what.title(),
                "y": "Mood" if not swap_freq_mood else "Number of entries",
                "color": "Number of entries" if not swap_freq_mood else "Mood",
            },
            title=f"Mood by {what}",
        )
        fig.update_layout(
            template="plotly_dark",
            xaxis={"dtick": 1},
        )

        if what in {"weekday", "month"}:
            fig.update_xaxes(
                tickmode="array",
                tickvals=list(range(7)) if what == "weekday" else list(range(1, 13)),
                ticktext=WEEKDAYS if what == "weekday" else MONTHS,
            )
        return fig

    def mood_change_activity(self, activity: str) -> go.Figure:
        dates = []
        with_ = []
        without_ = []
        errors_with_ = []
        errors_without_ = []
        no_activity_in: list[datetime.date] = []
        for month, df_month in self.monthly_datasets().items():
            try:
                with_without = df_month.mood_with_without(activity)
                without_.append(with_without.without.mood)
                with_.append(with_without.with_.mood)
                errors_with_.append(with_without.with_.std)
                errors_without_.append(with_without.without.std)
                dates.append(month)
            except ValueError:
                no_activity_in.append(month)

        if not dates:
            raise ValueError(f"No activity {activity!r} in the dataset")

        if no_activity_in:
            print(
                f"No {activity!r} in {', '.join(month.strftime('%B %Y') for month in no_activity_in)}"
            )

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=dates,
                    y=with_,
                    name="with",
                    mode="lines+markers",
                    marker=dict(color="#97F66B", symbol="cross-dot", size=10),
                    line=dict(shape="spline", smoothing=1.3, dash="dash"),
                ),
                go.Scatter(
                    x=dates,
                    y=without_,
                    name="without",
                    mode="lines+markers",
                    marker=dict(color="#FF522D", symbol="x-dot", size=10),
                    line=dict(shape="spline", smoothing=1.3, dash="dash"),
                ),
            ]
        )
        fig.update_layout(
            template="plotly_dark",
            title=f"Mood with and without {activity!r}",
            xaxis=dict(title="Month"),
            yaxis=dict(title="Mood"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        return fig

    def entries_differences(self) -> go.Figure:
        dts = self.get_datetimes()[::-1]
        diffs = [dt2 - dt1 for dt1, dt2 in pairwise(dts)]
        fig = go.Figure(
            data=go.Scatter(
                x=dts[1:],
                y=[el.total_seconds() / (3600 * 24) for el in diffs],
                mode="lines+markers",
                line=dict(dash="dashdot"),
            )
        )
        if len(dts) > 100:
            window_size = len(dts) // 40
            sliding_average = [
                mean(
                    [
                        el.total_seconds() / (3600 * 24)
                        for el in diffs[i : i + window_size]
                    ]
                )
                for i in range(len(diffs) - window_size + 1)
            ]
            fig.add_trace(
                go.Scatter(
                    x=dts[window_size - 1 :],
                    y=sliding_average,
                    mode="lines",
                    name=f"Sliding average (window size {window_size})",
                    marker=dict(size=0),
                    line=dict(width=2),
                )
            )
            # legend
            fig.update_layout(
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                )
            )
        fig.update_layout(
            yaxis_title="difference, days",
            title="Difference between consecutive entries in days",
            template="plotly_dark",
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
        for day, entries in self.group_by("day").items():
            tmp = []
            for entry in entries:
                tmp.append(len(entry.note))
            day_to_total_note_len[day] = mean(tmp)

        window_size = 11
        import numpy as np

        sliding_average = np.convolve(
            list(day_to_total_note_len.values()),
            np.ones(window_size) / window_size,
            mode="valid",
        )

        fig = px.scatter(
            x=day_to_total_note_len.keys(),
            y=day_to_total_note_len.values(),
            labels={"x": "Date", "y": "Average note length"},
            title="Average note length",
        )
        fig.add_trace(
            go.Scatter(
                x=list(day_to_total_note_len.keys())[window_size - 1 :],
                y=sliding_average,
                mode="lines",
                name=f"Sliding average (window size {window_size})",
                marker=dict(size=0),
                line=dict(width=2),
            )
        )
        fig.update_layout(
            template="plotly_dark",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        return fig

    def generate_activity_correlation_matrix(self) -> None:
        def corr(a1: str, a2: str) -> float:
            has_a1 = self.sub(include=a1)
            try:
                has_both = has_a1.sub(include=a2)
            except ValueError:
                return 0.0
            return len(has_both) / len(has_a1)

        activities = self.activities()
        act = list(activities.keys())
        corr_mat = np.zeros((len(act), len(act)))

        for i, a1 in enumerate(act):
            for j, a2 in enumerate(act):
                if a1 == a2:
                    corr_mat[i, j] = 1.0
                else:
                    corr_mat[i, j] = corr(a1, a2)
        import plotly.graph_objects as go

        fig = go.Figure(data=go.Heatmap(z=corr_mat, x=act, y=act, colorscale="Viridis"))

        fig.update_layout(
            autosize=False,
            width=1200,
            height=1200,
            margin=dict(l=50, r=50, b=100, t=100, pad=4),
        )

        fig.write_html("corr.html")


if __name__ == "__main__":
    # run `python -i dataset.py` to use in the terminal
    DATA_DIR = pathlib.Path("data")
    path = next(DATA_DIR.glob("*.csv"))
    print("using file", path.name)

    df = Dataset(csv_file=path)
