import datetime
import itertools
import textwrap
from math import sqrt
from typing import Literal, Callable, TYPE_CHECKING
from statistics import mean, stdev, median
from collections import defaultdict

import calplot
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.utils import WEEKDAYS, MONTHS, GroupByTypes

if TYPE_CHECKING:
    from src.dataset import Dataset


class Plotter:
    @staticmethod
    def mood_plot(df: "Dataset", by: GroupByTypes = "day") -> go.Figure:
        dd = df.group_by(by)
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

    @staticmethod
    def by_time_bar_plot(
        df,
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
        for entry in df:
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

    @staticmethod
    def mood_change_activity(df: "Dataset", activity: str) -> go.Figure:
        dates = []
        with_ = []
        without_ = []
        n_entries_with_ = []
        n_entries_without = []
        errors_with_ = []
        errors_without = []
        no_activity_in: list[datetime.date] = []
        for month, df_month in df.monthly_datasets().items():
            try:
                with_without = df_month.mood_with_without(activity)
                without_.append(with_without.without.mood)
                with_.append(with_without.with_.mood)
                n_entries_with_.append(with_without.n_entries_with_)
                n_entries_without.append(with_without.n_entries_without)
                errors_with_.append(
                    with_without.with_.std / sqrt(with_without.n_entries_with_)
                )
                errors_without.append(
                    with_without.without.std / sqrt(with_without.n_entries_without)
                )
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
                    error_y=dict(type="data", array=errors_with_),
                    name="with",
                    mode="lines+markers",
                    marker=dict(color="#97F66B", symbol="cross-dot", size=10),
                    line=dict(shape="spline", smoothing=1.3, dash="dash"),
                    customdata=n_entries_with_,
                    hovertemplate="%{y}<extra>%{customdata} entries</extra>",
                ),
                go.Scatter(
                    x=dates,
                    y=without_,
                    error_y=dict(type="data", array=errors_without),
                    name="without",
                    mode="lines+markers",
                    marker=dict(color="#FF522D", symbol="x-dot", size=10),
                    line=dict(shape="spline", smoothing=1.3, dash="dash"),
                    customdata=n_entries_without,
                    hovertemplate="%{y}<extra>%{customdata} entries</extra>",
                ),
            ]
        )
        fig.update_layout(
            template="plotly_dark",
            title=f"Mood with and without {activity!r}; mean ± se",
            xaxis=dict(title="Month"),
            yaxis=dict(title="Mood"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        return fig

    @staticmethod
    def entries_differences(df: "Dataset") -> go.Figure:
        dts = df.get_datetimes()[::-1]
        diffs = [dt2 - dt1 for dt1, dt2 in itertools.pairwise(dts)]
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

    @staticmethod
    def note_length_plot(df: "Dataset", groupby: GroupByTypes) -> go.Figure:
        """
        Generates a line plot showing the average note lengths vs date.

        Args:
            cap_length (int, optional): If not -1, the length of each note is capped at this value.
                If a note is longer than cap_length, its length is set to cap_length. Default is -1.

        Returns:
            go.Figure: A plotly figure object representing the line plot.

        """

        def se(data: list[int]) -> float:
            return stdev(data) / len(data) if len(data) > 1 else 0

        time_to_note_lens = {
            time_per: [len(entry.note) for entry in entries]
            for time_per, entries in df.group_by(groupby).items()
        }

        x_vals = list(time_to_note_lens.keys())
        y_vals = [mean(time_per) for time_per in time_to_note_lens.values()]
        medians = [median(time_per) for time_per in time_to_note_lens.values()]
        stdevs = [se(time_per) for time_per in time_to_note_lens.values()]
        n_entries = [len(time_per) for time_per in time_to_note_lens.values()]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                error_y=dict(type="data", array=stdevs, visible=True),
                mode="markers+lines",
                line_shape="spline",
                customdata=n_entries,
                hovertemplate="Date: %{x}<br>Average note length: %{y}<br>Number of entries: %{customdata}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=medians,
                mode="markers",
                marker=dict(color="red", size=7, symbol="x"),
                name="median",
            )
        )
        fig.update_layout(template="plotly_dark")
        return fig

    @staticmethod
    def generate_activity_correlation_matrix(df: "Dataset") -> None:
        def corr(a1: str, a2: str) -> float:
            has_a1 = df.sub(include=a1)
            try:
                has_both = has_a1.sub(include=a2)
            except ValueError:
                return 0.0
            return len(has_both) / len(has_a1)

        activities = df.activities()
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

    @staticmethod
    def people_frequency(df: "Dataset") -> go.Figure:
        ppl, frequencies = zip(*df.people().most_common())
        fig = go.Figure(data=[go.Bar(x=ppl, y=frequencies)])
        fig.update_layout(
            title="People Frequencies",
            xaxis_tickangle=45,
            margin=dict(l=10, r=10, t=40, b=10),
            template="plotly_dark",
        )
        return fig

    @staticmethod
    def books_read(
        df: "Dataset",
        full: bool,
        groupby: Literal["date", "month"],
    ) -> go.Figure:
        book_tags = df.get_book_tags()[::-1]

        def groupby_func(full_date: datetime.datetime) -> datetime.date:
            if groupby == "date":
                return full_date.date()
            if groupby == "month":
                return full_date.replace(day=1).date()
            raise ValueError(f"Invalid value for 'groupby': {groupby}")

        # repeat index for each book on the same date
        x_ind = list(
            (
                i
                for i, (_, book_group) in enumerate(
                    itertools.groupby(
                        book_tags, key=lambda book: groupby_func(book.full_date)
                    )
                )
                for _ in book_group
            )
        )
        x_labels = [
            book.full_date.strftime("%d.%m.%Y" if groupby == "date" else "%m.%Y")
            for book in book_tags
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=x_ind,
                y=[book.number_of_pages for book in book_tags],
                marker=dict(
                    color=[(book.rating or 5.5) for book in book_tags],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Rating"),
                    cmax=10.0,
                    cmin=1.0,
                    line=dict(
                        width=[3 if book.rating is None else 1 for book in book_tags],
                    ),
                ),
                textfont=dict(size=18),
                text=[book.title for book in book_tags],
                hovertemplate="<b>%{text}</b><br>%{customdata[0]}Rating: %{marker.color:.1f}<br>Number of pages: %{y} <extra>%{customdata[1]}</extra>",
                customdata=[
                    (
                        f"<i>{book.author}</i><br>" if book.author else "",  # author
                        "<br>".join(textwrap.wrap(book.body_clean, width=40)),  # body
                    )
                    for book in book_tags
                ],
            )
        )
        # if full:  # TODO: make this work when more data
        #     _slice = slice(None, None, len(x_ind) // 6)
        #     x_ind = x_ind[_slice]
        #     x_labels = x_labels[_slice]

        fig.update_layout(
            barmode="stack",
            xaxis_title="Date" if groupby == "date" else "Month",
            yaxis_title="Number of pages",
            xaxis=dict(
                tickvals=x_ind,
                ticktext=x_labels,
                tickangle=45,
                range=[x_ind[-min(len(x_ind) - 1, 12)], x_ind[-1]]
                if not full
                else None,
                fixedrange=full,
            ),
            yaxis=dict(fixedrange=True),
            dragmode="pan",
            margin=dict(l=10, r=10, t=10, b=10),
            template="plotly_dark",
        )

        return fig

    @staticmethod
    def show_calendar_plot(
        df: "Dataset",
        years: list[int] | None = None,
        cmap: str = "YlGn",
        colorbar: bool = False,
    ) -> None:
        def include_day(day: datetime.date) -> bool:
            return years is None or day.year in years

        daily_avg_mood = {
            day: mean(e.mood for e in entries)
            for day, entries in df.group_by("day").items()
            if include_day(day)
        }

        index = pd.to_datetime(list(daily_avg_mood.keys()))
        values = pd.Series(list(daily_avg_mood.values()), index=index)

        _ = calplot.calplot(
            values,
            cmap=cmap,
            colorbar=colorbar,
            yearlabel_kws={"fontname": "sans-serif", "fontsize": 14},
            dropzero=True,
            vmin=1.0,
            vmax=6.0,
        )
