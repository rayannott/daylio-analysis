import datetime
import itertools
import textwrap
from typing import Literal, Callable, TYPE_CHECKING
from statistics import mean, stdev
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.utils import WEEKDAYS, MONTHS

if TYPE_CHECKING:
    from src.dataset import Dataset


class Plotter:
    @staticmethod
    def mood_plot(
        df: "Dataset", by: Literal["day", "week", "month"] = "day"
    ) -> go.Figure:
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
        errors_with_ = []
        errors_without_ = []
        no_activity_in: list[datetime.date] = []
        for month, df_month in df.monthly_datasets().items():
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
    def note_length_plot(df: "Dataset") -> go.Figure:
        """
        Generates a line plot showing the average note lengths vs date.

        Args:
            cap_length (int, optional): If not -1, the length of each note is capped at this value.
                If a note is longer than cap_length, its length is set to cap_length. Default is -1.

        Returns:
            go.Figure: A plotly figure object representing the line plot.

        """
        day_to_total_note_len = defaultdict(float)
        for day, entries in df.group_by("day").items():
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
    def books_read(df: "Dataset") -> go.Figure:
        book_tags = df.get_book_tags()

        # repeat index for each book on the same date
        x_ind = list(
            (
                i
                for i, (_, book_group) in enumerate(
                    itertools.groupby(book_tags, key=lambda book: book.full_date.date())
                )
                for _ in book_group
            )
        )
        x_labels = [book.full_date.strftime("%d.%m.%Y") for book in book_tags]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=x_ind,
                y=[book.number_of_pages for book in book_tags],
                marker=dict(
                    color=[book.rating for book in book_tags],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Rating"),
                    cmax=10.0,
                    cmin=1.0,
                ),
                textfont=dict(size=18),
                text=[book.title for book in book_tags],
                hovertemplate="<b>%{text}</b><br>Rating: %{marker.color:.1f}<br>Number of pages: %{y} <extra>%{customdata}</extra>",
                customdata=[
                    "<br>".join(textwrap.wrap(book.body, width=40))
                    for book in book_tags
                ],
            )
        )

        fig.update_layout(
            title="Books",
            barmode="stack",
            xaxis_title="Date",
            yaxis_title="Number of pages",
            xaxis=dict(tickvals=x_ind, ticktext=x_labels),
            margin=dict(l=10, r=10, t=45, b=10),
            template="plotly_dark",
        )

        return fig
