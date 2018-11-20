import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from src.exploration.core.cohort import Cohort
from src.exploration.core.decorators import title, xlabel, ylabel
from src.exploration.stats.grouper import agg, event_duration_agg

registry = []


def register(f):
    registry.append(f)
    return f


@register
@xlabel("Number of Days")
@ylabel("Count (log)")
@title("distribution per month")
def plot_duration_distribution_per_day_as_line(figure: Figure, cohort: Cohort) -> Figure:
    assert cohort.is_duration_events()

    df = event_duration_agg(cohort, "count").sort_values("duration")
    ax = figure.gca()
    ax.plot(df.duration, df["count(1)"], color="salmon")
    ax.set_yscale('log')
    return figure


@register
@ylabel("Number of Months")
@xlabel("Count")
@title("distribution per month")
def plot_duration_distribution_per_month_as_bar(figure: Figure, cohort: Cohort) -> Figure:
    assert cohort.is_duration_events()

    df = event_duration_agg(cohort, "count").sort_values("duration")
    df.duration = np.ceil(df.duration / 30)
    df.duration = df.duration.astype('int32')
    df = df.groupby("duration").sum().reset_index()
    ax = figure.gca()
    sns.barplot(data=df, x="duration", y="count(1)", color="salmon", ax=ax)
    return figure


@register
@ylabel("Value")
@xlabel("Duration (in days)")
@title("mean duration per value")
def plot_mean_duration_per_value(figure: Figure, cohort: Cohort) -> Figure:
    assert cohort.is_duration_events()

    df = agg(cohort.events, frozenset(["value"]), "mean").sort_values("value")
    ax = figure.gca()
    ax.barh(y=range(len(df.value)), width=df["avg(duration)"].values)
    ax.set_yticklabels(df.value.values)
    ax.set_yticks(range(len(df.value)))
    return figure
