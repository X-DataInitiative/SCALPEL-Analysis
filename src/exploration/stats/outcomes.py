import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from src.exploration.core.cohort import Cohort
from src.exploration.core.decorators import title, xlabel, ylabel

registry = []


def register(f):
    registry.append(f)
    return f


def _prepare_data(outcomes: Cohort) -> pd.DataFrame:
    data = outcomes.events.groupBy("start").count().toPandas().sort_values("start")
    data = data.set_index("start")
    idx = pd.date_range(data.index.min(), data.index.max())
    data.index = pd.DatetimeIndex(data.index, name="start")
    return data.reindex(idx, fill_value=0).reset_index()


def _plot_outcomes_per_day(outcomes: Cohort, figure: Figure, time_series):
    data = outcomes.events.groupBy("start").count().toPandas().sort_values("start")
    ax = figure.gca()
    data = data.set_index("start")
    idx = pd.date_range(data.index.min(), data.index.max())
    data.index = pd.DatetimeIndex(data.index)
    data = data.reindex(idx, fill_value=0).reset_index()
    if time_series:
        data.groupby([data["index"].dt.year, data["index"].dt.month, ]).sum()[
            "count"].plot(
            kind="line", color=sns.xkcd_rgb["pumpkin orange"])
    else:

        data.groupby([data["index"].dt.year, data["index"].dt.month, ]).sum()[
            "count"].plot(
            kind="bar", color=sns.color_palette(palette="Paired", n_colors=12))
    return ax


@register
@title("Time series of outcomes")
@xlabel("Day")
@ylabel("Count")
def plot_outcomes_per_day_time_series(figure: Figure, cohort: Cohort) -> Figure:
    data = _prepare_data(cohort)
    ax = figure.gca()
    ax.plot(data["index"], data["count"], color=sns.xkcd_rgb["pumpkin orange"])
    return figure


@register
@title("Time series of outcomes per month")
@xlabel("Month")
@ylabel("Count")
def plot_outcomes_per_month_time_series(figure: Figure, cohort: Cohort) -> Figure:
    data = _prepare_data(cohort)
    data = data.groupby([data["index"].dt.year, data["index"].dt.month, ]).sum()["count"]
    data.plot(kind="line", color=sns.xkcd_rgb["pumpkin orange"])
    return figure


@register
@title("Time series of outcomes per week")
@xlabel("Week")
@ylabel("Count")
def plot_outcomes_per_week_time_series(figure: Figure, cohort: Cohort) -> Figure:
    data = _prepare_data(cohort)
    data = data.groupby([data["index"].dt.year,
                         data["index"].dt.week]).sum()["count"]
    data.plot(kind="line", color=sns.xkcd_rgb["pumpkin orange"])
    return figure


@register
@title("Outcomes per day")
@xlabel("Day")
@ylabel("Count")
def plot_outcomes_per_day_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    data = _prepare_data(cohort)
    ax = figure.gca()
    sns.barplot(x=data.index, y=data["count"], ax=ax,
                color=sns.xkcd_rgb["pumpkin orange"])
    return figure


@register
@title("Outcomes per week")
@xlabel("Week")
@ylabel("Count")
def plot_outcomes_per_week_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    data = _prepare_data(cohort)
    ax = figure.gca()
    data = data.groupby([data["index"].dt.year,
                         data["index"].dt.week]).sum().sort_index()
    sns.barplot(x=data.index, y=data["count"], ax=ax,
                color=sns.xkcd_rgb["pumpkin orange"])
    for i, label in enumerate(ax.xaxis.get_ticklabels()):
        if i % 10 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)
    return figure


@register
@title("Outcomes per month")
@xlabel("Month")
@ylabel("Count")
def plot_outcomes_per_month_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    data = _prepare_data(cohort)
    ax = figure.gca()
    data = data.groupby([data["index"].dt.year,
                         data["index"].dt.month]).sum().sort_index()
    sns.barplot(x=data.index, y=data["count"], ax=ax, color=sns.xkcd_rgb["pumpkin orange"])
    for i, label in enumerate(ax.xaxis.get_ticklabels()):
        if i % 4 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)
    return figure
