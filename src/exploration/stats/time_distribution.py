import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.exploration.core.cohort import Cohort
from src.exploration.core.decorators import xlabel, ylabel


def _prepare_data(outcomes: Cohort) -> pd.DataFrame:
    data = outcomes.events.groupBy("start").count().toPandas().sort_values("start")
    data = data.set_index(pd.DatetimeIndex(data.start, tz="Europe/Paris", ambiguous=True,
                                           name="index")).sort_index()
    return data.reset_index()


def _per_month(data: pd.DataFrame) -> pd.DataFrame:
    return data.groupby([data["index"].dt.year, data["index"].dt.month]).sum()["count"]


def _per_week(data: pd.DataFrame) -> pd.DataFrame:
    return data.groupby([data["index"].dt.year, data["index"].dt.week]).sum()["count"]


def _per_day(data: pd.DataFrame) -> pd.DataFrame:
    return data.groupby(data["index"]).sum()["count"]


def _prepare_data_2(cohort: Cohort, date_unit: str) -> pd.DataFrame:
    data = cohort.events.groupBy("start").count().toPandas().sort_values("start")
    data = data.set_index(pd.DatetimeIndex(data.start, tz="Europe/Paris", ambiguous=True,
                                           name="index")).sort_index().reset_index()
    if date_unit == "day":
        return _per_day(data)
    elif date_unit == "month":
        return _per_month(data)
    elif date_unit == "week":
        return _per_week(data)
    else:
        raise ValueError("Wrong date unit {}. day, month, week only.".format(date_unit))


def _plot_bars(data: pd.DataFrame, ax: Axes) -> Axes:
    ax.bar(range(len(data.index)), data.values, color=sns.xkcd_rgb["pumpkin orange"],
           tick_label=data.index)
    return ax


def _plot_line(data: pd.DataFrame, ax: Axes) -> Axes:
    data.plot(kind="line", color=sns.xkcd_rgb["pumpkin orange"])
    return ax


@xlabel("Month")
@ylabel("Count")
def plot_events_per_month_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    data = _prepare_data_2(cohort, "month")
    ax = figure.gca()
    ax = _plot_bars(data, ax)
    for i, label in enumerate(ax.xaxis.get_ticklabels()):
        if i % 12 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)
    return figure


@xlabel("Week")
@ylabel("Count")
def plot_events_per_week_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    data = _prepare_data_2(cohort, "week")
    ax = figure.gca()
    ax = _plot_bars(data, ax)
    for i, label in enumerate(ax.xaxis.get_ticklabels()):
        if i % 52 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)
    return figure


@xlabel("Day")
@ylabel("Count")
def plot_events_per_day_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    data = _prepare_data_2(cohort, "day")
    ax = figure.gca()
    ax = _plot_bars(data, ax)
    return figure


@xlabel("Month")
@ylabel("Count")
def plot_events_per_month_as_timeseries(figure: Figure, cohort: Cohort) -> Figure:
    data = _prepare_data_2(cohort, "month")
    ax = figure.gca()
    ax = _plot_line(data, ax)
    return figure


@xlabel("Week")
@ylabel("Count")
def plot_events_per_week_as_timeseries(figure: Figure, cohort: Cohort) -> Figure:
    data = _prepare_data_2(cohort, "week")
    ax = figure.gca()
    ax = _plot_line(data, ax)
    return figure


@xlabel("Day")
@ylabel("Count")
def plot_events_per_day_as_timeseries(figure: Figure, cohort: Cohort) -> Figure:
    data = _prepare_data_2(cohort, "day")
    ax = figure.gca()
    ax = _plot_line(data, ax)
    return figure
