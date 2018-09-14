import pandas as pd
from matplotlib.figure import Figure

from src.exploration.core.cohort import Cohort
from src.exploration.core.decorators import xlabel, ylabel
from src.exploration.stats.grouper import agg
from src.exploration.stats.plotter import plot_line, plot_bars


def _set_start_as_index(data: pd.DataFrame) -> pd.DataFrame:
    return data.set_index(pd.DatetimeIndex(data.start, tz="Europe/Paris", ambiguous=True,
                                           name="index")).sort_index()


def _set_end_as_index(data: pd.DataFrame) -> pd.DataFrame:
    return data.set_index(pd.DatetimeIndex(data.end, tz="Europe/Paris", ambiguous=True,
                                           name="index")).sort_index()


def _per_month(data: pd.Series) -> pd.Series:
    return data.groupby([data.index.year, data.index.month]).sum()


def _per_week(data: pd.Series) -> pd.Series:
    return data.groupby([data.index.year, data.index.week]).sum()


def _per_day(data: pd.Series) -> pd.Series:
    return data.groupby([data.index.year, data.index.week, data.index.day]).sum()


def _time_unit(data: pd.Series, time_unit: str) -> pd.Series:
    if time_unit == "day":
        return _per_day(data)
    elif time_unit == "month":
        return _per_month(data)
    elif time_unit == "week":
        return _per_week(data)
    else:
        raise ValueError("Wrong date unit {}. day, month, week only.".format(time_unit))


def _prepare_data(cohort: Cohort, date_unit: str) -> pd.Series:
    data = agg(cohort.events, frozenset(["start"]), "count").sort_values("start")
    data = _set_start_as_index(data)
    return _time_unit(data["count(1)"], date_unit)


@xlabel("Month")
@ylabel("Count")
def plot_events_per_month_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    data = _prepare_data(cohort, "month")
    ax = figure.gca()
    ax = plot_bars(data, ax)
    ax.xaxis_date()
    for i, label in enumerate(ax.xaxis.get_ticklabels()):
        if i % 12 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)
    return figure


@xlabel("Week")
@ylabel("Count")
def plot_events_per_week_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    data = _prepare_data(cohort, "week")
    ax = figure.gca()
    ax = plot_bars(data, ax)
    for i, label in enumerate(ax.xaxis.get_ticklabels()):
        if i % 52 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)
    return figure


@xlabel("Day")
@ylabel("Count")
def plot_events_per_day_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    data = _prepare_data(cohort, "day")
    ax = figure.gca()
    _ = plot_bars(data, ax)
    return figure


@xlabel("Month")
@ylabel("Count")
def plot_events_per_month_as_timeseries(figure: Figure, cohort: Cohort) -> Figure:
    data = _prepare_data(cohort, "month")
    ax = figure.gca()
    _ = plot_line(data, ax)
    return figure


@xlabel("Week")
@ylabel("Count")
def plot_events_per_week_as_timeseries(figure: Figure, cohort: Cohort) -> Figure:
    data = _prepare_data(cohort, "week")
    ax = figure.gca()
    _ = plot_line(data, ax)
    return figure


@xlabel("Day")
@ylabel("Count")
def plot_events_per_day_as_timeseries(figure: Figure, cohort: Cohort) -> Figure:
    data = _prepare_data(cohort, "day")
    ax = figure.gca()
    _ = plot_line(data, ax)
    return figure
