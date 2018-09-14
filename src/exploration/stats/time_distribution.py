import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import IndexLocator

from src.exploration.core.cohort import Cohort
from src.exploration.core.decorators import xlabel, ylabel
from src.exploration.stats.grouper import agg
from src.exploration.stats.plotter import plot_bars, plot_line


def _set_start_as_index(data: pd.DataFrame) -> pd.DataFrame:
    return data.set_index(pd.DatetimeIndex(data.start, tz="Europe/Paris", ambiguous=True,
                                           name="index")).sort_index()


def _set_end_as_index(data: pd.DataFrame) -> pd.DataFrame:
    return data.set_index(pd.DatetimeIndex(data.end, tz="Europe/Paris", ambiguous=True,
                                           name="index")).sort_index()


def _per_month(data: pd.Series) -> pd.Series:
    return data.groupby(pd.Grouper(freq="M")).sum()


def _per_week(data: pd.Series) -> pd.Series:
    return data.groupby(pd.Grouper(freq="W")).sum()


def _per_day(data: pd.Series) -> pd.Series:
    return data.groupby(pd.Grouper(freq="D")).sum()


def _time_unit(data: pd.Series, time_unit: str) -> pd.Series:
    if time_unit == "day":
        return _per_day(data)
    elif time_unit == "month":
        return _per_month(data)
    elif time_unit == "week":
        return _per_week(data)
    else:
        raise ValueError("Wrong date unit {}. day, month, week only.".format(time_unit))


def _patch_day(data, ax: Axes) -> Axes:
    major = IndexLocator(365, +0.4)
    minor = IndexLocator(7, +0.4)
    ax.xaxis.set_minor_locator(minor)
    ax.xaxis.set_major_locator(major)
    ax.set_xticklabels(data.index.strftime('%d %b %Y')[::365])
    ax.grid(True, which="major", axis="x")
    return ax


def _patch_week(data, ax: Axes) -> Axes:
    major = IndexLocator(52, +0.4)
    minor = IndexLocator(4, +0.4)
    ax.xaxis.set_minor_locator(minor)
    ax.xaxis.set_major_locator(major)
    ax.set_xticklabels(data.index.strftime('%d %b %Y')[::52])
    ax.grid(True, which="major", axis="x")
    ax.grid(True, which="minor", axis="x", linestyle='--')
    return ax


def _patch_month(data, ax: Axes) -> Axes:
    major = IndexLocator(12, +0.4)
    minor = IndexLocator(1, +0.4)
    ax.xaxis.set_minor_locator(minor)
    ax.xaxis.set_major_locator(major)
    ax.set_xticklabels(data.index.strftime('%b %Y')[::12])
    ax.grid(True, which="major", axis="x", linestyle='--')
    return ax


def _patch_date_axe(data: pd.Series, axe: Axes, time_unit: str) -> Axes:
    if time_unit == "day":
        return _patch_day(data, axe)
    elif time_unit == "month":
        return _patch_month(data, axe)
    elif time_unit == "week":
        return _patch_week(data, axe)
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
    _patch_date_axe(data, ax, "month")
    return figure


@xlabel("Week")
@ylabel("Count")
def plot_events_per_week_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    data = _prepare_data(cohort, "week")
    ax = figure.gca()
    ax = plot_bars(data, ax)
    _patch_date_axe(data, ax, "week")
    return figure


@xlabel("Day")
@ylabel("Count")
def plot_events_per_day_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    data = _prepare_data(cohort, "day")
    ax = figure.gca()
    _ = plot_bars(data, ax)
    _patch_date_axe(data, ax, "day")
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
