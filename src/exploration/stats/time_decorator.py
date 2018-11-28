from abc import ABC, abstractmethod

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import IndexLocator

from src.exploration.core.cohort import Cohort
from src.exploration.stats.plotter import BarPlotter, LinePlotter


def _set_start_as_index(data: pd.DataFrame) -> pd.DataFrame:
    return data.set_index(
        pd.DatetimeIndex(data.start, tz="Europe/Paris", ambiguous=True, name="index")
    ).sort_index()


def _set_end_as_index(data: pd.DataFrame) -> pd.DataFrame:
    return data.set_index(
        pd.DatetimeIndex(data.end, tz="Europe/Paris", ambiguous=True, name="index")
    ).sort_index()


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
    ax.set_xticklabels(data.index.strftime("%d %b %Y")[::365])
    ax.grid(True, which="major", axis="x")
    return ax


def _patch_week(data, ax: Axes) -> Axes:
    major = IndexLocator(52, +0.4)
    minor = IndexLocator(4, +0.4)
    ax.xaxis.set_minor_locator(minor)
    ax.xaxis.set_major_locator(major)
    ax.set_xticklabels(data.index.strftime("%d %b %Y")[::52])
    ax.grid(True, which="major", axis="x")
    ax.grid(True, which="minor", axis="x", linestyle="--")
    return ax


def _patch_month(data, ax: Axes) -> Axes:
    major = IndexLocator(12, +0.4)
    minor = IndexLocator(1, +0.4)
    ax.xaxis.set_minor_locator(minor)
    ax.xaxis.set_major_locator(major)
    ax.set_xticklabels(data.index.strftime("%b %Y")[::12])
    ax.grid(True, which="major", axis="x", linestyle="--")
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


def _plot_count_per_time_unit(
    data: pd.DataFrame, time_unit: str, ax, plotter, patch_date_axe=True
) -> Axes:
    data = _set_start_as_index(data)
    data = _time_unit(data["count(1)"], time_unit)
    plotter(data, ax)
    if patch_date_axe:
        _patch_date_axe(data, ax, time_unit)
    return ax


def _plot_concept_count_per_time_unit(
    figure: Figure, time_unit: str, cohort: Cohort, agg, plotter, patch=True
) -> Figure:
    """Basic function"""
    data = agg(cohort, "count")
    _plot_count_per_time_unit(data, time_unit, figure.gca(), plotter, patch)
    return figure


class TimeUnit(ABC):
    @property
    @abstractmethod
    def time_unit(self) -> str:
        pass


class MonthUnit(TimeUnit):
    @property
    def time_unit(self) -> str:
        return "month"


class WeekUnit(TimeUnit):
    @property
    def time_unit(self):
        return "week"


class DayUnit(TimeUnit):
    @property
    def time_unit(self):
        return "day"


class TimedAggregatedCounterPlotter(ABC):
    def __call__(self, figure: Figure, cohort: Cohort) -> Figure:
        return _plot_concept_count_per_time_unit(
            figure, self.time_unit, cohort, self.agg, self.plotter, self.patch
        )


class MonthCounter(TimedAggregatedCounterPlotter, MonthUnit):
    pass


class MonthCounterBar(MonthCounter, BarPlotter):
    pass


class MonthCounterLine(MonthCounter, LinePlotter):
    pass


class WeekCounter(TimedAggregatedCounterPlotter, WeekUnit):
    pass


class WeekCounterBar(WeekCounter, BarPlotter):
    pass


class WeekCounterLine(WeekCounter, LinePlotter):
    pass


class DayCounter(TimedAggregatedCounterPlotter, DayUnit):
    pass


class DayCounterBar(DayCounter, BarPlotter):
    pass


class DayCounterLine(DayCounter, LinePlotter):
    pass
