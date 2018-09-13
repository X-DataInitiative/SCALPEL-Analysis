import logging
from functools import lru_cache

import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame as pdDataFrame

from src.exploration.core.cohort import Cohort
from src.exploration.core.decorators import logged, title, xlabel, ylabel
from src.exploration.stats.grouper import agg
from src.exploration.stats.time_distribution import _plot_bars, _set_start_as_index, \
    _time_unit

registry = []


def register(f):
    registry.append(f)
    return f


def _admission_count(events) -> pdDataFrame:
    data = agg(events, frozenset(["patientID", "start"]), "count")
    return data.sort_values("start")


def _admission_count_per_start(cohort: Cohort) -> pdDataFrame:
    return _admission_count(cohort.events).groupby("start")[
        "patientID"].count().reset_index()


def _plot_admission_per_time_unit(cohort: Cohort, time_unit: str, ax) -> Axes:
    data = _admission_count_per_start(cohort)
    data = _set_start_as_index(data)
    data = _time_unit(data.patientID, time_unit)
    _plot_bars(data, ax)
    return ax


@register
@logged(logging.INFO, "Fractures count per body site")
@xlabel("Fractures count")
@ylabel("Body site")
@title("Fractures count per body site")
def plot_fractures_by_site(figure: Figure, cohort: Cohort) -> Figure:
    axe = figure.gca()
    fractures_site = agg(cohort.events,
                         frozenset(["groupID"]),
                         "count").sort_values("count(1)", ascending=True)
    axe.barh(y=range(len(fractures_site)), width=fractures_site["count(1)"].values,
             tick_label=fractures_site.groupID.values,
             color=sns.xkcd_rgb["pumpkin orange"], )
    axe.grid(True, which="major", axis="x")
    return figure


@register
@logged(logging.INFO, "Fractures count per admission")
@ylabel("Admissions count")
@xlabel("Fractures count")
@title("Fractures count per admission")
def plot_fractures_count_per_admission(figure: Figure, cohort: Cohort) -> Figure:
    ax = figure.gca()
    data = _admission_count(cohort.events)
    data = data.groupby("count(1)").count().patientID
    _plot_bars(data, ax)
    ax.grid(True, which="major", axis="y", linestyle='-')
    return figure


@register
@logged(logging.INFO, "Number of admission per subject")
@xlabel("Admissions count")
@ylabel("Subjects count")
@title("Number of admission per subject")
def plot_admission_number_per_patient(figure: Figure, cohort: Cohort) -> Figure:
    ax = figure.gca()
    data = _admission_count(cohort.events)[["start", "patientID"]].groupby(
        "patientID").count().reset_index().groupby("start").count().patientID
    _plot_bars(data, ax)
    ax.grid(True, which="major", axis="y")
    return figure


@register
@logged(logging.INFO, "Number of admission for fractures per day")
@xlabel("Day")
@ylabel("Admission count")
@title("Admission distribution per day")
def plot_admission_per_day(figure: Figure, cohort: Cohort) -> Figure:
    ax = figure.gca()
    _plot_admission_per_time_unit(cohort, "day", ax)
    ax.grid(True, which="major", axis="y", linestyle='-')
    return figure


@register
@logged(logging.INFO, "Number of admission for fractures per week")
@xlabel("Week")
@ylabel("Admission count")
@title("Admission distribution per week")
def plot_admission_per_week(figure: Figure, cohort: Cohort) -> Figure:
    ax = figure.gca()
    _plot_admission_per_time_unit(cohort, "week", ax)
    ax.grid(True, which="major", axis="y", linestyle='-')
    return figure


@register
@logged(logging.INFO, "Number of admission for fractures per month")
@xlabel("Month")
@ylabel("Admission count")
@title("Admission distribution per month")
def plot_admission_per_month(figure: Figure, cohort: Cohort) -> Figure:
    ax = figure.gca()
    _plot_admission_per_time_unit(cohort, "month", ax)
    ax.grid(True, which="major", axis="y", linestyle='-')
    return figure
