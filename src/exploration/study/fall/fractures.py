import logging

import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame as pdDataFrame

from src.exploration.core.cohort import Cohort
from src.exploration.core.decorators import logged, title, xlabel, ylabel
from src.exploration.stats.grouper import agg, event_group_id_agg
from src.exploration.stats.plotter import plot_bars
from src.exploration.stats.time_distribution import _set_start_as_index, \
    _time_unit, _patch_date_axe, _plot_concept_count_per_start_time, \
    _plot_concept_count_per_start_time_as_bars, \
    _plot_concept_count_per_start_time_as_timeseries

registry = []


def register(f):
    registry.append(f)
    return f


def _admission_count(cohort: Cohort) -> pdDataFrame:
    return admission_agg(cohort, "count")


def _admission_count_per_start(cohort: Cohort) -> pdDataFrame:
    return _admission_count(cohort.events).groupby("start")[
        "patientID"].count().reset_index()


def admission_agg(cohort: Cohort, agg_func: str):
    return agg(cohort.events, frozenset(["patientID", "start"]), agg_func).sort_values(
        "start")


def _plot_admission_count_as_bars(figure: Figure, time_unit: str,
                                  cohort: Cohort) -> Figure:
    return _plot_concept_count_per_start_time_as_bars(figure, time_unit, cohort,
                                                      admission_agg)


def _plot_admission_count_as_timeseries(figure: Figure, time_unit: str,
                                        cohort: Cohort) -> Figure:
    return _plot_concept_count_per_start_time_as_timeseries(figure, time_unit, cohort,
                                                            admission_agg)


@register
@logged(logging.INFO, "Fractures count per body site")
@xlabel("Fractures count")
@ylabel("Body site")
@title("Fractures count per body site")
def plot_fractures_by_site(figure: Figure, cohort: Cohort) -> Figure:
    axe = figure.gca()
    fractures_site = event_group_id_agg(cohort, "count").sort_values("count(1)",
                                                                     ascending=True)
    axe.barh(y=range(len(fractures_site)), width=fractures_site["count(1)"].values,
             tick_label=fractures_site.groupID.values)
    return figure


@register
@logged(logging.INFO, "Fractures count per admission")
@ylabel("Admissions count")
@xlabel("Fractures count")
@title("Fractures count per admission")
def plot_fractures_count_per_admission(figure: Figure, cohort: Cohort) -> Figure:
    ax = figure.gca()
    data = _admission_count(cohort)
    data = data.groupby("count(1)").count().patientID
    sns.barplot(x=data.index.values, y=data.values)
    ax.grid(True, which="major", axis="y", linestyle='-')
    return figure


@register
@logged(logging.INFO, "Number of admission per subject")
@xlabel("Admissions count")
@ylabel("Subjects count")
@title("Number of admission per subject")
def plot_admission_number_per_patient(figure: Figure, cohort: Cohort) -> Figure:
    ax = figure.gca()
    data = _admission_count(cohort)[["start", "patientID"]].groupby(
        "patientID").count().reset_index().groupby("start").count().patientID
    sns.barplot(x=data.index.values, y=data.values)
    ax.grid(True, which="major", axis="y")
    return figure


@register
@logged(logging.INFO, "Number of admission for fractures per day")
@xlabel("Day")
@ylabel("Admission count")
@title("Admission distribution per day")
def plot_admission_per_day_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    return _plot_admission_count_as_bars(figure, "day", cohort)


@register
@logged(logging.INFO, "Number of admission for fractures per week")
@xlabel("Week")
@ylabel("Admission count")
@title("Admission distribution per week")
def plot_admission_per_week_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    return _plot_admission_count_as_bars(figure, "week", cohort)


@register
@logged(logging.INFO, "Number of admission for fractures per month")
@xlabel("Month")
@ylabel("Admission count")
@title("Admission distribution per month")
def plot_admission_per_month_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    return _plot_admission_count_as_bars(figure, "month", cohort)
