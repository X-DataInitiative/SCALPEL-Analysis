# License: BSD 3 clause

import logging

import seaborn as sns
from matplotlib.figure import Figure
from pandas import DataFrame as pdDataFrame

from scalpel.core.cohort import Cohort
from scalpel.core.decorators import logged
from scalpel.stats.decorators import title, xlabel, ylabel
from scalpel.stats.grouper import Aggregator, agg, event_group_id_agg
from scalpel.stats.time_decorator import (
    DayCounterBar,
    DayCounterLine,
    MonthCounterBar,
    MonthCounterLine,
    WeekCounterBar,
    WeekCounterLine,
)

registry = []


def register(f):
    registry.append(f)
    return f


def _admission_count(cohort: Cohort) -> pdDataFrame:
    return admission_agg(cohort, "count")


def admission_agg(cohort: Cohort, agg_func: str):
    return agg(cohort.events, frozenset(["patientID", "start"]), agg_func).sort_values(
        "start"
    )


@register
@logged(logging.INFO, "Fractures count per body site")
@xlabel("Fractures count")
@ylabel("Body site")
@title("Fractures count per body site")
def plot_fractures_by_site(figure: Figure, cohort: Cohort) -> Figure:
    axe = figure.gca()
    fractures_site = event_group_id_agg(cohort, "count").sort_values(
        "count(1)", ascending=True
    )
    axe.barh(
        range(len(fractures_site)),
        fractures_site["count(1)"].values,
        tick_label=fractures_site.groupID.values,
    )
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
    sns.barplot(data.index.values, data.values)
    ax.grid(True, which="major", axis="y", linestyle="-")
    return figure


@register
@logged(logging.INFO, "Number of admission per subject")
@xlabel("Admissions count")
@ylabel("Subjects count")
@title("Number of admission per subject")
def plot_admission_number_per_patient(figure: Figure, cohort: Cohort) -> Figure:
    ax = figure.gca()
    data = (
        _admission_count(cohort)[["start", "patientID"]]
        .groupby("patientID")
        .count()
        .reset_index()
        .groupby("start")
        .count()
        .patientID
    )
    sns.barplot(data.index.values, data.values)
    ax.grid(True, which="major", axis="y")
    return figure


@register
@logged(logging.INFO, "Number of admission for fractures per day")
@xlabel("Day")
@ylabel("Admission count")
@title("Admission distribution per day")
def plot_admission_per_day_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    return DayCounterBarAdmission()(figure, cohort)


@register
@logged(logging.INFO, "Number of admission for fractures per week")
@xlabel("Week")
@ylabel("Admission count")
@title("Admission distribution per week")
def plot_admission_per_week_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    return WeekCounterBarAdmission()(figure, cohort)


@register
@logged(logging.INFO, "Number of admission for fractures per month")
@xlabel("Month")
@ylabel("Admission count")
@title("Admission distribution per month")
def plot_admission_per_month_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    return MonthCounterBarAdmission()(figure, cohort)


@register
@logged(logging.INFO, "Number of admission for fractures per day")
@xlabel("Day")
@ylabel("Admission count")
@title("Admission distribution per day")
def plot_admission_per_day_as_timeseries(figure: Figure, cohort: Cohort) -> Figure:
    return DayCounterLineAdmission()(figure, cohort)


@register
@logged(logging.INFO, "Number of admission for fractures per week")
@xlabel("Week")
@ylabel("Admission count")
@title("Admission distribution per week")
def plot_admission_per_week_as_timeseries(figure: Figure, cohort: Cohort) -> Figure:
    return WeekCounterLineAdmission()(figure, cohort)


@register
@logged(logging.INFO, "Number of admission for fractures per month")
@xlabel("Month")
@ylabel("Admission count")
@title("Admission distribution per month")
def plot_admission_per_month_as_timeseries(figure: Figure, cohort: Cohort) -> Figure:
    return MonthCounterLineAdmission()(figure, cohort)


class AdmissionStartAgg(Aggregator):
    @property
    def agg(self):
        return admission_agg


class MonthCounterBarAdmission(MonthCounterBar, AdmissionStartAgg):
    pass


class WeekCounterBarAdmission(WeekCounterBar, AdmissionStartAgg):
    pass


class DayCounterBarAdmission(DayCounterBar, AdmissionStartAgg):
    pass


class MonthCounterLineAdmission(MonthCounterLine, AdmissionStartAgg):
    pass


class WeekCounterLineAdmission(WeekCounterLine, AdmissionStartAgg):
    pass


class DayCounterLineAdmission(DayCounterLine, AdmissionStartAgg):
    pass
