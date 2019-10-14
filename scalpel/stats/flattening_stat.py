# License: BSD 3 clause

import logging
from abc import ABC, abstractmethod
from typing import Callable, List

import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame as pdDataFrame
from pyspark.sql.functions import col

from scalpel.core.decorators import logged
from scalpel.stats.decorators import ylabel, title, CONTEXT_SEABORN
from scalpel.flattening.flat_table import FlatTable
from scalpel.stats.grouper import agg, Aggregator
from scalpel.stats.plotter import Plotter

registry = []


def register(f):
    registry.append(f)
    return f


@register
@title("Compare patient events each year on months", CONTEXT_SEABORN)
@ylabel("Number of events", CONTEXT_SEABORN)
@logged(logging.INFO, "Patient events counted each year on months")
def plot_patient_events_each_year_on_months(
    figure: Figure,
    cohort: FlatTable,
    id_col: str = "NUM_ENQ",
    date_col: str = "EXE_SOI_DTD",
    years: List[int] = None,
) -> Figure:
    """
    This method is used to visualize the 'patient events each year on months stat' in
    seaborn context
    :param figure: 'matplotlib.figure.Figure' Users can define it like plt.figure() or
    plt.gcf()
    :param cohort: 'FlatTable'
    :param id_col: 'str' identity column default = 'NUM_ENQ'
    :param date_col: 'str' data column used for 'group by' statement
    default = 'EXE_SOI_DTD'
    :param years: a list of special years in which the data will be loaded,
    default is None
    :return: 'matplotlib.figure.Figure'
    """
    item = "{} as id, year({}) as year, month({}) as month".format(
        id_col, date_col, date_col
    )
    new_cohort = FlatTable(
        cohort.name, cohort[item], cohort.characteristics, ["id", "year", "month"]
    )
    return FlatteningEventsEachYearOnMonthsStat()(
        figure, new_cohort, id_col=id_col, date_col=date_col, years=years
    )


@register
@title("Compare patients each year on months", CONTEXT_SEABORN)
@ylabel("Number of patients", CONTEXT_SEABORN)
@logged(logging.INFO, "Patients counted each year on months")
def plot_patients_each_year_on_months(
    figure: Figure,
    cohort: FlatTable,
    id_col: str = "NUM_ENQ",
    date_col: str = "EXE_SOI_DTD",
    years: List[int] = None,
) -> Figure:
    """
       This method is used to visualize the 'patients each year on months stat' in seaborn
       context
       :param figure: 'matplotlib.figure.Figure' Users can define it like plt.figure() or
       plt.gcf()
       :param cohort: 'FlatTable'
       :param id_col: 'str' identity column default = 'NUM_ENQ'
       :param date_col: 'str' data column used for 'group by' statement
       default = 'EXE_SOI_DTD'
       :param years: a list of special years in which the data will be loaded, default is
       None
       :return: 'matplotlib.figure.Figure'
       """
    item = "distinct {} as id, year({}) as year, month({}) as month".format(
        id_col, date_col, date_col
    )
    new_cohort = FlatTable(
        cohort.name, cohort[item], cohort.characteristics, ["id", "year", "month"]
    )
    return FlatteningEventsEachYearOnMonthsStat()(
        figure, new_cohort, id_col=id_col, date_col=date_col, years=years
    )


@register
@title("Compare patient events on years", CONTEXT_SEABORN)
@ylabel("Number of events", CONTEXT_SEABORN)
@logged(logging.INFO, "Patient events counted on years")
def plot_patient_events_on_years(
    figure: Figure,
    cohort: FlatTable,
    id_col: str = "NUM_ENQ",
    date_col: str = "EXE_SOI_DTD",
    years: List[int] = None,
) -> Figure:
    """
     This method is used to visualize the 'patient events on years stat' int seaborn
     context
    :param figure: 'matplotlib.figure.Figure' Users can define it like plt.figure() or
    plt.gcf()
    :param cohort: 'FlatTable'
    :param id_col: 'str' identity column default = 'NUM_ENQ'
    :param date_col: 'str' data column used for 'group by' statement
    default = 'EXE_SOI_DTD'
    :param years: a list of special years in which the data will be loaded,
    default is None
    :return: 'matplotlib.figure.Figure'
    """
    item = "{} as id, year({}) as year".format(id_col, date_col)
    new_cohort = FlatTable(
        cohort.name, cohort[item], cohort.characteristics, ["id", "year"]
    )
    return FlatteningEventsOnYearsStat()(
        figure, new_cohort, id_col=id_col, date_col=date_col, years=years
    )


@register
@title("Compare patients on years", CONTEXT_SEABORN)
@ylabel("Number of patients", CONTEXT_SEABORN)
@logged(logging.INFO, "Patients counted on years")
def plot_patients_on_years(
    figure: Figure,
    cohort: FlatTable,
    id_col: str = "NUM_ENQ",
    date_col: str = "EXE_SOI_DTD",
    years: List[int] = None,
) -> Figure:
    """
     This method is used to visualize the 'patients on years stat' int seaborn context
    :param figure: 'matplotlib.figure.Figure' Users can define it like plt.figure() or
    plt.gcf()
    :param cohort: 'FlatTable'
    :param id_col: 'str' identity column default = 'NUM_ENQ'
    :param date_col: 'str' data column used for 'group by' statement
    default = 'EXE_SOI_DTD'
    :param years: a list of special years in which the data will be loaded,
    default is None
    :return: 'matplotlib.figure.Figure'
    """
    item = "distinct {} as id, year({}) as year".format(id_col, date_col)
    new_cohort = FlatTable(
        cohort.name, cohort[item], cohort.characteristics, ["id", "year"]
    )
    return FlatteningEventsOnYearsStat()(
        figure, new_cohort, id_col=id_col, date_col=date_col, years=years
    )


def _events_each_year_on_months_agg(cohort: FlatTable, **kwargs) -> pdDataFrame:
    # aggregate a flat table by year and month
    years_condition = kwargs.get("years", None)
    # if date column exists null replace year and month by 0
    df = cohort.source.fillna(0)
    if years_condition:
        df = df.where(col("year").isin(years_condition))
    return agg(df, frozenset(["year", "month"]), "count").sort_values(["year", "month"])


def _events_on_years_agg(cohort: FlatTable, **kwargs) -> pdDataFrame:
    # aggregate a flat table by year
    years_condition = kwargs.get("years", None)
    # if date column exists null replace year and month by 0
    df = cohort.source.fillna(0)
    if years_condition:
        df = df.where(col("year").isin(years_condition))
    return agg(df, frozenset(["year"]), "count").sort_values(["year"])


def _events_each_year_on_months_plotter(
    figure: Figure, data: pdDataFrame, **kwargs
) -> Axes:
    # Draw a set of vertical bars with nested grouping by years and months
    return sns.barplot(x="month", y="count(1)", hue="year", ax=figure.gca(), data=data)


def _events_on_years_plotter(figure: Figure, data: pdDataFrame, **kwargs) -> Axes:
    # Draw a set of vertical bars grouping by years
    return sns.barplot(x="year", y="count(1)", ax=figure.gca(), data=data)


def _events_each_year_on_months_patcher(ax: Axes, **kwargs):
    label = "EXE_SOI_DTD"
    if "date_col" in kwargs.keys():
        label = kwargs["date_col"]
    ax.set_xlabel("{} group by month".format(label))


def _events_on_years_patcher(ax: Axes, **kwargs):
    label = "EXE_SOI_DTD"
    if "date_col" in kwargs.keys():
        label = kwargs["date_col"]
    ax.set_xlabel("{} group by year".format(label))


def _plot_concept_flattening_stat(
    figure: Figure, cohort: FlatTable, agg_func, plotter, patch, patcher, **kwargs
) -> Figure:
    data = agg_func(cohort, **kwargs)
    ax = plotter(figure, data, **kwargs)
    if patch:
        patcher(ax, **kwargs)
    return figure


class FlatteningStat(ABC):
    def __call__(self, figure: Figure, cohort: FlatTable, **kwargs) -> Figure:
        return _plot_concept_flattening_stat(
            figure, cohort, self.agg, self.plotter, self.patch, self.patcher, **kwargs
        )

    @property
    @abstractmethod
    def patcher(self) -> Callable:
        pass


class FlatteningEventsEachYearOnMonthsAgg(Aggregator):
    @property
    def agg(self):
        return _events_each_year_on_months_agg


class FlatteningEventsOnYearsAgg(Aggregator):
    @property
    def agg(self):
        return _events_on_years_agg


class FlatteningEventsEachYearOnMonthsPlotter(Plotter):
    @property
    def plotter(self):
        return _events_each_year_on_months_plotter

    @property
    def patch(self):
        return True


class FlatteningEventsOnYearsPlotter(Plotter):
    @property
    def plotter(self):
        return _events_on_years_plotter

    @property
    def patch(self):
        return True


class FlatteningEventsEachYearOnMonthsStat(
    FlatteningStat,
    FlatteningEventsEachYearOnMonthsAgg,
    FlatteningEventsEachYearOnMonthsPlotter,
):
    """
    A flattening stat, comparison of a flat table each year on months, is provided to
    check the validation of flat
    tables(DCIR, MCO, MCO_CE, etc)

    Note that any concrete Flattening stat should provide aggregator, plotter, patcher

    """

    @property
    def patcher(self):
        return _events_each_year_on_months_patcher


class FlatteningEventsOnYearsStat(
    FlatteningStat, FlatteningEventsOnYearsAgg, FlatteningEventsOnYearsPlotter
):
    """
    A flattening stat, comparison of a flat table on years, is provided to check
    the validation of flat tables
    (DCIR, MCO, MCO_CE, etc)

    Note that any concrete Flattening stat should provide aggregator, plotter, patcher

    """

    @property
    def patcher(self):
        return _events_on_years_patcher
