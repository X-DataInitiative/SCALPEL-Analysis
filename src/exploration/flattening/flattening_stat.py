import logging
from abc import ABC, abstractmethod
from typing import Callable, List

import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame as pdDataFrame
from pyspark.sql.functions import col, year, month

from src.exploration.core.decorators import logged, ylabel, title, CONTEXT_SEABORN
from src.exploration.flattening.flattable import FlatTable
from src.exploration.stats.grouper import agg, Aggregator
from src.exploration.stats.plotter import Plotter

registry = []


def register(f):
    registry.append(f)
    return f


@register
@title("Compare each year events on month", CONTEXT_SEABORN)
@ylabel('Number of events', CONTEXT_SEABORN)
@logged(logging.INFO, "Events counted each year on month")
def plot_flat_table_each_year_on_months(figure: Figure,
                                        cohort: FlatTable,
                                        id_col: str = 'NUM_ENQ',
                                        date_col: str = 'EXE_SOI_DTD',
                                        years: List[int] = None) -> Figure:
    """
    This method is used to visualize the 'flat table each year on months stat' int seaborn context
    :param figure: 'matplotlib.figure.Figure' Users can define it like plt.figure() or plt.gcf()
    :param cohort: 'FlatTable'
    :param id_col: 'str' identity column default = 'NUM_ENQ'
    :param date_col: 'str' data column used for 'group by' statement default = 'EXE_SOI_DTD'
    :param years: a list of special years in which the data will be loaded, default is None
    :return: 'matplotlib.figure.Figure'
    """
    return FlatteningEventsEachYearOnMonthsStat()(figure, cohort, id_col=id_col, date_col=date_col, years=years)


@register
@title("Compare events on year", CONTEXT_SEABORN)
@ylabel('Number of events', CONTEXT_SEABORN)
@logged(logging.INFO, "Events counted on year")
def plot_flat_table_on_years(figure: Figure,
                             cohort: FlatTable,
                             id_col: str = 'NUM_ENQ',
                             date_col: str = 'EXE_SOI_DTD',
                             years: List[int] = None) -> Figure:
    """
     This method is used to visualize the 'flat table on years stat' int seaborn context
    :param figure: 'matplotlib.figure.Figure' Users can define it like plt.figure() or plt.gcf()
    :param cohort: 'FlatTable'
    :param id_col: 'str' identity column default = 'NUM_ENQ'
    :param date_col: 'str' data column used for 'group by' statement default = 'EXE_SOI_DTD'
    :param years: a list of special years in which the data will be loaded, default is None
    :return: 'matplotlib.figure.Figure'
    """
    return FlatteningEventsOnYearsStat()(figure, cohort, id_col=id_col, date_col=date_col, years=years)


def _events_each_year_on_months_agg(cohort: FlatTable, **kwargs) -> pdDataFrame:
    # aggregate a flat table by year and month

    id_col = kwargs.get('id_col', 'NUM_ENQ')
    date_col = kwargs.get('date_col', 'EXE_SOI_DTD')
    years_condition = kwargs.get('years', None)
    # if date column exists null replace year and month by 0
    df = cohort.source.select(col(id_col).alias('id'), year(date_col).alias('year'),
                              month(date_col).alias('month')).fillna(0)
    if years_condition is not None and len(years_condition) > 0:
        df = df.where(col('year').isin(years_condition))
    return agg(df, frozenset(['year', 'month']), 'count').sort_values(['year', 'month'])


def _events_on_years_agg(cohort: FlatTable, **kwargs) -> pdDataFrame:
    # aggregate a flat table by year
    id_col = kwargs.get('id_col', 'NUM_ENQ')
    date_col = kwargs.get('date_col', 'EXE_SOI_DTD')
    years_condition = kwargs.get('years', None)
    # if date column exists null replace year and month by 0
    df = cohort.source.select(col(id_col).alias('id'), year(date_col).alias('year')).fillna(0)
    if years_condition is not None and len(years_condition) > 0:
        df = df.where(col('year').isin(years_condition))
    return agg(df, frozenset(['year']), 'count').sort_values(['year'])


def _events_each_year_on_months_plotter(figure: Figure, data: pdDataFrame, **kwargs) -> Axes:
    # Draw a set of vertical bars with nested grouping by years and months
    return sns.barplot(x='month', y='count(1)', hue='year', ax=figure.gca(), data=data)


def _events_on_years_plotter(figure: Figure, data: pdDataFrame, **kwargs) -> Axes:
    # Draw a set of vertical bars grouping by years
    return sns.barplot(x='year', y='count(1)', ax=figure.gca(), data=data)


def _events_each_year_on_months_patcher(ax: Axes, **kwargs):
    x_label = 'EXE_SOI_DTD'
    if "date_col" in kwargs.keys():
        x_label = kwargs['date_col']
    ax.set_xlabel("%s group by month" % x_label)


def _events_on_years_patcher(ax: Axes, **kwargs):
    x_label = 'EXE_SOI_DTD'
    if "date_col" in kwargs.keys():
        x_label = kwargs['date_col']
    ax.set_xlabel("%s group by year" % x_label)


def _plot_concept_flattening_stat(figure: Figure, cohort: FlatTable, agg_func, plotter, patch, patcher,
                                  **kwargs) -> Figure:
    data = agg_func(cohort, **kwargs)
    ax = plotter(figure, data, **kwargs)
    if patch:
        patcher(ax, **kwargs)
    return figure


class FlatteningStat(ABC):
    def __call__(self, figure: Figure, cohort: FlatTable, **kwargs) -> Figure:
        return _plot_concept_flattening_stat(figure, cohort, self.agg, self.plotter, self.patch, self.patcher, **kwargs)

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


class FlatteningEventsEachYearOnMonthsStat(FlatteningStat, FlatteningEventsEachYearOnMonthsAgg,
                                           FlatteningEventsEachYearOnMonthsPlotter):
    """
    A flattening stat, comparison of a flat table each year on months, is provided to check the validation of flat
    tables(DCIR, MCO, MCO_CE, etc)

    Note that any concrete Flattening stat should provide aggregator, plotter, patcher

    """

    @property
    def patcher(self):
        return _events_each_year_on_months_patcher


class FlatteningEventsOnYearsStat(FlatteningStat, FlatteningEventsOnYearsAgg, FlatteningEventsOnYearsPlotter):
    """
    A flattening stat, comparison of a flat table on years, is provided to check the validation of flat tables
    (DCIR, MCO, MCO_CE, etc)

    Note that any concrete Flattening stat should provide aggregator, plotter, patcher

    """

    @property
    def patcher(self):
        return _events_on_years_patcher
