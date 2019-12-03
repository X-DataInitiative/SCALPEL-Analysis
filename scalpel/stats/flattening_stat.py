# License: BSD 3 clause

import logging
from abc import ABC, abstractmethod
from typing import Callable, List

import pandas
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame as pdDataFrame
from pyspark.sql.functions import col, count

from scalpel.core.decorators import logged
from scalpel.core.io import write_from_pandas_data_frame
from scalpel.flattening.flat_table import FlatTable
from scalpel.stats.decorators import ylabel, title, CONTEXT_SEABORN
from scalpel.stats.grouper import Aggregator
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
    show=False,
    show_func=print,
    save_path=None,
    id_col: str = "NUM_ENQ",
    date_col: str = "EXE_SOI_DTD",
    years: List[int] = None,
) -> Figure:
    """
    This method is used to visualize the 'patient events each year on months stat'
    in seaborn context.

    Parameters
    ----------
    figure: matplotlib.figure.Figure, users can define it like plt.figure() or plt.gcf().
    cohort: FlatTable, a flat table.
    show: {False, True}, optional,
        If show the pandas table of confidence degree, default first when optional.
    show_func: optional
        Function to show a pandas table, print by default.
    save_path: str, optional
        the HDFS path to persist the pandas table, None by default, the save data can be
        used in stat history api.
    id_col: str, identity column default = 'NUM_ENQ'.
    date_col: str, data column used for 'group by' statement, default = 'EXE_SOI_DTD'.
    years: a list of special years in which the data will be loaded, default is None.

    Examples
    --------
    This is an example to illustrate how to use the function in jupyter.
    >>> with open("metadata_flattening.json", "r") as f:
    ...     dcir_collection = FlatTableCollection.from_json(f.read())
    >>> dcir = dcir_collection.get("DCIR")
    >>> plot_patient_events_each_year_on_months(plt.figure(figsize=(12, 8)), dcir)
    >>> plt.show()
    """
    item = "{} as id, year({}) as year, month({}) as month".format(
        id_col, date_col, date_col
    )
    new_cohort = FlatTable(
        cohort.name,
        cohort[item],
        cohort.characteristics,
        ["id", "year", "month"],
        cohort.single_tables,
    )
    return FlatteningEventsEachYearOnMonthsStat()(
        figure,
        new_cohort,
        show=show,
        show_func=show_func,
        save_path=save_path,
        id_col=id_col,
        date_col=date_col,
        years=years,
    )


@register
@title("Compare patients each year on months", CONTEXT_SEABORN)
@ylabel("Number of patients", CONTEXT_SEABORN)
@logged(logging.INFO, "Patients counted each year on months")
def plot_patients_each_year_on_months(
    figure: Figure,
    cohort: FlatTable,
    show=False,
    show_func=print,
    save_path=None,
    id_col: str = "NUM_ENQ",
    date_col: str = "EXE_SOI_DTD",
    years: List[int] = None,
) -> Figure:
    """This method is used to visualize the 'patients each year on months stat'
    in seaborn context.

    Parameters
    ----------
    figure: matplotlib.figure.Figure, users can define it like plt.figure() or plt.gcf().
    cohort: FlatTable, a flat table.
    show: {False, True}, optional,
        If show the pandas table of confidence degree, default first when optional.
    show_func: optional
        Function to show a pandas table, print by default.
    save_path: str, optional
        the HDFS path to persist the pandas table, None by default, the save data can be
        used in stat history api.
    id_col: str, identity column default = 'NUM_ENQ'.
    date_col: str, data column used for 'group by' statement, default = 'EXE_SOI_DTD'.
    years: a list of special years in which the data will be loaded, default is None.

    Examples
    --------
    This is an example to illustrate how to use the function in jupyter.
    >>> with open("metadata_flattening.json", "r") as f:
    ...     dcir_collection = FlatTableCollection.from_json(f.read())
    >>> dcir = dcir_collection.get("DCIR")
    >>> plot_patients_each_year_on_months(plt.figure(figsize=(12, 8)), dcir)
    >>> plt.show()
    """
    item = "distinct {} as id, year({}) as year, month({}) as month".format(
        id_col, date_col, date_col
    )
    new_cohort = FlatTable(
        cohort.name,
        cohort[item],
        cohort.characteristics,
        ["id", "year", "month"],
        cohort.single_tables,
    )
    return FlatteningEventsEachYearOnMonthsStat()(
        figure,
        new_cohort,
        show=show,
        show_func=show_func,
        save_path=save_path,
        id_col=id_col,
        date_col=date_col,
        years=years,
    )


@register
@title("Compare patient events on years", CONTEXT_SEABORN)
@ylabel("Number of events", CONTEXT_SEABORN)
@logged(logging.INFO, "Patient events counted on years")
def plot_patient_events_on_years(
    figure: Figure,
    cohort: FlatTable,
    show=False,
    show_func=print,
    save_path=None,
    id_col: str = "NUM_ENQ",
    date_col: str = "EXE_SOI_DTD",
    years: List[int] = None,
) -> Figure:
    """This method is used to visualize the 'patient events on years stat'
    int seaborn context.

    Parameters
    ----------
    figure: matplotlib.figure.Figure, users can define it like plt.figure() or plt.gcf().
    cohort: FlatTable, a flat table.
     show: {False, True}, optional,
        If show the pandas table of confidence degree, default first when optional.
    show_func: optional
        Function to show a pandas table, print by default.
    save_path: str, optional
        the HDFS path to persist the pandas table, None by default, the save data can be
        used in stat history api.
    id_col: str, identity column default = 'NUM_ENQ'.
    date_col: str, data column used for 'group by' statement, default = 'EXE_SOI_DTD'.
    years: a list of special years in which the data will be loaded, default is None.

    Examples
    --------
    This is an example to illustrate how to use the function in jupyter.
    >>> with open("metadata_flattening.json", "r") as f:
    ...     dcir_collection = FlatTableCollection.from_json(f.read())
    >>> dcir = dcir_collection.get("DCIR")
    >>> plot_patient_events_on_years(plt.figure(figsize=(12, 8)), dcir)
    >>> plt.show()
    """
    item = "{} as id, year({}) as year".format(id_col, date_col)
    new_cohort = FlatTable(
        cohort.name,
        cohort[item],
        cohort.characteristics,
        ["id", "year"],
        cohort.single_tables,
    )
    return FlatteningEventsOnYearsStat()(
        figure,
        new_cohort,
        show=show,
        show_func=show_func,
        save_path=save_path,
        id_col=id_col,
        date_col=date_col,
        years=years,
    )


@register
@title("Compare patients on years", CONTEXT_SEABORN)
@ylabel("Number of patients", CONTEXT_SEABORN)
@logged(logging.INFO, "Patients counted on years")
def plot_patients_on_years(
    figure: Figure,
    cohort: FlatTable,
    show=False,
    show_func=print,
    save_path=None,
    id_col: str = "NUM_ENQ",
    date_col: str = "EXE_SOI_DTD",
    years: List[int] = None,
) -> Figure:
    """This method is used to visualize the 'patients on years stat'
    in seaborn context

    Parameters
    ----------
    figure: matplotlib.figure.Figure, users can define it like plt.figure() or plt.gcf().
    cohort: FlatTable, a flat table.
     show: {False, True}, optional,
        If show the pandas table of confidence degree, default first when optional.
    show_func: optional
        Function to show a pandas table, print by default.
    save_path: str, optional
        the HDFS path to persist the pandas table, None by default, the save data can be
        used in stat history api.
    id_col: str, identity column default = 'NUM_ENQ'.
    date_col: str, data column used for 'group by' statement, default = 'EXE_SOI_DTD'.
    years: a list of special years in which the data will be loaded, default is None.

    Examples
    --------
    This is an example to illustrate how to use the function in jupyter.
    >>> with open("metadata_flattening.json", "r") as f:
    ...     dcir_collection = FlatTableCollection.from_json(f.read())
    >>> dcir = dcir_collection.get("DCIR")
    >>> plot_patients_on_years(plt.figure(figsize=(12, 8)), dcir)
    >>> plt.show()
    """
    item = "distinct {} as id, year({}) as year".format(id_col, date_col)
    new_cohort = FlatTable(
        cohort.name,
        cohort[item],
        cohort.characteristics,
        ["id", "year"],
        cohort.single_tables,
    )
    return FlatteningEventsOnYearsStat()(
        figure,
        new_cohort,
        show=show,
        show_func=show_func,
        save_path=save_path,
        id_col=id_col,
        date_col=date_col,
        years=years,
    )


def _events_each_year_on_months_agg(cohort: FlatTable, **kwargs) -> pdDataFrame:
    # aggregate a flat table by year and month
    years_condition = kwargs.get("years", None)
    # if date column exists null replace year and month by 0
    df = cohort.source.fillna(0)
    if years_condition:
        df = df.where(col("year").isin(years_condition))
    return (
        df.groupBy(list(frozenset(["year", "month"])))
        .agg(count("id").alias("count"))
        .toPandas()
        .sort_values(["year", "month"])
    )


def _events_on_years_agg(cohort: FlatTable, **kwargs) -> pdDataFrame:
    # aggregate a flat table by year
    years_condition = kwargs.get("years", None)
    # if date column exists null replace year and month by 0
    df = cohort.source.fillna(0)
    if years_condition:
        df = df.where(col("year").isin(years_condition))
    return (
        df.groupBy(list(frozenset(["year"])))
        .agg(count("id").alias("count"))
        .toPandas()
        .sort_values(["year"])
    )


def _events_each_year_on_months_plotter(
    figure: Figure, data: pdDataFrame, **kwargs
) -> Axes:
    # Draw a set of vertical bars with nested grouping by years and months
    return sns.barplot(x="month", y="count", hue="year", ax=figure.gca(), data=data)


def _events_on_years_plotter(figure: Figure, data: pdDataFrame, **kwargs) -> Axes:
    # Draw a set of vertical bars grouping by years
    return sns.barplot(x="year", y="count", ax=figure.gca(), data=data)


def _events_each_year_on_months_patcher(ax: Axes, **kwargs):
    label = "EXE_SOI_DTD"
    if "date_col" in kwargs.keys():
        label = kwargs["date_col"]
    ax.set_xlabel("{} group by month".format(label))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    ax.legend(loc="center right", bbox_to_anchor=(1.25, 0.5), ncol=1)


def _events_on_years_patcher(ax: Axes, **kwargs):
    label = "EXE_SOI_DTD"
    if "date_col" in kwargs.keys():
        label = kwargs["date_col"]
    ax.set_xlabel("{} group by year".format(label))


def _plot_concept_flattening_stat(
    figure: Figure,
    cohort: FlatTable,
    agg_func,
    plotter,
    patch,
    patcher,
    show=False,
    show_func=print,
    save_path=None,
    **kwargs
) -> Figure:
    data = agg_func(cohort, **kwargs)
    fig = plotter(figure, data, **kwargs)
    if patch:
        patcher(fig, **kwargs)
    if show:
        with pandas.option_context(
            "display.max_rows", None, "display.max_columns", None
        ):
            show_func(data)
    if save_path:
        write_from_pandas_data_frame(data, save_path)
    return figure


class FlatteningStat(ABC):
    def __call__(
        self,
        figure: Figure,
        cohort: FlatTable,
        show=False,
        show_func=print,
        save_path=None,
        **kwargs
    ) -> Figure:
        return _plot_concept_flattening_stat(
            figure,
            cohort,
            self.agg,
            self.plotter,
            self.patch,
            self.patcher,
            show,
            show_func,
            save_path,
            **kwargs
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
