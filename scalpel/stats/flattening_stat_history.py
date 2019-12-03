#  License: BSD 3 clause
import logging
from typing import Callable, List, Dict

import seaborn as sns
from matplotlib.figure import Figure
from pandas import DataFrame as PDDataFrame
from pyspark.sql.functions import col

from scalpel.core.decorators import logged
from scalpel.core.io import read_data_frame
from scalpel.flattening.flat_table import HistoryTable
from scalpel.stats.decorators import ylabel, CONTEXT_SEABORN
from scalpel.stats.flattening_stat import FlatteningStat
from scalpel.stats.grouper import Aggregator
from scalpel.stats.plotter import Plotter


@logged(logging.INFO, "Histories of patient events")
def compare_stats_patient_events_on_months(
    figure: Figure,
    his_patient_events: Dict[str, str],
    show=False,
    show_func=print,
    save_path=None,
    years: List[int] = None,
) -> Figure:
    """
    This method is used to compare histories of patient events on months.

    Parameters
    ----------
    figure: matplotlib.figure.Figure, users can define it like plt.figure() or plt.gcf().
    his_patient_events: Dict, a dict of paths of patient events stats
    show: {False, True}, optional,
        If show the pandas table of confidence degree, default first when optional.
    show_func: optional
        Function to show a pandas table, print by default.
    save_path: str, optional
       the HDFS path to persist the pandas table, None by default, the save data can be
       used in stat history api.
    years: a list of special years in which the data will be loaded, default is None.

    Examples
    --------
    This is an example to illustrate how to use the function in jupyter.

    >>> his = {"A":"/path/A", "B":"/path/B"}
    ... compare_stats_patient_events_on_months(plt.gcf(), his)
    ... plt.show()
    """
    data = {name: read_data_frame(path) for (name, path) in his_patient_events.items()}
    cohort = HistoryTable.build(
        "Histories of patient events", "Histories of patient events", data
    )
    return _compare_stats_each_year_on_months(
        figure,
        cohort,
        show=show,
        show_func=show_func,
        save_path=save_path,
        title="History of patient events",
        ylabel="number of patient events",
        years=years,
    )


@logged(logging.INFO, "Histories of patients")
def compare_stats_patients_on_months(
    figure: Figure,
    his_patients: Dict[str, str],
    show=False,
    show_func=print,
    save_path=None,
    years: List[int] = None,
) -> Figure:
    """
    This method is used to compare histories of patients on months.

    Parameters
    ----------
    figure: matplotlib.figure.Figure, users can define it like plt.figure() or plt.gcf().
    his_patients: Dict, a dict of paths of patients stats
    show: {False, True}, optional,
        If show the pandas table of confidence degree, default first when optional.
    show_func: optional
        Function to show a pandas table, print by default.
    save_path: str, optional
       the HDFS path to persist the pandas table, None by default, the save data can be
       used in stat history api.
    years: a list of special years in which the data will be loaded, default is None.

    Examples
    --------
    This is an example to illustrate how to use the function in jupyter.

    >>> his = {"A":"/path/A", "B":"/path/B"}
    ... compare_stats_patients_on_months(plt.gcf(), his, show=True, show_func=display)
    ... plt.show()
    """
    data = {name: read_data_frame(path) for (name, path) in his_patients.items()}
    cohort = HistoryTable.build("Histories of patients", "Histories of patients", data)
    return _compare_stats_each_year_on_months(
        figure,
        cohort,
        show=show,
        show_func=show_func,
        save_path=save_path,
        title="History of patients",
        ylabel="number of patients",
        years=years,
    )


@ylabel("number of patient events", CONTEXT_SEABORN)
@logged(logging.INFO, "Histories of patient events")
def compare_stats_patient_events_on_years(
    figure: Figure,
    his_patient_events: Dict[str, str],
    show=False,
    show_func=print,
    save_path=None,
    years: List[int] = None,
) -> Figure:
    """
    This method is used to compare histories of patient events on years.

    Parameters
    ----------
    figure: matplotlib.figure.Figure, users can define it like plt.figure() or plt.gcf().
    his_patient_events: Dict, a dict of paths of patient events stats
    show: {False, True}, optional,
        If show the pandas table of confidence degree, default first when optional.
    show_func: optional
        Function to show a pandas table, print by default.
    save_path: str, optional
       the HDFS path to persist the pandas table, None by default, the save data can be
       used in stat history api.
    years: a list of special years in which the data will be loaded, default is None.

    Examples
    --------
    This is an example to illustrate how to use the function in jupyter.

    >>> his = {"A":"/path/A", "B":"/path/B"}
    ... compare_stats_patient_events_on_years(plt.gcf(), his)
    ... plt.show()
    """

    data = {name: read_data_frame(path) for (name, path) in his_patient_events.items()}
    cohort = HistoryTable.build(
        "Histories of patient events", "Histories of patient events", data
    )
    return _compare_stats_on_years(
        figure,
        cohort,
        show=show,
        show_func=show_func,
        save_path=save_path,
        title="Histories of patient events",
        years=years,
    )


@ylabel("number of patients", CONTEXT_SEABORN)
@logged(logging.INFO, "Histories of patients")
def compare_stats_patients_on_years(
    figure: Figure,
    his_patients: Dict[str, str],
    show=False,
    show_func=print,
    save_path=None,
    years: List[int] = None,
) -> Figure:
    """
    This method is used to compare histories of patients on years.

    Parameters
    ----------
    figure: matplotlib.figure.Figure, users can define it like plt.figure() or plt.gcf().
    his_patients: Dict, a dict of paths of patients stats
    show: {False, True}, optional,
        If show the pandas table of confidence degree, default first when optional.
    show_func: optional
        Function to show a pandas table, print by default.
    save_path: str, optional
       the HDFS path to persist the pandas table, None by default, the save data can be
       used in stat history api.
    years: a list of special years in which the data will be loaded, default is None.

    Examples
    --------
    This is an example to illustrate how to use the function in jupyter.

    >>> his = {"A":"/path/A", "B":"/path/B"}
    ... compare_stats_patients_on_years(plt.gcf(), his, show=True, show_func=display)
    ... plt.show()
    """

    data = {name: read_data_frame(path) for (name, path) in his_patients.items()}
    cohort = HistoryTable.build("Histories of patients", "Histories of patients", data)
    return _compare_stats_on_years(
        figure,
        cohort,
        show=show,
        show_func=show_func,
        save_path=save_path,
        title="History of patients",
        years=years,
    )


def _compare_stats_each_year_on_months(
    figure: Figure,
    cohort: HistoryTable,
    show=False,
    show_func=print,
    save_path=None,
    title: str = None,
    ylabel: str = None,
    years: List[int] = None,
) -> Figure:
    return FlatteningCompareYearMonthHistory()(
        figure,
        cohort,
        show=show,
        show_func=show_func,
        save_path=save_path,
        title=title,
        ylabel=ylabel,
        years=years,
    )


def _compare_stats_on_years(
    figure: Figure,
    cohort: HistoryTable,
    show=False,
    show_func=print,
    save_path=None,
    title: str = None,
    years: List[int] = None,
) -> Figure:
    return FlatteningCompareYearHistory()(
        figure,
        cohort,
        show=show,
        show_func=show_func,
        save_path=save_path,
        title=title,
        years=years,
    )


def _comparison_year_month_agg(history: HistoryTable, **kwargs) -> PDDataFrame:
    years_condition = kwargs.get("years", None)

    df = history.source
    if years_condition:
        df = df.where(col("year").isin(years_condition))
    return df.toPandas().sort_values(["year", "month", "history"])


def _comparison_year_month_plotter(figure: Figure, data: PDDataFrame, **kwargs):
    return sns.catplot(
        x="month", y="count", hue="history", col="year", data=data, kind="bar"
    )


def _comparison_year_month_patcher(figure, **kwargs):
    ylabel = kwargs.get("ylabel", None)
    title = kwargs.get("title", None)
    for ax in figure.axes.flat:
        if ax.get_ylabel() and ylabel:
            ax.set_ylabel(ylabel)
        if ax.get_title() and title:
            new_title = "{} {}".format(title, ax.get_title())
            ax.set_title(new_title)


def _comparison_year_agg(history: HistoryTable, **kwargs) -> PDDataFrame:
    years_condition = kwargs.get("years", None)

    df = history.source
    if years_condition:
        df = df.where(col("year").isin(years_condition))
    return df.toPandas().sort_values(["year", "history"])


def _comparison_year_plotter(figure: Figure, data: PDDataFrame, **kwargs):
    return sns.barplot(x="year", y="count", hue="history", ax=figure.gca(), data=data)


def _comparison_year_patcher(figure, **kwargs):
    title = kwargs.get("title", None)
    if title:
        figure.set_title(title)
    box = figure.get_position()
    figure.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    figure.legend(loc="center right", bbox_to_anchor=(1.25, 0.5), ncol=1)


class FlatteningCompareYearMonthAgg(Aggregator):
    @property
    def agg(self) -> Callable:
        return _comparison_year_month_agg


class FlatteningCompareYearMonthPlotter(Plotter):
    @property
    def plotter(self) -> Callable:
        return _comparison_year_month_plotter

    @property
    def patch(self) -> bool:
        return True


class FlatteningCompareYearMonthHistory(
    FlatteningStat, FlatteningCompareYearMonthAgg, FlatteningCompareYearMonthPlotter
):
    @property
    def patcher(self) -> Callable:
        return _comparison_year_month_patcher


class FlatteningCompareYearAgg(Aggregator):
    @property
    def agg(self) -> Callable:
        return _comparison_year_agg


class FlatteningCompareYearPlotter(Plotter):
    @property
    def plotter(self) -> Callable:
        return _comparison_year_plotter

    @property
    def patch(self) -> bool:
        return True


class FlatteningCompareYearHistory(
    FlatteningStat, FlatteningCompareYearAgg, FlatteningCompareYearPlotter
):
    @property
    def patcher(self) -> Callable:
        return _comparison_year_patcher
