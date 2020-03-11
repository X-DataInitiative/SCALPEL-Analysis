#  License: BSD 3 clause
import logging
from functools import reduce
from typing import Callable, FrozenSet

import seaborn as sns
from matplotlib.figure import Figure
from pandas import DataFrame as PDDataFrame
from pyspark.sql.functions import col, count, lit, when, min, max, countDistinct, sum
from pyspark.sql.types import DecimalType

from scalpel.core.decorators import logged
from scalpel.flattening.flat_table import FlatTable
from scalpel.flattening.single_table import SingleTable
from scalpel.flattening.table import Table
from scalpel.stats.decorators import title, CONTEXT_SEABORN, ylabel
from scalpel.stats.flattening_stat import FlatteningStat
from scalpel.stats.grouper import Aggregator
from scalpel.stats.plotter import Plotter

_DCIR_COLS = {
    "ER_PHA_F": {"CIP13": "PHA_PRS_C13", "CIP07": "PHA_PRS_IDE"},
    "ER_CAM_F": {"CamCode": "CAM_PRS_IDE"},
    "ER_ETE_F": {
        "GHSCode": "ETE_GHS_NUM",
        "InstitutionCode": "ETE_TYP_COD",
        "Sector": "PRS_PPU_SEC",
    },
}

_MCO_COLS = {
    "MCO_A": {"CamCode": "CDC_ACT"},
    "MCO_B": {
        "DP": "DGN_PAL",
        "DR": "DGN_REL",
        "GHMCode": "GRG_GHM",
        "ExitMode": "SOR_MOD",
        "StayEndMonth": "SOR_MOI",
        "StayEndYear": "SOR_ANN",
        "StayLength": "SEJ_NBJ",
    },
    "MCO_D": {"DA": "ASS_DGN"},
}

_MCO_CE_COLS = {
    "MCO_FMSTC": {"CamCode": "CCAM_COD"},
    "MCO_FASTC": {"CodeSex": "COD_SEX", "ExitYear": "SOR_ANN", "ExitMonth": "SOR_MOI"},
}

_CNAM_COLS_MAPPING = {"DCIR": _DCIR_COLS, "MCO": _MCO_COLS, "MCO_CE": _MCO_CE_COLS}


@ylabel("Confidence Degree(%)", CONTEXT_SEABORN)
@title("Confidence Degree", CONTEXT_SEABORN)
@logged(logging.INFO, "Validation of flat table join")
def plot_flat_table_confidence_degree(
    figure: Figure,
    cohort: FlatTable,
    show=False,
    show_func=print,
    save_path=None,
    group_by_cols=None,
) -> Figure:
    """
    This method is used to calculate the confidence degree of a flat table
    and show the result in seaborn context.

    Parameters
    ----------
    figure: matplotlib.figure.Figure
        Users can define it like plt.figure() or plt.gcf().
    cohort: FlatTable
        A flat table.
    show: {False, True}, optional,
        If show the pandas table of confidence degree, default first when optional.
    show_func: optional
        Function to show a pandas table, print by default.
    save_path: str, optional
        the HDFS path to persist the pandas table, None by default
    group_by_cols: List
        Cols to group flat table, None by default.

    Notes
    -----
    only set group_by_cols when the flat table is not DCIR, MCO, or MCO_CE.

    Examples
    --------
    This is an example to illustrate how to use the function in jupyter.
    >>> with open("metadata_flattening.json", "r") as f:
    ...     dcir_collection = FlatTableCollection.from_json(f.read())
    >>> dcir = dcir_collection.get("DCIR")
    >>> plot_flat_table_confidence_degree(fig, dcir, show=True, show_func=display)
    >>> plt.show()
    """
    return FlatteningConfidenceDegreeStat()(
        figure,
        cohort,
        show=show,
        show_func=show_func,
        save_path=save_path,
        flat_table_name=cohort.name,
        group_by_cols=group_by_cols,
    )


def _union_not_null_cols_count(
    flat_table: FlatTable,
    single_table: SingleTable,
    nick_name: str,
    col_name: str,
    group_by_cols: FrozenSet[str] = None,
):
    flat_col_name = "{}__{}".format(single_table.name, col_name)
    flat = _count_not_null_cols(flat_table, nick_name, flat_col_name, group_by_cols)
    single = _count_not_null_cols(single_table, nick_name, col_name, group_by_cols)
    return flat.union(single)


def _count_not_null_cols(
    table: Table, nick_name: str, col_name: str, group_by_cols: FrozenSet[str]
):
    if group_by_cols:
        # algorithm used in MCO and MCO_CE
        return (
            table.source.filter(col(col_name).isNotNull())
            .groupBy(list(group_by_cols))
            .agg(countDistinct(col_name).alias("count"))
            .agg(sum("count").alias("count"))
            .withColumn("ColName", lit(nick_name))
        )
    else:
        # algorithm used in DCIR
        return (
            table.source.filter(col(col_name).isNotNull())
            .agg(count(col_name).alias("count"))
            .withColumn("ColName", lit(nick_name))
        )


def _confidence_degree_agg(flat_table: FlatTable, **kwargs) -> PDDataFrame:
    if kwargs.get("group_by_cols", None):
        group_by_cols = kwargs.get("group_by_cols")
    elif flat_table.name == "MCO":
        group_by_cols = frozenset(["ETA_NUM", "RSA_NUM"])
    elif flat_table.name == "MCO_CE":
        group_by_cols = frozenset(["ETA_NUM", "SEQ_NUM"])
    else:
        group_by_cols = None
    col = when(max("count") != 0, min("count") / max("count") * 100).otherwise(0)
    df = (
        reduce(
            lambda a, b: a.union(b),
            [
                _union_not_null_cols_count(
                    flat_table,
                    flat_table.single_tables[single_table_name],
                    nick_name,
                    col_name,
                    group_by_cols,
                )
                for single_table_name, cols in _CNAM_COLS_MAPPING[
                    flat_table.name
                ].items()
                for nick_name, col_name in cols.items()
            ],
        )
        .groupBy("ColName")
        .agg(col.cast(DecimalType(32, 2)).alias("ConfidenceDegree"))
    )
    return df.toPandas()


def _confidence_degree_plotter(figure: Figure, data: PDDataFrame, **kwargs) -> Figure:
    ax = sns.barplot(x="ColName", y="ConfidenceDegree", ax=figure.gca(), data=data)
    ax.set_xlabel("")
    return figure


def _confidence_degree_patcher(figure: Figure, **kwargs):
    label = "DCIR"
    if "flat_table_name" in kwargs.keys():
        label = kwargs["flat_table_name"]
    desc = "\n".join(
        [
            "{} in {}".format(", ".join(cols), single_table_name)
            for single_table_name, cols in _CNAM_COLS_MAPPING[label].items()
        ]
    )
    figure.text(0.95, 0.5, desc)
    figure.subplots_adjust(right=0.9)


class FlatteningConfidenceDegreeAgg(Aggregator):
    @property
    def agg(self) -> Callable:
        return _confidence_degree_agg


class FlatteningConfidenceDegreePlotter(Plotter):
    @property
    def plotter(self) -> Callable:
        return _confidence_degree_plotter

    @property
    def patch(self) -> bool:
        return True


class FlatteningConfidenceDegreeStat(
    FlatteningStat, FlatteningConfidenceDegreeAgg, FlatteningConfidenceDegreePlotter
):
    @property
    def patcher(self) -> Callable:
        return _confidence_degree_patcher
