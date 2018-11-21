from abc import ABC, abstractmethod
from functools import lru_cache
from typing import FrozenSet, Callable

import pandas as pd
from pyspark.sql import DataFrame

from src.exploration.core.cohort import Cohort


@lru_cache(maxsize=128)
def agg_by_col(df: DataFrame, group_by_cols: FrozenSet[str], agg_col: str,
               agg_func: str) -> pd.DataFrame:
    """
    Aggregates a Spark Dataframe and returns a Pandas Dataframe. The main aim of this
    function is to enhance performance by caching already computed aggregations.
    :param df: DataFrame to perform the agg on.
    :param group_by_cols: frozenset of cols to group by with.
    :param agg_col: the column on which to apply the group by.
    :param agg_func: the name of the aggregation function.
    :return: pandas Dataframe.
    """
    return df.groupBy(list(group_by_cols)).agg({agg_col: agg_func}).toPandas()


@lru_cache(maxsize=128)
def agg(df: DataFrame, group_by_cols: FrozenSet[str], agg_func: str) -> pd.DataFrame:
    """
    Aggregates a Spark Dataframe and returns a Pandas Dataframe. The main aim of this
    function is to enhance performance by caching already computed aggregations.
    :param df: DataFrame to perform the agg on.
    :param group_by_cols: frozenset of cols to group by with.
    :param agg_col: the column on which to apply the group by.
    :param agg_func: the name of the aggregation function.
    :return: pandas Dataframe.
    """
    return df.groupBy(list(group_by_cols)).agg({"*": agg_func}).toPandas()


def event_start_agg(cohort: Cohort, agg_func: str)-> pd.DataFrame:
    return agg(cohort.events, frozenset(["start"]), agg_func).sort_values("start")


def event_group_id_agg(cohort: Cohort, agg_func: str)-> pd.DataFrame:
    return agg(cohort.events, frozenset(["groupID"]), agg_func)


def event_duration_agg(cohort: Cohort, agg_func: str)-> pd.DataFrame:
    return agg(cohort.events, frozenset(["duration"]), agg_func)


class Aggregator(ABC):
    @property
    @abstractmethod
    def agg(self) -> Callable:
        pass