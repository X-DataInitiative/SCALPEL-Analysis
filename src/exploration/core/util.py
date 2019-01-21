from functools import reduce
from typing import Callable, Iterable

from pyspark.sql import DataFrame


def fold_right(f: Callable, cohorts: Iterable):
    t = iter(cohorts)
    init_value = next(t)
    return reduce(f, t, init_value)


def data_frame_equality(df1: DataFrame, df2: DataFrame) -> bool:
    if isinstance(df1, DataFrame) and (isinstance(df2, DataFrame)):
        return (df1.subtract(df2).count() == 0) and\
               (df2.subtract(df1).count() == 0)
    else:
        return False
