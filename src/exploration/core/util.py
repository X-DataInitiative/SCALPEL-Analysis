from functools import reduce
from typing import Callable, Iterable, List, Tuple, Optional

from pyspark.sql import DataFrame
import pyspark.sql.functions as sf
from pyspark.ml.feature import StringIndexer


def fold_right(f: Callable, cohorts: Iterable):
    t = iter(cohorts)
    init_value = next(t)
    return reduce(f, t, init_value)


def data_frame_equality(df1: DataFrame, df2: Optional[DataFrame]) -> bool:
    df1.schema.fieldNames()
    if isinstance(df1, DataFrame) and (isinstance(df2, DataFrame)):
        return (df1.subtract(df2.select(df1.schema.fieldNames())).count() == 0) and (
            df2.subtract(df1.select(df2.schema.fieldNames())).count() == 0
        )
    else:
        return False


def rename_df_columns(
    df: DataFrame,
    new_names: List[str] = None,
    prefix: str = "",
    suffix: str = "",
    keys: Tuple[str] = ("patientID",),
) -> DataFrame:
    """Rename columns of a pyspark DataFrame.

    :param df: dataframe whose columns will be renamed
    :param new_names: If not None, these name will replace the old ones.
     The order should be the same as df.columns where the keys has been
     removed.
    :param prefix: Prefix added to colnames.
    :param suffix: Suffix added to colnames.
    :param keys: Columns whose name will not be modified (useful for joining
     keys for example).
    :return: Dataframe with renamed columns.
    """
    old_names = [c for c in df.columns if c not in keys]
    if new_names is None:
        new_names = [prefix + c + suffix for c in old_names]
    return df.select(
        *keys, *[sf.col(c).alias(new_names[i]) for i, c in enumerate(old_names)]
    )


def index_string_column(
    dataframe: DataFrame, input_col: str, output_col: str
) -> Tuple[DataFrame, int, List[str]]:
    """Add a column containing an index corresponding to a string column.

    :param dataframe: Dataframe on which add index column.
    :param input_col: Name of the column to index
    :param output_col: Name of the index column in the resulting dataframe.
    :return: (resulting_dataframe, n_values_in_index, index_mapping)
    """
    indexer = StringIndexer(
        inputCol=input_col, outputCol=output_col, stringOrderType="alphabetAsc"
    )
    model = indexer.fit(dataframe)
    output = model.transform(dataframe)

    mapping = model.labels
    n_categories = len(mapping)

    return output, n_categories, mapping
