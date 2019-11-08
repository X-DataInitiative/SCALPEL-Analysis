# License: BSD 3 clause

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
    if isinstance(df1, DataFrame) and (isinstance(df2, DataFrame)):
        return (df1.subtract(df2.select(df1.schema.fieldNames())).count() == 0) and (
            df2.subtract(df1.select(df2.schema.fieldNames())).count() == 0
        )
    else:
        return False


def is_same_struct_type(df1: DataFrame, df2: Optional[DataFrame]) -> bool:
    if isinstance(df1, DataFrame) and (isinstance(df2, DataFrame)):
        schema_1 = {field.name: field.dataType for field in df1.schema.fields}
        schema_2 = {field.name: field.dataType for field in df2.schema.fields}
        return schema_1 == schema_2
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

    Parameters
    ----------
    df: dataframe whose columns will be renamed
    new_names: If not None, these name will replace the old ones.
     The order should be the same as df.columns where the keys has been
     removed.
    prefix: Prefix added to colnames.
    suffix: Suffix added to colnames.
    keys: Columns whose name will not be modified (useful for joining
     keys for example).

    Returns
    -------
       `pyspark.sql.DataFrame` with renamed columns.
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

    Parameters
    ----------
    dataframe: Dataframe on which add index column.
    input_col: Name of the column to index
    output_col: Name of the index column in the resulting dataframe.

    Returns
    Tuple (resulting_dataframe, n_values_in_index, index_mapping)
    """
    indexer = StringIndexer(
        inputCol=input_col, outputCol=output_col, stringOrderType="alphabetAsc"
    )
    model = indexer.fit(dataframe)
    output = model.transform(dataframe)

    mapping = model.labels
    n_categories = len(mapping)

    return output, n_categories, mapping
