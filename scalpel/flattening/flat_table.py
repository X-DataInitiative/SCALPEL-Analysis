# License: BSD 3 clause

from functools import partial
from typing import Dict, Iterable, List

from pyspark.sql import DataFrame

from scalpel.core.io import read_data_frame
from scalpel.core.util import data_frame_equality, fold_right


class FlatTable:
    """This object encapsulate flat table representing a date set of SNDS database
    """

    def __init__(
        self, name: str, source: DataFrame, characteristics: str, join_keys: List[str]
    ):
        assert isinstance(name, str), "Name should be a str"
        assert isinstance(source, DataFrame), "Source should be a spark data frame"
        assert isinstance(characteristics, str), "Characteristics should be a str"
        assert isinstance(join_keys, List), "Join keys should be a list"
        self._name = name
        self._df = source
        self._chs = characteristics
        self._jk = join_keys

    @staticmethod
    def from_json(json_content: Dict) -> "FlatTable":
        return FlatTable(
            json_content["output_table"],
            read_data_frame(json_content["output_path"]),
            json_content["output_table"],
            json_content["join_keys"],
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def source(self) -> DataFrame:
        return self._df

    @property
    def characteristics(self) -> str:
        return self._chs

    @property
    def join_keys(self):
        return self._jk

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError("Expected a string")
        self._name = value

    @source.setter
    def source(self, value: DataFrame):
        if not isinstance(value, DataFrame):
            raise TypeError("Expected a Spark DataFrame")
        self._df = value

    @characteristics.setter
    def characteristics(self, value):
        if not isinstance(value, str):
            raise TypeError("Expected a string")
        self._chs = value

    @join_keys.setter
    def join_keys(self, value):
        if not isinstance(value, List):
            raise TypeError("Expected a List")
        self._jk = value

    def __eq__(self, other):
        if isinstance(other, FlatTable) and _is_same_struct_type(
            self.source, other.source
        ):
            return data_frame_equality(self.source, other.source)
        else:
            return False

    def __contains__(self, item):
        if isinstance(item, FlatTable) and _is_same_struct_type(
            self.source, item.source
        ):
            df1 = self.source.distinct().cache()
            df2 = item.source.distinct().cache()
            c1 = df1.count()
            c2 = df2.count()
            diff = c1 - c2
            if c1 - c2 >= 0:
                return df1.subtract(df2).count() - diff == 0
            else:
                return False
        else:
            return False

    def __getitem__(self, item):
        if not isinstance(item, str):
            raise TypeError("Expected a str")
        self._df.createOrReplaceTempView(self.name)
        sql_statement = "SELECT {} FROM {}".format(item, self.name)
        return self._df.sql_ctx.sql(sql_statement)

    def union(self, other: "FlatTable") -> "FlatTable":
        return _union(self, other)

    def intersection(self, other: "FlatTable") -> "FlatTable":
        return _intersection(self, other, self._jk)

    def difference(self, other: "FlatTable") -> "FlatTable":
        return _difference(self, other, self._jk)

    @staticmethod
    def union_all(flat_tables: Iterable["FlatTable"]) -> "FlatTable":
        return fold_right(_union, flat_tables)

    @staticmethod
    def intersection_all(
        flat_tables: Iterable["FlatTable"], join_keys: List[str]
    ) -> "FlatTable":
        return fold_right(partial(_intersection, join_keys=join_keys), flat_tables)

    @staticmethod
    def difference_all(
        flat_tables: Iterable["FlatTable"], join_keys: List[str]
    ) -> "FlatTable":
        return fold_right(partial(_difference, join_keys=join_keys), flat_tables)


def _is_same_struct_type(a: DataFrame, b: DataFrame) -> bool:
    schema_1 = {field.name: field.dataType for field in a.schema.fields}
    schema_2 = {field.name: field.dataType for field in b.schema.fields}
    return schema_1 == schema_2


def _union(a: FlatTable, b: FlatTable) -> FlatTable:
    if not _is_same_struct_type(a.source, b.source):
        raise ValueError("The passed tables do not share the same schema")
    return FlatTable(
        "{} Or {}".format(a.name, b.name),
        a.source.union(b.source),
        "{} Or {}".format(a.characteristics, b.characteristics),
        a.join_keys,
    )


def _intersection(a: FlatTable, b: FlatTable, join_keys: List[str]) -> FlatTable:
    if not _is_same_struct_type(a.source, b.source):
        raise ValueError("The passed tables do not share the same schema")
    intersect_keys = a.source.select(join_keys).intersect(b.source.select(join_keys))
    return FlatTable(
        "{} with {}".format(a.name, b.name),
        a.source.join(intersect_keys, on=join_keys, how="inner"),
        "{} with {}".format(a.characteristics, b.characteristics),
        join_keys,
    )


def _difference(a: FlatTable, b: FlatTable, join_keys: List[str]) -> FlatTable:
    if not _is_same_struct_type(a.source, b.source):
        raise ValueError("The passed tables do not share the same schema")
    difference_keys = a.source.select(join_keys).subtract(b.source.select(join_keys))
    return FlatTable(
        "{} without {}".format(a.name, b.name),
        a.source.join(difference_keys, on=join_keys, how="inner"),
        "{} without {}".format(a.characteristics, b.characteristics),
        join_keys,
    )
