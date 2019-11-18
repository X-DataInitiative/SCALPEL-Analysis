# License: BSD 3 clause

from functools import partial
from typing import Dict, Iterable, List

from pyspark.sql import DataFrame

from scalpel.core.io import read_data_frame
from scalpel.core.util import fold_right, is_same_struct_type
from scalpel.flattening.single_table import SingleTable
from scalpel.flattening.table import Table


class FlatTable(Table):
    """This object encapsulate flat table representing a date set of SNDS database.

    Parameters
    ----------
    name: str
        Flat table name.
    source: str
        Data collection of this flat table.
    characteristics: str
        The flat table name that will show in the pyplot figure.
    join_keys: List[str]
        The join keys by which single tables are joined.
    single_tables: Dict[str, SingleTable]
        The tables which constitute this flat table.
    """

    def __init__(
        self,
        name: str,
        source: DataFrame,
        characteristics: str,
        join_keys: List[str],
        single_tables: Dict[str, SingleTable],
    ):
        assert isinstance(name, str), "Name should be a str"
        assert isinstance(source, DataFrame), "Source should be a spark data frame"
        assert isinstance(characteristics, str), "Characteristics should be a str"
        assert isinstance(join_keys, List), "Join keys should be a list"
        assert isinstance(single_tables, Dict), "Single tables should be a Dict"
        self._name = name
        self._df = source
        self._chs = characteristics
        self._jk = join_keys
        self._single_tables = single_tables

    @staticmethod
    def from_json(
        json_content: Dict, single_tables: Dict[str, SingleTable]
    ) -> "FlatTable":
        """
        Build flat table from metadata.

        Parameters
        ----------
        json_content : Dict, flat table part in metadata.
        single_tables: Dict, single tables in this flat table.

        See Also
        --------
        FlatTableCollection.from_json(json_file) : FlatTableCollection.
        SingleTable.from_json(json_content) : SingleTable.

        Notes
        -----
        Generally, this method is called by FlatTableCollection.from_json(json_file).
        We got a flat table by FlatTableCollection.get(flat_table_name).
        """
        path = "{}/{}".format(json_content["output_path"], json_content["output_table"])
        return FlatTable(
            json_content["output_table"],
            read_data_frame(path),
            json_content["output_table"],
            json_content["join_keys"],
            single_tables,
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
    def join_keys(self) -> List[str]:
        return self._jk

    @property
    def single_tables(self) -> Dict[str, SingleTable]:
        return self._single_tables

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

    @single_tables.setter
    def single_tables(self, value):
        if not isinstance(value, Dict):
            raise TypeError("Expected a Dict")
        self._single_tables = value

    def union(self, other: "FlatTable") -> "FlatTable":
        """
        Return a new flat table containing union of values in self and another.

        Parameters
        ----------
        other: FlatTable, a flat table that will be united.
        """
        return _union(self, other)

    def intersection(self, other: "FlatTable") -> "FlatTable":
        """
        Return a new flat table containing rows only in both self and another.

        Parameters
        ----------
        other: FlatTable, a FlatTable that will be joined with self.
        """
        return _intersection(self, other, self._jk)

    def difference(self, other: "FlatTable") -> "FlatTable":
        """
        Return each value in self that is not contained in another.

        Parameters
        ----------
        other: FlatTable, a flat table that will be compared with self.
        """
        return _difference(self, other, self._jk)

    @staticmethod
    def union_all(flat_tables: Iterable["FlatTable"]) -> "FlatTable":
        """
        Return a new flat table containing union of values in an iteration of flat tables.

        Parameters
        ----------
        flat_tables: Iterable, an iteration of flat tables that will be united.
        """
        return fold_right(_union, flat_tables)

    @staticmethod
    def intersection_all(
        flat_tables: Iterable["FlatTable"], join_keys: List[str]
    ) -> "FlatTable":
        """
        Return a new flat table containing rows only in each from an iteration.

        Parameters
        ----------
        flat_tables: Iterable, an iteration of flat tables that will be joined with self.
        join_keys: List, join keys used to join each in the iteration.
        """
        return fold_right(partial(_intersection, join_keys=join_keys), flat_tables)

    @staticmethod
    def difference_all(
        flat_tables: Iterable["FlatTable"], join_keys: List[str]
    ) -> "FlatTable":
        """
        Return each values in the first that is not contained in others.

        Parameters
        ----------
        flat_tables: Iterable
            An iteration of flat tables that will be compared with self.
        join_keys: List
            Join keys used to join each in the iteration.
        """
        return fold_right(partial(_difference, join_keys=join_keys), flat_tables)


def _union(a: FlatTable, b: FlatTable) -> FlatTable:
    if not is_same_struct_type(a.source, b.source):
        raise ValueError("The passed tables do not share the same schema")

    single_tables = {**a.single_tables, **b.single_tables}

    return FlatTable(
        "{} Or {}".format(a.name, b.name),
        a.source.union(b.source),
        "{} Or {}".format(a.characteristics, b.characteristics),
        [*a.join_keys],
        single_tables,
    )


def _intersection(a: FlatTable, b: FlatTable, join_keys: List[str]) -> FlatTable:
    if not is_same_struct_type(a.source, b.source):
        raise ValueError("The passed tables do not share the same schema")
    intersect_keys = a.source.select(join_keys).intersect(b.source.select(join_keys))
    return FlatTable(
        "{} with {}".format(a.name, b.name),
        a.source.join(intersect_keys, on=join_keys, how="inner"),
        "{} with {}".format(a.characteristics, b.characteristics),
        [*a.join_keys],
        {**a.single_tables},
    )


def _difference(a: FlatTable, b: FlatTable, join_keys: List[str]) -> FlatTable:
    if not is_same_struct_type(a.source, b.source):
        raise ValueError("The passed tables do not share the same schema")
    difference_keys = a.source.select(join_keys).subtract(b.source.select(join_keys))
    return FlatTable(
        "{} without {}".format(a.name, b.name),
        a.source.join(difference_keys, on=join_keys, how="inner"),
        "{} without {}".format(a.characteristics, b.characteristics),
        [*a.join_keys],
        {**a.single_tables},
    )
