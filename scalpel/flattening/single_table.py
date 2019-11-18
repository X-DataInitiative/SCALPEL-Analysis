#  License: BSD 3 clause
from typing import Dict

from pyspark.sql import DataFrame

from scalpel.core.io import read_data_frame
from scalpel.flattening.table import Table


class SingleTable(Table):
    """This object encapsulate single table representing a date set of original data
    in Parquet format.

    Parameters
    ----------
    name: str
        Single table name.
    source: str
        Data collection of this single table.
    characteristics: str
        The single table name that will show in the pyplot figure.
    """

    def __init__(self, name: str, source: DataFrame, characteristics: str):
        assert isinstance(name, str), "Name should be a str"
        assert isinstance(source, DataFrame), "Source should be a spark data frame"
        assert isinstance(characteristics, str), "Characteristics should be a str"
        self._name = name
        self._df = source
        self._chs = characteristics

    @staticmethod
    def from_json(json_content: Dict) -> "SingleTable":
        """
        Build single table from metadata.

        Parameters
        ----------
        json_content : Dict, single table part in metadata.

        See Also
        --------
        FlatTableCollection.from_json(json_file) : FlatTableCollection.
        FlatTable.from_json(json_content, single_tables) : FlatTable.

        Notes
        -----
        Generally, this method is called by FlatTableCollection.from_json(json_file).
        We got a single table by FlatTable.single_tables.get(single_table_name).
        """
        path = "{}/{}".format(json_content["output_path"], json_content["output_table"])
        return SingleTable(
            json_content["output_table"],
            read_data_frame(path),
            json_content["output_table"],
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
