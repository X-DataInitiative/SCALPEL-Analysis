#  License: BSD 3 clause
from abc import ABC, abstractmethod

from pyspark.sql import DataFrame

from scalpel.core.util import data_frame_equality, is_same_struct_type


class Table(ABC):
    """This object is the abstraction of single table and flat table.

    Attributes
    ----------
    name : str
           The table name.
    source : DataFrame
            The data collection of table.
    characteristics : str
            The table name that will show in the pyplot figure.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def source(self) -> DataFrame:
        pass

    @property
    @abstractmethod
    def characteristics(self) -> str:
        pass

    def __eq__(self, other):
        """ Return self==value."""
        if isinstance(other, Table) and is_same_struct_type(self.source, other.source):
            return data_frame_equality(self.source, other.source)
        else:
            return False

    def __contains__(self, item):
        """ Return key in self."""
        if isinstance(item, Table) and is_same_struct_type(self.source, item.source):
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
        """ Return self[SQL], this method is alias of sparkSQL select."""
        if not isinstance(item, str):
            raise TypeError("Expected a str")
        self.source.createOrReplaceTempView(self.name)
        sql_statement = "SELECT {} FROM {}".format(item, self.name)
        return self.source.sql_ctx.sql(sql_statement)
