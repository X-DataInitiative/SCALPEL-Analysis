import copy
import json
from typing import Dict, Set, Iterable

from src.exploration.core.util import fold_right
from src.exploration.flattening.flat_table import FlatTable


class FlatTableMetadata:
    """This object encapsulates a map of flat tables generated from flattening spark jobs.
    """

    def __init__(self, flat_tables: Dict[str, FlatTable]):
        assert flat_tables, "Dict can't be empty"
        self._tables = flat_tables

    @staticmethod
    def from_json(json_file: str) -> "FlatTableMetadata":
        metadata_json = json.loads(json_file)
        tables = metadata_json["flat_tables"]
        return FlatTableMetadata(
            {table["name"]: FlatTable.from_json(table) for table in tables}
        )

    @property
    def flat_tables(self) -> Dict[str, FlatTable]:
        return self._tables

    def flat_table_names(self) -> Set[str]:
        return set(self._tables.keys())

    def exists(self, flat_name: str) -> bool:
        return flat_name in self._tables

    def get(self, flat_name: str) -> FlatTable:
        if self.exists(flat_name):
            return self._tables[flat_name]
        else:
            raise KeyError(
                "Flat table {} do not exist in current repository".format(flat_name)
            )

    def add_flat_table(self, name: str, flat_table: FlatTable) -> "FlatTableMetadata":
        new_repo = copy.copy(self)
        new_repo._tables[name] = flat_table
        return new_repo

    def __iter__(self):
        return iter(self._tables)

    def union(self, other: "FlatTableMetadata") -> "FlatTableMetadata":
        return _union(self, other)

    def difference(self, other: "FlatTableMetadata") -> "FlatTableMetadata":
        return _difference(self, other)

    def intersection(self, other: "FlatTableMetadata") -> "FlatTableMetadata":
        return _intersection(self, other)

    @staticmethod
    def union_all(repos: Iterable["FlatTableMetadata"]) -> "FlatTableMetadata":
        return fold_right(_union, repos)

    @staticmethod
    def difference_all(repos: Iterable["FlatTableMetadata"]) -> "FlatTableMetadata":
        return fold_right(_difference, repos)

    @staticmethod
    def intersection_all(repos: Iterable["FlatTableMetadata"]) -> "FlatTableMetadata":
        return fold_right(_intersection, repos)


def _union(a: FlatTableMetadata, b: FlatTableMetadata) -> FlatTableMetadata:
    new_dict = copy.copy(a.flat_tables)
    new_dict.update(b.flat_tables)
    return FlatTableMetadata(new_dict)


def _intersection(a: FlatTableMetadata, b: FlatTableMetadata) -> FlatTableMetadata:
    return FlatTableMetadata(
        {key: a.flat_tables[key] for key in set(a).intersection(set(b))}
    )


def _difference(a: FlatTableMetadata, b: FlatTableMetadata) -> FlatTableMetadata:
    return FlatTableMetadata({key: a.flat_tables[key] for key in set(a) - set(b)})
