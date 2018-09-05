import copy
import json
from typing import Set

from .cohort import *


ALLOWED_OPERATIONS = frozenset(["union", "intersection", "difference"])


class Metadata:
    """Encaspulates the Metadata generated from SNIIRAM-featuring. Represents a set of
    Cohorts."""

    def __init__(self, cohorts: Dict[str, Cohort]):
        self.cohorts = cohorts

    @staticmethod
    def from_json(input: str) -> 'Metadata':
        metadata_json = json.loads(input)
        operations = metadata_json["operations"]
        return Metadata(
            {operation["name"]: Cohort.from_json(operation) for operation in operations})

    def get_from_description(self, description: Dict) -> Cohort:
        # TODO : this should be a method called from cohortz
        operation_type = description["type"]  # type: str
        if operation_type.lower() not in ALLOWED_OPERATIONS:
            raise KeyError(
                "{} not permitted. Only available operations: {}".format(operation_type,
                                                                        ALLOWED_OPERATIONS))
        else:
            parents = [self.get(parent) for parent in description['parents']]
            new_cohort = Cohort.union_all(parents)
            new_cohort.name = description["name"]
            return new_cohort

    def cohorts_names(self) -> Set[str]:
        return set(self.cohorts.keys())

    def exists(self, cohort_name: str) -> bool:
        return cohort_name in self.cohorts

    def get(self, cohort_name: str) -> Cohort:
        if self.exists(cohort_name):
            return self.cohorts[cohort_name]
        else:
            raise KeyError(
                "Cohort {} do not exist in current Metadata".format(cohort_name))

    def add_cohort(self, name: str, cohort: Cohort) -> 'Metadata':
        new_metadata = copy.copy(self)
        new_metadata.cohorts[name] = cohort
        return new_metadata

    def union(self, other: 'Metadata') -> 'Metadata':
        return _union(self, other)

    def difference(self, other: 'Metadata') -> 'Metadata':
        return _difference(self, other)

    def intersection(self, other: 'Metadata') -> 'Metadata':
        return _intersection(self, other)

    @staticmethod
    def union_all(metadatas: Iterable['Metadata']) -> 'Metadata':
        return fold_right(_union, metadatas)

    @staticmethod
    def intersect_all(metadatas: Iterable['Metadata']) -> 'Metadata':
        return fold_right(_intersection, metadatas)

    def __iter__(self):
        return iter(self.cohorts)


def _union(a, b) -> Metadata:
    new_dict = copy.copy(a.cohorts)
    new_dict.update(b.cohorts)
    return Metadata(new_dict)


def _difference(a, b) -> Metadata:
    return Metadata({k: a.cohorts[k] for k in set(a) - set(b)})


def _intersection(a, b) -> Metadata:
    return Metadata({k: a.cohorts[k] for k in set(a).intersection(set(b))})
