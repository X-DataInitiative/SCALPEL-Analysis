# License: BSD 3 clause

import copy
import json
from typing import Dict, Iterable, Set, Tuple

from scalpel.core.cohort import Cohort
from scalpel.core.io import get_logger
from scalpel.core.util import fold_right

ALLOWED_OPERATIONS = frozenset(["union", "intersection", "difference"])


class CohortCollection:
    """Represents a set of Cohorts. It can be used to encapsulate the cc files
     generated from SCALPEL-Extraction and perform operation on sets of cohorts.

    Parameters
    ----------
    cohorts: Dict containing `scalpel.core.cohort.Cohort` objects.
    """

    def __init__(self, cohorts: Dict[str, Cohort]):
        self.cohorts = cohorts

    def __iter__(self):
        return iter(self.cohorts)

    def __eq__(self, other):
        if not isinstance(other, CohortCollection):
            return False
        else:
            if not self.cohorts_names.issuperset(
                other.cohorts_names
            ) or not self.cohorts_names.issubset(other.cohorts_names):
                return False
            else:
                for name in self.cohorts_names:
                    if self.get(name) != other.get(name):
                        return False
                return True

    @property
    def cohorts_names(self) -> Set[str]:
        return set(self.cohorts.keys())

    def exists(self, cohort_name: str) -> bool:
        return cohort_name in self.cohorts

    def get(self, cohort_name: str) -> Cohort:
        if self.exists(cohort_name):
            return self.cohorts[cohort_name]
        else:
            raise KeyError(
                "Cohort {} do not exist in current Metadata".format(cohort_name)
            )

    def get_from_description(self, description: Dict) -> Cohort:
        # TODO : this should be a method called from cohorts
        operation_type = description["type"]  # type: str
        if operation_type.lower() not in ALLOWED_OPERATIONS:
            raise KeyError(
                "{} not permitted. Only available operations: {}".format(
                    operation_type, ALLOWED_OPERATIONS
                )
            )
        else:
            parents = [self.get(parent) for parent in description["parents"]]
            new_cohort = Cohort.union_all(parents)
            new_cohort.name = description["name"]
            return new_cohort

    def add_cohort(self, name: str, cohort: Cohort) -> "CohortCollection":
        new_cohort_collection = copy.copy(self)
        new_cohort_collection.cohorts[name] = cohort
        return new_cohort_collection

    def add_subjects_information(self, missing_patients, reference_date=None) -> None:
        """For the current cc it will fetch a base cohort that contains all the
        subjects with extra information and spread it through all cohorts.
        Warning: This mutate the state of the cohort within the CohortCollection.

        Parameters
        ----------
        reference_date: The study reference date used to compute age of subjects.
        If None don't add, if value use it to compute age of patients.
        missing_patients: behaviour from when missing patients are detected.
        possible values are "error" or "omit_all" to omit patients and their events
        or "omit_patients" to omit events and keep their events.

        Returns
        -------
        None. Mutation in place.
        """
        base_cohort_name, base_cohort = self._find_base_cohort()
        if reference_date is not None:
            base_cohort.add_age_information(reference_date)
        self._add_subjects_information(missing_patients, base_cohort_name, base_cohort)

    def union(self, other: "CohortCollection") -> "CohortCollection":
        return _union(self, other)

    def intersection(self, other: "CohortCollection") -> "CohortCollection":
        return _intersection(self, other)

    def difference(self, other: "CohortCollection") -> "CohortCollection":
        return _difference(self, other)

    def _find_base_cohort(self) -> Tuple[str, Cohort]:
        base_cohort = None
        max_count = 0
        base_cohort_name = None
        for name, cohort in self.cohorts.items():
            if cohort.has_subject_information() and cohort.subjects.count() > max_count:
                base_cohort = cohort
                max_count = cohort.subjects.count()
                base_cohort_name = name
        if max_count == 0:
            get_logger().error("There is no Base cohort in this Metadata")
        else:
            return base_cohort_name, base_cohort

    def _add_subjects_information(
        self, missing_patients, base_cohort_name, base_cohort
    ) -> None:
        for name, cohort in self.cohorts.items():
            if name != base_cohort_name and not cohort.has_subject_information():
                cohort.add_subject_information(base_cohort, missing_patients)

    @staticmethod
    def union_all(
        cohort_collections: Iterable["CohortCollection"]
    ) -> "CohortCollection":
        return fold_right(_union, cohort_collections)

    @staticmethod
    def intersect_all(
        cohort_collections: Iterable["CohortCollection"]
    ) -> "CohortCollection":
        return fold_right(_intersection, cohort_collections)

    @staticmethod
    def from_json(path: str) -> "CohortCollection":
        """Load CohortCollection object from JSON file."""
        with open(path, "r") as file:
            metadata = json.load(file)
        return CohortCollection.load(metadata)

    @staticmethod
    def load(input: Dict) -> "CohortCollection":
        """Load a CohortCollection object from a dict."""
        operations = input["operations"]
        return CohortCollection(
            {operation["name"]: Cohort.load(operation) for operation in operations}
        )

    def save(self, output_directory: str, mode="overwrite") -> Dict:
        """Saves the CohortCollection object to the output directory. Loops through
        the Cohorts, and for each writes it "output_directory/cohort_name".

        Parameters
        ----------
        output_directory
            Path to the root directory.
        mode
            Writing mode for parquet files.
            * ``append``: Append contents of this :class:`DataFrame` to existing data.
            * ``overwrite``(default case): Overwrite existing data.
            * ``ignore``: Silently ignore this operation if data already exists.
            * ``error`` or ``errorifexists``: Throw an exception if data already \
                exists.
        Returns
        -------
            Dict with one entry 'operations' containing a list of dict, each dict
            indicates where each cohort have been saved. It can be written in json
            and used as a cc file to load the CohortCollection directly.
        """
        operations = list()

        for name, cohort in self.cohorts.items():
            operations.append(cohort.save_cohort(output_directory, mode))

        return {"operations": operations}


def _union(a, b) -> CohortCollection:
    new_dict = copy.copy(a.cohorts)
    new_dict.update(b.cohorts)
    return CohortCollection(new_dict)


def _difference(a, b) -> CohortCollection:
    return CohortCollection({k: a.cohorts[k] for k in set(a) - set(b)})


def _intersection(a, b) -> CohortCollection:
    return CohortCollection({k: a.cohorts[k] for k in set(a).intersection(set(b))})
