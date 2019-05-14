import copy
import json
from typing import Set, Tuple, Dict, Iterable

from src.exploration.core.cohort import Cohort
from src.exploration.core.io import get_logger
from src.exploration.core.util import fold_right

ALLOWED_OPERATIONS = frozenset(["union", "intersection", "difference"])


class Metadata:
    """Encaspulates the Metadata generated from SNIIRAM-featuring. Represents a set of
    Cohorts."""

    def __init__(self, cohorts: Dict[str, Cohort]):
        self.cohorts = cohorts

    @staticmethod
    def from_json(input: str) -> "Metadata":
        metadata_json = json.loads(input)
        operations = metadata_json["operations"]
        return Metadata(
            {operation["name"]: Cohort.from_json(operation) for operation in operations}
        )

    def get_from_description(self, description: Dict) -> Cohort:
        # TODO : this should be a method called from cohortz
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

    def add_cohort(self, name: str, cohort: Cohort) -> "Metadata":
        new_metadata = copy.copy(self)
        new_metadata.cohorts[name] = cohort
        return new_metadata

    def union(self, other: "Metadata") -> "Metadata":
        return _union(self, other)

    def difference(self, other: "Metadata") -> "Metadata":
        return _difference(self, other)

    def intersection(self, other: "Metadata") -> "Metadata":
        return _intersection(self, other)

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

    def add_subjects_information(self, missing_patients, reference_date=None) -> None:
        """For the current metadata it will fetch a base cohort that contains all the
        subjects with extra information and spread it through all cohorts.
        Warning: This mutate the state of the cohort within the metadata.
        :param reference_date: The study reference date used to compute age of subjects.
        If None don't add, if value use it to compute age of patients.
        :param missing_patients: behaviour from when missing patients are detected.
        possible values are "error" or "omit_all" to omit patients and their events
        or "omit_patients" to omit events and keep their events.
        :return None
        """
        base_cohort_name, base_cohort = self._find_base_cohort()
        if reference_date is not None:
            base_cohort.add_age_information(reference_date)
        self._add_subjects_information(missing_patients, base_cohort_name, base_cohort)

    @staticmethod
    def union_all(metadatas: Iterable["Metadata"]) -> "Metadata":
        return fold_right(_union, metadatas)

    @staticmethod
    def intersect_all(metadatas: Iterable["Metadata"]) -> "Metadata":
        return fold_right(_intersection, metadatas)

    def __iter__(self):
        return iter(self.cohorts)

    def dump_metadata(self, output_directory: str, mode="overwrite") -> Dict:
        """Dump the current the Metadata object to the output directory. Loops through
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
            Dict with one entry operations that contains a list of dict, one dict for
            each Cohort.
        """
        operations = list()

        for name, cohort in self.cohorts.items():
            operations.append(cohort.save_cohort(output_directory, mode))

        return {"operations": operations}


def _union(a, b) -> Metadata:
    new_dict = copy.copy(a.cohorts)
    new_dict.update(b.cohorts)
    return Metadata(new_dict)


def _difference(a, b) -> Metadata:
    return Metadata({k: a.cohorts[k] for k in set(a) - set(b)})


def _intersection(a, b) -> Metadata:
    return Metadata({k: a.cohorts[k] for k in set(a).intersection(set(b))})
