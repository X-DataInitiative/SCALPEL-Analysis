from datetime import datetime
from typing import Iterable, Dict

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, months_between, floor, lit

from src.exploration.core.io import read_data_frame, get_logger
from src.exploration.core.util import data_frame_equality
from .util import fold_right


class Cohort(object):
    """A data representation that encapsulates a cohort. A cohort is a set of
    subjects who experienced a common event in a selected time period.
    """

    def __init__(self, name: str, characteristics: str, subjects: DataFrame,
                 events: DataFrame = None):
        self._name = name
        self._characteristics = characteristics
        self._subjects = subjects
        self._events = events

    @staticmethod
    def from_json(input: Dict) -> 'Cohort':
        if input["output_type"] == "patients":
            return Cohort(input["name"], input["name"],
                          read_data_frame(input["output_path"]))
        else:
            return Cohort(input["name"], "Subjects with event {}".format(input["name"]),
                          read_data_frame(input["population_path"]),
                          read_data_frame(input["output_path"]))

    @staticmethod
    def from_description(description: str) -> 'Cohort':
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._name = value

    @property
    def characteristics(self) -> str:
        return self._characteristics

    @characteristics.setter
    def characteristics(self, value):
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._characteristics = value

    @property
    def subjects(self) -> DataFrame:
        return self._subjects

    @subjects.setter
    def subjects(self, value):
        if not isinstance(value, DataFrame):
            raise TypeError('Expected a Spark DataFrame')
        self._subjects = value

    @property
    def events(self) -> DataFrame:
        return self._events

    @events.setter
    def events(self, value):
        if not isinstance(value, DataFrame):
            raise TypeError('Expected a Spark DataFrame')
        self._events = value

    def describe(self) -> str:
        if self.events is None:
            return ("This a subject cohort, no event needed" +
                    "Subjects are from operation {}".format(self.name))
        else:
            return ("Events are {}.".format(self.name) +
                    "Events contain only {}".format(self.characteristics))

    def union(self, other: 'Cohort') -> 'Cohort':
        return _union(self, other)

    def intersection(self, other: 'Cohort') -> 'Cohort':
        return _intersection(self, other)

    def difference(self, other: 'Cohort') -> 'Cohort':
        return _difference(self, other)

    def has_subject_information(self) -> bool:
        """Returns true if this cohort is the Base Cohort. The base population contains
        extra columns specifically birthDate, deathDate and gender"""
        return set(self.subjects.columns) == {"gender", "patientID", "deathDate",
                                              "birthDate"}

    def add_subject_information(self, base_cohort: 'Cohort',
                                missing_patients="error") -> None:
        """
        Add information of gender and birthDate to subjects in place.
        WARNING: CHANGE OF STATE.
        :param missing_patients: behaviour from when missing patients are detected.
        possible values are "error" or "omit_all" to omit patients and their events
        or "omit_patients" to omit events and keep their events.
        :param base_cohort: cohort that contain information.
        :return: None. Mutation in place.
        """
        if missing_patients == "error":
            subjects = self.subjects.join(base_cohort.subjects, on="patientID",
                                          how="left")
            extra_subjects_count = subjects.where(col("gender").isNull()).count()
            if extra_subjects_count > 0:
                raise ValueError(
                    "Cohort {} contains {} subjects not in base cohort {}".format(
                        self.name, extra_subjects_count, base_cohort.name))
            else:
                self._subjects = subjects
        elif missing_patients == "omit_all":
            get_logger().warning("Some patients and their events might"
                                 + " be ommited")
            self._subjects = self.subjects.join(base_cohort.subjects, on="patientID",
                                                how="inner")
            if self.events is not None:
                self._events = self.events.join(
                    self.subjects.select("patientID").distinct(),
                    on="patientID",
                    how="inner")
        elif missing_patients == "omit":
            get_logger().warning("Some patients might be ommited."
                                 + " Their events are kept")
            self._subjects = self.subjects.join(base_cohort.subjects, on="patientID",
                                                how="inner")
        else:
            raise ValueError("missing_patients is erroneous. Possible options are "
                             + "error, omit, omit_all")

    def add_age_information(self, date: datetime) -> None:
        self._subjects = self.subjects.withColumn("age", floor(
            months_between(lit(date), col("birthDate")) / 12)).withColumn(
            "ageBucket", floor(col("age") / 5)
        )

    @staticmethod
    def union_all(cohorts: Iterable['Cohort']) -> 'Cohort':
        return fold_right(_union, cohorts)

    @staticmethod
    def intersect_all(cohorts: Iterable['Cohort']) -> 'Cohort':
        return fold_right(_intersection, cohorts)

    @staticmethod
    def difference_all(cohorts: Iterable['Cohort']) -> 'Cohort':
        return fold_right(_difference, cohorts)

    def __eq__(self, other):
        if isinstance(other, Cohort):
            if self.events is not None and other.events is not None:
                return (data_frame_equality(self.subjects, other.subjects)
                        and data_frame_equality(self.events, other.events))
            else:
                return data_frame_equality(self.subjects, other.subjects)
        else:
            return False


def _union(a: Cohort, b: Cohort) -> Cohort:
    if a.events is None or b.events is None:
        return Cohort("{} Or {}".format(a.name, b.name),
                      "{} Or {}".format(a.characteristics, b.characteristics),
                      a.subjects.union(b.subjects))
    else:
        return Cohort("{} Or {}".format(a.name, b.name),
                      "{} Or {}".format(a.characteristics, b.characteristics),
                      a.subjects.union(b.subjects), a.events.union(b.events))


def _intersection(a: Cohort, b: Cohort) -> Cohort:
    subjects_id = a.subjects.select("patientID").intersect(b.subjects.select("patientID"))
    subjects = a.subjects.join(subjects_id, on="patientID", how="right")
    events = None
    if a.events is not None:
        events = a.events.join(subjects_id, on="patientID", how='right')
    return Cohort(a.name,
                  "{} with {}".format(a.characteristics, b.characteristics),
                  subjects,
                  events)


def _difference(a: Cohort, b: Cohort) -> Cohort:
    subjects_id = a.subjects.select("patientID").subtract(b.subjects.select("patientID"))
    subjects = a.subjects.join(subjects_id, on="patientID", how="right")
    events = None
    if a.events is not None:
        events = a.events.join(subjects_id, on="patientID", how='right')
    return Cohort(a.name,
                  "{} without {}".format(a.characteristics, b.characteristics),
                  subjects,
                  events)
