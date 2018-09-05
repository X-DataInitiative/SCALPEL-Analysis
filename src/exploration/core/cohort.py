from typing import Iterable, Dict

from pyspark.sql import DataFrame

from src.exploration.core.io import read_data_frame
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
            Cohort(input["name"], input["name"],
                   read_data_frame(input["output_path"]))
        else:
            Cohort(input["name"], "Subjects_with_event_{}".format(input["name"]),
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

    def union(self, other: 'Cohort') -> 'Cohort':
        return _union(self, other)

    def intersection(self, other: 'Cohort') -> 'Cohort':
        return _intersection(self, other)

    def difference(self, other: 'Cohort') -> 'Cohort':
        return _difference(self, other)

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
        return Cohort("{}_Or_{}".format(a.name, b.name),
                      "{}_Or_{}".format(a.characteristics, b.characteristics),
                      a.subjects.union(b.subjects))
    else:
        return Cohort("{}_Or_{}".format(a.name, b.name),
                      "{}_Or_{}".format(a.characteristics, b.characteristics),
                      a.subjects.union(b.subjects), a.events.union(b.events))


def _intersection(a: Cohort, b: Cohort) -> Cohort:
    subjects = a.subjects.intersect(b.subjects)
    events = None
    if a.events is not None:
        events = a.events.join(subjects.select("patientID"), on="patientID")
    return Cohort(a.name,
                  "{}_with_{}".format(a.characteristics, b.characteristics),
                  subjects,
                  events)


def _difference(a: Cohort, b: Cohort) -> Cohort:
    subjects = a.subjects.subtract(b.subjects)
    events = None
    if a.events is not None:
        events = a.events.join(subjects.select("patientID"), on="patientID")
    return Cohort(a.name,
                  "{}_without_{}".format(a.characteristics, b.characteristics),
                  subjects,
                  events)
