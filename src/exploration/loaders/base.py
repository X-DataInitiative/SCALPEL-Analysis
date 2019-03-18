from copy import copy
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pyspark.sql.functions as sf
import pytz
from pyspark.ml.feature import Bucketizer
from pyspark.sql import DataFrame

from src.exploration.core.cohort import Cohort
from src.exploration.core.util import rename_df_columns


class BaseLoader(object):
    """Base class of for loaders, which should load data from cohorts to a specific
    format.

    Loaders should not be designed to do filtering or sanitizing. Users are responsible
    for the inputs: all event timestamps should be in (study_start, study_end) and
    age_groups should match the ages of the base_population contained in the studied
    cohort.

    These checks should be implemented in the `check_metadata` method.

    There are important considerations to be aware of when working with timestamps and
    dataframes, see pyspark documentation:
    https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html#timestamp-with-time-zone-semantics

    Arguments:
    ----------
    :param base_population: `Cohort` Initial cohort containing the subjects for which
        we want to compute the data. This cohort should contain subjects, but its
        events are not used.

    :param followups: `Cohort` Cohort containing the follow up information. This cohort
        should contain subjects and events matching the FollowUp case class, that is the
         columns: 'patientID', 'start', 'end', 'endReason'.

    :param study_start: `datetime` Date of the study start. Beware of timezones issues!

    :param study_end: `datetime` Date of the study end. Beware of timezones issues!

    :param age_reference_date: `datetime` Date used to compute the age of the
        base_population contained in the cohorts.

    :param age_groups: `List[int]`, default=None Bounds defining longitudinal age
        groups. If set to None, age related data are not computed. These bounds must be
        sorted. Beware: take into account the ageing of base_population when defining
        these. Minimum bound should be <= minimum age, maximum bound should be >=
        maximum age + study length.

    :param run_checks: `bool`, default=True Automated checks are performed on cohorts
        passed to the loaders. If you don't want these checks to be ran, set this option
        to False. Disabling the checks might increase performance, but use at your own
        risks!
    """

    def __init__(
        self,
        base_population: Cohort,
        followups: Cohort,
        study_start: datetime,
        study_end: datetime,
        age_reference_date: datetime,
        age_groups: list = None,
        run_checks: bool = True,
    ):
        self.run_checks = run_checks
        self._study_start = None
        self._study_end = None
        self._is_using_longitudinal_age_groups = False

        if not self._has_timezone(study_start):
            raise ValueError("study_start should have a timezone. Please use pytz.")
        if not self._has_timezone(study_end):
            raise ValueError("study_end should have a timezone. Please use pytz.")
        if study_start >= study_end:
            raise ValueError("study_start should be < study_end")
        self._study_start = study_start
        self._study_end = study_end

        if not self._has_timezone(age_reference_date):
            raise ValueError(
                "age_reference_date should have a timezone. " "Please use pytz."
            )
        if age_reference_date < self.study_start:
            raise ValueError("age_reference_date should be >= study_start.")
        self._age_reference_date = age_reference_date
        self._age_groups = None
        self.age_groups = age_groups
        self.n_age_groups = len(age_groups) - 1 if age_groups is not None else 0

        self._followups = None
        self._base_population = None
        self.followups = followups
        self.base_population = base_population

    def load(self):
        raise NotImplementedError("load method is not implemented in BaseLoader.")

    @property
    def study_start(self):
        return self._study_start

    @study_start.setter
    def study_start(self, value):
        raise PermissionError(
            "study_start should not be updated after loader initialisation"
        )

    @property
    def study_end(self):
        return self._study_end

    @study_end.setter
    def study_end(self, value):
        raise PermissionError(
            "study_end should not be updated after loader initialisation"
        )

    @property
    def age_reference_date(self) -> datetime:
        return self._age_reference_date

    @age_reference_date.setter
    def age_reference_date(self, value: datetime) -> None:
        raise PermissionError(
            "age_reference_date should not be updated after loader initialisation."
        )

    @property
    def age_groups(self) -> List[float]:
        return self._age_groups

    @age_groups.setter
    def age_groups(self, value: List[float]) -> None:
        if value != sorted(value):
            raise ValueError("age_groups bounds should be sorted.")
        self._age_groups = value

    @property
    def base_population(self) -> Cohort:
        return self._base_population

    @base_population.setter
    def base_population(self, value: Cohort) -> None:
        if self.run_checks and self.age_groups is not None:
            invalid = self._find_subjects_with_age_inconsistent_w_age_groups(value)
            if invalid.subjects.take(1):
                raise ValueError(
                    self._log_invalid_events_cohort(invalid, log_invalid_subjects=True)
                )
        self._base_population = value

    @property
    def followups(self) -> Cohort:
        return self._followups

    @followups.setter
    def followups(self, value: Cohort) -> None:
        if self.run_checks:
            invalid = self._find_events_not_in_study_dates(value)
            if invalid.events.take(1):
                raise ValueError(
                    self._log_invalid_events_cohort(invalid, log_invalid_events=True)
                )
            invalid = self._find_inconsistent_start_end_ordering(value)
            if invalid.events.take(1):
                raise ValueError(
                    self._log_invalid_events_cohort(invalid, log_invalid_events=True)
                )
        self._followups = value

    @property
    def is_using_longitudinal_age_groups(self) -> bool:
        return self._is_using_longitudinal_age_groups

    @is_using_longitudinal_age_groups.setter
    def is_using_longitudinal_age_groups(self, value: bool) -> None:
        raise PermissionError(
            "is_using_longitudinal_age_groups should not be set manually."
        )

    def _bucketize_age_column(
        self, dataframe: DataFrame, input_col: str, output_col: str
    ) -> Tuple[DataFrame, int, List[str]]:
        bucketizer = Bucketizer(
            splits=self.age_groups, inputCol=input_col, outputCol=output_col
        )
        output = bucketizer.setHandleInvalid("keep").transform(dataframe)
        splits = [s for s in bucketizer.getSplits()]
        mapping = [
            "[{}, {})".format(splits[i], splits[i + 1]) for i in range(len(splits) - 1)
        ]
        n_age_groups = len(mapping)
        return output, n_age_groups, mapping

    def _find_events_not_in_followup_bounds(self, cohort: Cohort) -> Cohort:
        fups = copy(self.followups)
        fups.events = rename_df_columns(fups.events, prefix="fup_")
        events = cohort.events.join(fups.events, "patientID")
        # between returns false when col is null
        invalid_events = events.where(
            ~(
                sf.col("start").between(sf.col("fup_start"), sf.col("fup_end"))
                & sf.col("end").between(sf.col("fup_start"), sf.col("fup_end"))
            )
        )
        return Cohort(
            cohort.name + "_inconsistent_w_followup_bounds",
            "events inconsistent with followup bounds",
            invalid_events.select("patientID").distinct(),
            invalid_events,
        )

    def _find_events_not_in_study_dates(self, cohort: Cohort) -> Cohort:
        # between returns false when col is null
        invalid_events = cohort.events.where(
            ~(
                sf.col("start").between(
                    sf.lit(self.study_start), sf.lit(self.study_end)
                )
                & sf.col("end").between(
                    sf.lit(self.study_start), sf.lit(self.study_end)
                )
            )
        )
        return Cohort(
            cohort.name + "_inconsistent_w_study_dates",
            "events inconsistent with study dates",
            invalid_events.select("patientID").distinct(),
            invalid_events,
        )

    def _find_subjects_with_age_inconsistent_w_age_groups(
        self, cohort: Cohort
    ) -> Cohort:
        """Check if min and max age_groups are consistent with subjects ages."""
        if not cohort.has_subject_information():
            raise ValueError("Cohort should have subject information.")
        duplicate = copy(cohort)
        duplicate.add_age_information(self.age_reference_date)  # add starting age
        study_length = (
            np.ceil((self.study_end - self.study_start).days / 365.25)
            if self.is_using_longitudinal_age_groups
            else 0
        )
        min_starting_age = min(self.age_groups)
        max_starting_age = max(self.age_groups) - np.ceil(study_length)
        invalid_subjects = duplicate.subjects.where(
            ~sf.col("age").between(min_starting_age, max_starting_age)
        )
        return Cohort(
            cohort.name + "_inconsistent_w_ages_and_age_groups",
            "subjects inconsistent with age groups",
            invalid_subjects,
        )

    @staticmethod
    def _find_inconsistent_start_end_ordering(cohort: Cohort) -> Cohort:
        events = cohort.events
        invalid_events = events.where(sf.col("start") >= sf.col("end"))
        return Cohort(
            cohort.name + "_inconsistent_w_start_end_ordering",
            "events where start >= end dates are inconsistent",
            invalid_events.select("patientID").distinct(),
            invalid_events,
        )

    @staticmethod
    def _log_invalid_events_cohort(
        cohort: Cohort,
        log_invalid_events: bool = False,
        log_invalid_subjects: bool = False,
    ) -> str:
        cohort_name, reference = cohort.name.split("_inconsistent_w_")
        n_subjects = cohort.subjects.count()
        msg = (
            "Found {n_subjects} subjects in cohort {cohort_name} inconsistent with"
            " {reference}.\n".format(
                n_subjects=n_subjects, cohort_name=cohort_name, reference=reference
            )
        )
        if log_invalid_events:
            msg += "Showing first 10 invalid events below:\n"
            msg += cohort.events.limit(10).toPandas().to_string(index=False)
            msg += "\n"
        if log_invalid_subjects:
            msg += "Showing first 10 invalid subjects below:\n"
            msg += cohort.subjects.limit(10).toPandas().to_string(index=False)
            msg += "\n"
        return msg

    @staticmethod
    def _has_timezone(date: pytz.datetime.datetime) -> bool:
        """Check if date has timezone."""
        return date.tzinfo is not None
