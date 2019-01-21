from copy import copy
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pyspark.sql.functions as sf
import pytz
from pyspark.ml.feature import Bucketizer, StringIndexer
from pyspark.sql import DataFrame

from src.exploration.core.cohort import Cohort


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
         columns: 'patientID', 'start', 'stop', 'endReason'.

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
        self._feature_mapping = []

        if self.run_checks:
            self._has_timezone(study_start)
            self._has_timezone(study_end)
            if study_start >= study_end:
                raise ValueError("study_start should be < study_end")
        self._study_start = study_start
        self._study_end = study_end

        if not run_checks:
            self._has_timezone(age_reference_date)
        self.age_reference_date = age_reference_date
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
            "study_start should not be modified after loader initialisation"
        )

    @property
    def study_end(self):
        return self._study_end

    @study_end.setter
    def study_end(self, value):
        raise PermissionError(
            "study_end should not be modified after loader initialisation"
        )

    @property
    def age_groups(self) -> List[float]:
        return self._age_groups

    @age_groups.setter
    def age_groups(self, value: List[float]) -> None:
        if self.run_checks and value != sorted(value):
            raise ValueError("age_groups bounds should be sorted.")
        self._age_groups = value

    @property
    def feature_mapping(self) -> List[str]:
        return self._feature_mapping

    @feature_mapping.setter
    def feature_mapping(self, value: List) -> None:
        raise NotImplementedError("feature_mapping should not be set manually.")

    @property
    def base_population(self) -> Cohort:
        return self._base_population

    @base_population.setter
    def base_population(self, value: Cohort) -> None:
        if self.run_checks and self.age_groups is not None:
            self._check_subjects_age_consistency_w_age_groups(value)
        self._base_population = value

    @property
    def followups(self) -> Cohort:
        return self._followups

    @followups.setter
    def followups(self, value: Cohort) -> None:
        # Why on earth end of followup is called stop when everything else's end
        # is called end? WTF?
        value_ = copy(value)
        value_.events = value_.events.withColumnRenamed("stop", "end")
        if self.run_checks:
            self._check_consistency_with_study_dates(value_, ["start", "end"])
            self._check_followups_start_end_ordering(value_)
        self._followups = value_

    @property
    def is_using_longitudinal_age_groups(self) -> bool:
        return self._is_using_longitudinal_age_groups

    @is_using_longitudinal_age_groups.setter
    def is_using_longitudinal_age_groups(self, value: bool) -> None:
        raise NotImplementedError(
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

    def _is_before_study_start(self, date: pytz.datetime.datetime) -> bool:
        if not self._has_timezone(date):
            date = pytz.UTC.localize(date)
        else:
            date = pytz.UTC.normalize(date)
        return date < self.study_start

    def _is_after_study_end(self, date: pytz.datetime.datetime) -> bool:
        if not self._has_timezone(date):
            date = pytz.UTC.localize(date)
        else:
            date = pytz.UTC.normalize(date)
        return date > self.study_end

    def _check_event_dates_consistency_w_followup_bounds(self, other: Cohort) -> None:
        fups = copy(self.followups)
        fups.events = self._rename_columns(fups.events, prefix="fup_")
        events = other.events.join(fups.events, "patientID")
        events.select(sf.countDistinct("patientID"))
        # between returns false when col is null
        invalid_events = events.where(
            ~(
                sf.col("start").between(sf.col("fup_start"), sf.col("fup_end"))
                & sf.col("end").between(sf.col("fup_start"), sf.col("fup_end"))
            )
        )
        [[patients_count, events_count]] = invalid_events.select(
            sf.countDistinct("patientID"), sf.count("patientID")
        ).collect()
        if events_count != 0:
            raise ValueError(
                (
                    "Cohort contains {n_events} events (concerns "
                    "{n_patients} patients) which are not between "
                    "followup start and followup end."
                ).format(n_events=events_count, n_patients=patients_count)
            )

    def _check_consistency_with_study_dates(
        self, cohort: Cohort, column_names: list
    ) -> None:
        """Check that events start and end dates are within study dates."""
        columns = [sf.min(c) for c in column_names]
        columns.extend([sf.max(c) for c in column_names])  # no proper flatmap
        extrema = cohort.events.select(columns).toPandas()
        minima = extrema[[col for col in extrema if col.startswith("min")]]
        maxima = extrema[[col for col in extrema if col.startswith("max")]]
        if np.any(minima.applymap(self._is_before_study_start)):
            raise ValueError("Found date < study_start.")
        if np.any(maxima.applymap(self._is_after_study_end)):
            raise ValueError("Found date > study_end.")

    def _check_subjects_age_consistency_w_age_groups(self, cohort: Cohort) -> None:
        """Check if min and max age_groups are consistent with subjects ages."""
        if not cohort.has_subject_information():
            raise ValueError("Cohort should have subject information.")
        duplicate = copy(cohort)
        duplicate.add_age_information(self.age_reference_date)
        ages_extrema = duplicate.subjects.select(
            sf.min("age"), sf.max("age")
        ).toPandas()
        [[min_age]] = ages_extrema[["min(age)"]].values
        [[max_age]] = ages_extrema[["max(age)"]].values
        if self.is_using_longitudinal_age_groups:
            study_length = np.ceil((self.study_end - self.study_start).days / 365)
        else:
            study_length = 0

        if min_age < min(self.age_groups):
            raise ValueError("Found patients whose age is < min(age_group)")

        if (max_age + study_length) > max(self.age_groups):
            raise ValueError(
                "Found patients whose age is > max(age_group)."
                "Not that max(age_group) is corrected using "
                "study_length when working with longitudinal"
                "age groups."
            )

    @staticmethod
    def _check_followups_start_end_ordering(cohort: Cohort) -> None:
        """Check that start < end for each event. If end is null, ignore row."""
        events = cohort.events
        invalid_events = events.where(sf.col("start") >= sf.col("end"))
        [[events_count, patients_count]] = invalid_events.select(
            sf.countDistinct("patientID"), sf.count("patientID")
        ).collect()
        if events_count != 0:
            raise ValueError(
                (
                    "Cohort contains {n_events} followups "
                    "(concerns {n_patients} patients) for which"
                    "followup start >= followup end."
                ).format(n_events=events_count, n_patients=patients_count)
            )

    @staticmethod
    def _has_timezone(date: pytz.datetime.datetime) -> None:
        """Check if date has timezone."""
        return date.tzinfo is not None

    @staticmethod
    def _rename_columns(
        df: DataFrame,
        new_names: List[str] = None,
        prefix: str = "",
        suffix: str = "",
        keys: Tuple[str] = ("patientID",),
    ) -> DataFrame:
        """Rename columns of a pyspark DataFrame.

        :param df: dataframe whose columns will be renamed
        :param new_names: If not None, these name will replace the old ones.
         The order should be the same as df.columns where the keys has been
         removed.
        :param prefix: Prefix added to colnames.
        :param suffix: Suffix added to colnames.
        :param keys: Columns whose name will not be modified (useful for joining
         keys for example).
        :return: Dataframe with renamed columns.
        """
        old_names = [c for c in df.columns if c not in keys]
        if new_names is None:
            new_names = [prefix + c + suffix for c in old_names]
        return df.select(
            *keys, *[sf.col(c).alias(new_names[i]) for i, c in enumerate(old_names)]
        )

    @staticmethod
    def _index_string_column(
        dataframe: DataFrame, input_col: str, output_col: str
    ) -> Tuple[DataFrame, int, List[str]]:
        """Add a column containing an index corresponding to a string column.

        :param dataframe: Dataframe on which add index column.
        :param input_col: Name of the column to index
        :param output_col: Name of the index column in the resulting dataframe.
        :return: (resulting_dataframe, n_values_in_index, index_mapping)
        """
        indexer = StringIndexer(
            inputCol=input_col, outputCol=output_col, stringOrderType="alphabetAsc"
        )
        model = indexer.fit(dataframe)
        output = model.transform(dataframe)

        mapping = model.labels
        n_categories = len(mapping)

        return output, n_categories, mapping
