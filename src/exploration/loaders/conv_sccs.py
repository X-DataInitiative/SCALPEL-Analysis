from datetime import datetime
from typing import Tuple, List

import numpy as np
import pyspark.sql.functions as sf
from pandas.core.series import Series
from pyspark.sql import DataFrame
from scipy.sparse import csr_matrix

from src.exploration.core.cohort import Cohort
from src.exploration.loaders.base import BaseLoader
from src.exploration.core.util import rename_df_columns, index_string_column

# TODO: (later) Implement trunc strategy for bucket rounding.
#  If bucket_rounding is equal to 'trunc', then this bucket is removed and
#  so are associated events. This last option might result in subjects loss.

# TODO later: add an option to keep only patients whose followup lasts at least a
#      minimum number of time periods?


class ConvSccsLoader(BaseLoader):
    """Load data read from cohorts to ConvSCCS format.

     The following cohorts should be provided: `base_population`, `followup`,
     `exposures`, `outcomes` (cf. description below).

    This is a loader, ie. it is not designed to do filtering or sanitizing. You
    are responsible for the inputs: all event timestamps should be in
    (study_start, study_end) and age_groups should match the ages of the
    base_population contained in the base cohort.

    :param base_population: `Cohort` Initial cohort containing the subjects for which we
        want to compute the data. This cohort should contain subjects, but its
        events are not used.
    :param followups: `Cohort` Cohort containing the follow up information. This cohort
        should contain subjects and events matching the FollowUp case class,
        that is the columns: 'patientID', 'start', 'stop', 'endReason'.
    :param exposures: `Cohort` Cohort containing the exposures information. This cohort
        should contain subjects and events matching the Event case class,
        that is the columns: 'patientID', 'category', 'groupID', 'value',
        'weight', 'start', 'end'.
    :param outcomes: `Cohort` Cohort containing the outcomes information. This cohort
        should contain subjects and events matching the Event case class,
        that is the columns: 'patientID', 'category', 'groupID', 'value',
        'weight', 'start', 'end'.
    :param study_start: `datetime` Date of the study start. Beware of timezones issues!
    :param study_end: `datetime` Date of the study end. Beware of timezones issues!
    :param bucket_size: `int` Size of the buckets used to disctretize time (expressed
        in number of days).
    :param age_reference_date: `datetime` Date used to compute the age when creating the
        base cohort.
    :param age_groups: `list[int]`, default=None Bounds defining longitudinal age
        groups. If set to None, age related data are not computed. Beware: take into
        account the ageing of subjects when defining these.
        Minimum bound should be <= minimum age, maximum bound should be >=
        maximum age + study length.
    :param bucket_rounding: `string`, default='ceil' Depending on bucket_size and study
        start / end, the last bucket might be shorter the previous ones.
        If bucket_rounding is equal to 'ceil', then this bucket is kept as is.
        If bucket_rounding is equal to 'floor', then this bucket is removed
        and its events are moved to the last full-sized bucket.
    :param  run_checks: `bool`, default=True Automated checks are performed on cohorts
        passed to the loaders. If you don't want these checks to be ran, set this option
        to False. Disabling the checks might increase performance, but use at your own
        risks!
    :param  exposures_split_column: `str`, default='value' Events field used to
        identify the different types of exposures when computing the features.
    :param  outcomes_split_column: `str`, default='value' Events field used to
        identify the different types of exposures when computing the outcomes.
    """

    def __init__(
        self,
        base_population: Cohort,
        followups: Cohort,
        exposures: Cohort,
        outcomes: Cohort,
        bucket_size: int,
        study_start: datetime,
        study_end: datetime,
        age_reference_date: datetime,
        age_groups: list = None,
        bucket_rounding="ceil",
        run_checks=True,
        exposures_split_column="value",
        outcomes_split_column="value",
    ):
        super().__init__(
            base_population,
            followups,
            study_start,
            study_end,
            age_reference_date,
            age_groups,
            run_checks,
        )
        self._is_using_longitudinal_age_groups = True
        self.bucket_size = bucket_size
        self._bucket_rounding = None
        self.bucket_rounding = bucket_rounding
        n_buckets = (study_end - study_start).days / bucket_size
        # Positivity of n_buckets is implied by condition
        # study_start < study_end implemented in BaseLoader
        self.n_buckets = int(
            np.ceil(n_buckets)
            if (self.bucket_rounding == "ceil")
            else np.floor(n_buckets)
        )
        # Cohorts
        self._exposures_split_column = None
        self._outcomes_split_column = None
        self.exposures_split_column = exposures_split_column
        self.outcomes_split_column = outcomes_split_column
        self._exposures = None
        self._outcomes = None
        self._final_cohort = None
        self.exposures = exposures
        self.outcomes = outcomes
        # output
        self._features = None
        self._labels = None
        self._censoring = None
        self._feature_mapping = []
        self._outcome_mapping = []

    def load(self) -> Tuple[List[csr_matrix], List[np.ndarray], np.ndarray]:
        assert len(self.features) == len(self.labels), (
            "Number of feature matrices does not match number of label matrices. "
            "You might want to investigate this"
        )
        assert len(self.features) == len(self.censoring), (
            "Number of feature matrices does not match number of censoring values. "
            "You might want to investigate this"
        )
        return self.features, self.labels, self.censoring

    @property
    def bucket_rounding(self) -> str:
        return self._bucket_rounding

    @bucket_rounding.setter
    def bucket_rounding(self, value: str) -> None:
        if value not in ["ceil", "floor"]:
            raise ValueError(
                "bucket_rounding should be equal to either 'ceil' or 'floor'"
            )
        self._bucket_rounding = value

    @property
    def exposures_split_column(self) -> str:
        return self._exposures_split_column

    @exposures_split_column.setter
    def exposures_split_column(self, value: str) -> None:
        if value not in ["category", "groupID", "value"]:
            raise ValueError(
                "exposures_split_column should be either "
                "'category', 'groupID', or 'value'"
            )
        else:
            self._exposures_split_column = value

    @property
    def outcomes_split_column(self) -> str:
        return self._outcomes_split_column

    @outcomes_split_column.setter
    def outcomes_split_column(self, value: str) -> None:
        if value not in ["category", "groupID", "value"]:
            raise ValueError(
                "outcomes_split_column should be either "
                "'category', 'groupID', or 'value'"
            )
        else:
            self._outcomes_split_column = value

    @property
    def exposures(self) -> Cohort:
        return self._exposures

    @exposures.setter
    def exposures(self, value: Cohort) -> None:
        if self.run_checks:
            invalid = self._find_events_not_in_followup_bounds(value)
            if invalid.events.take(1):
                raise ValueError(
                    self._log_invalid_events_cohort(invalid, log_invalid_events=True)
                )
        self._exposures = value

    @property
    def outcomes(self) -> Cohort:
        return self._outcomes

    @outcomes.setter
    def outcomes(self, value: Cohort) -> None:
        if self.run_checks:
            n_outcomes_types = (
                value.events.select(sf.col(self.outcomes_split_column))
                .drop_duplicates()
                .count()
            )
            assert n_outcomes_types == 1, (
                "There are more than one type of outcomes, check the 'value' field of"
                " outcomes cohort events."
            )

            invalid = self._find_events_not_in_followup_bounds(value)
            if invalid.events.take(1):
                raise ValueError(
                    self._log_invalid_events_cohort(invalid, log_invalid_events=True)
                )

            many_outcomes = self._find_subjects_with_many_outcomes(value)
            if many_outcomes.subjects.take(1):
                raise ValueError(
                    self._log_invalid_events_cohort(
                        many_outcomes, log_invalid_subjects=True
                    )
                )

        self._outcomes = value

    @property
    def final_cohort(self) -> Cohort:
        if self._final_cohort is None:
            final_cohort = self.base_population.intersect_all(
                [self.followups, self.exposures, self.outcomes]
            )
            final_cohort.add_subject_information(
                self.base_population, missing_patients="omit_all"
            )
            assert final_cohort.subjects.count() != 0, (
                "Final cohort is empty, please check that "
                "the intersection of the provided cohorts "
                "is nonempty"
            )
            self._final_cohort = final_cohort
        return self._final_cohort

    @final_cohort.setter
    def final_cohort(self, value: Cohort) -> None:
        raise PermissionError(
            "final_cohort should not be set manually,"
            "it is computed from initial cohorts."
        )

    @property
    def features(self) -> List[csr_matrix]:
        if self._features is None:
            self._features = self._load_features()
        return self._features

    @features.setter
    def features(self, value: List[csr_matrix]) -> None:
        raise PermissionError(
            "features should not be set manually,"
            "they are computed from initial cohorts."
        )

    @property
    def labels(self) -> List[np.ndarray]:
        if self._labels is None:
            self._labels = self._load_labels()
        return self._labels

    @labels.setter
    def labels(self, value: List[np.ndarray]) -> None:
        raise PermissionError(
            "labels should not be set manually,"
            "they are computed from initial cohorts."
        )

    @property
    def censoring(self) -> np.ndarray:
        if self._censoring is None:
            self._censoring = self._load_censoring()
        return self._censoring

    @censoring.setter
    def censoring(self, value: np.ndarray) -> None:
        raise PermissionError(
            "censoring should not be set manually,"
            "it is computed from initial cohorts."
        )

    @property
    def mappings(self) -> Tuple[List[str], List[str]]:
        return self._feature_mapping, self._outcome_mapping

    @mappings.setter
    def mappings(self, value: Tuple[List[str], List[str]]) -> None:
        raise PermissionError(
            "mappings should not be set manually,"
            "they are computed from initial cohorts."
        )

    def _load_features(self) -> List[csr_matrix]:
        cohort = self.exposures.intersection(self.final_cohort)

        features, n_cols, mapping = self._get_bucketized_events(
            cohort, self.exposures_split_column
        )
        self._feature_mapping = mapping

        if self.age_groups is not None:
            cohort.add_subject_information(self.final_cohort)
            age_features, mapping = self._compute_longitudinal_age_groups(
                cohort, n_cols
            )
            self._feature_mapping.extend(mapping)
            features = age_features.union(features)

        feature_matrix_shape = (int(self.n_buckets), int(n_cols + self.n_age_groups))
        features_matrices = self._get_csr_matrices(features, feature_matrix_shape)
        return features_matrices

    def _load_labels(self) -> List[np.ndarray]:
        cases = self.outcomes.intersection(self.final_cohort)

        labels, n_cols, mapping = self._get_bucketized_events(
            cases, self.outcomes_split_column
        )
        self._outcome_mapping = mapping

        labels_matrix_shape = (int(self.n_buckets), int(n_cols))
        label_matrices = [
            l.toarray() for l in self._get_csr_matrices(labels, labels_matrix_shape)
        ]
        return label_matrices

    def _load_censoring(self) -> np.ndarray:
        followups = self.followups.intersection(self.final_cohort)
        events = self._discretize_start_end(followups.events)

        censoring_ = (
            events.sort("patientID")
            .select("endBucket")
            .na.fill(self.n_buckets)
            .toPandas()
            .values
        )
        return censoring_

    def _get_bucketized_events(
        self, cohort: Cohort, split_column: str
    ) -> Tuple[DataFrame, int, List[str]]:
        events = self._discretize_start_end(cohort.events).select(
            "patientID", split_column, "startBucket", "endBucket"
        )
        events, n_cols, mapping = index_string_column(events, split_column, "colIndex")

        features_ = events.select(
            "patientID", sf.col("startBucket").alias("rowIndex"), sf.col("colIndex")
        )
        return features_, n_cols, mapping

    def _compute_longitudinal_age_groups(
        self, cohort: Cohort, col_offset: int
    ) -> Tuple[DataFrame, List[str]]:
        """
        :param cohort: cohort on which the age groups should be computed
        :param col_offset: number of columns used by lagged exposure features
        :return: (age_features, mapping) a dataframe containing the age features
            in aij format and a mapping giving the correspondance between column
            number and age group.
        """
        # This implementation is suboptimal, but we need to have something
        # working with inconsistent python versions across the cluster.
        assert (
            cohort.has_subject_information()
        ), "Cohort subjects should have gender and birthdate information"

        subjects = cohort.subjects.select("patientID", "gender", "birthDate")

        bucket_ids = sf.array([sf.lit(i) for i in range(self.n_buckets)])
        subjects = (
            subjects.withColumn("bucketID", bucket_ids)
            .select(
                "PatientID",
                "gender",
                "birthDate",
                sf.explode("bucketID").alias("bucket"),
            )
            .withColumn("dateShift", sf.col("bucket") * self.bucket_size)
            .withColumn("referenceDate", sf.lit(self.age_reference_date))
        )
        # Longitudinal age is based on referenceDate instead of minDate to
        # be consistent with cohort definition.
        time_references = sf.expr("date_add(referenceDate, dateShift)")
        longitudinal_age = sf.floor(
            sf.months_between(time_references, sf.col("birthdate")) / 12
        )
        subjects = subjects.withColumn("longitudinalAge", longitudinal_age).select(
            "patientID", "gender", "birthDate", "bucket", "longitudinalAge"
        )

        subjects, n_age_groups, mapping = self._bucketize_age_column(
            subjects, "longitudinalAge", "longitudinalAgeBucket"
        )

        assert n_age_groups == self.n_age_groups, (
            "Computed number of age groups is different from the number of specified"
            " age groups at initialization. There might be empty age_groups,"
            " you should investigate this."
        )

        age_features = subjects.select(
            sf.col("patientID"),
            sf.col("bucket").alias("rowIndex"),
            (sf.col("longitudinalAgeBucket") + col_offset).alias("colIndex"),
        )

        # Remove "age events" which are not in follow-up
        fup_events = self.followups.intersection(self.final_cohort).events
        fup_events = self._discretize_start_end(fup_events)
        fup_events = rename_df_columns(fup_events, prefix="fup_")
        age_features_columns = age_features.columns
        age_features = age_features.join(fup_events, on="patientID")
        age_features = age_features.where(
            sf.col("rowIndex").between(
                sf.col("fup_startBucket"), sf.col("fup_endBucket")
            )
        )
        age_features = age_features.select(*age_features_columns)

        return age_features, mapping

    def _discretize_time(self, column: sf.Column) -> sf.Column:
        days_since_study_start = sf.datediff(column, sf.lit(self.study_start))
        bucket = sf.floor(days_since_study_start / self.bucket_size).cast("int")

        if self.bucket_rounding == "floor":
            bucket = (
                sf.when((bucket < self.n_buckets) | bucket.isNull(), bucket)
                .otherwise(self.n_buckets - 1)
                .cast("int")
            )
        return bucket

    def _discretize_start_end(self, events: DataFrame) -> DataFrame:
        start_bucket = self._discretize_time(sf.col("start"))
        end_bucket = self._discretize_time(sf.col("end"))

        return events.withColumn("startBucket", start_bucket).withColumn(
            "endBucket", end_bucket
        )

    def _get_csr_matrices(
        self, events: DataFrame, csr_matrix_shape: Tuple[int, int]
    ) -> List[csr_matrix]:
        if "rowIndex" not in events.columns:
            raise ValueError("rowIndex should be in events columns.")
        if "colIndex" not in events.columns:
            raise ValueError("colIndex should be in events columns.")
        events = events.groupBy("patientID").agg(
            sf.collect_list("rowIndex").alias("rowIndexes"),
            sf.collect_list("colIndex").alias("colIndexes"),
        )

        events_df = events.sort("patientID").toPandas()

        return events_df.apply(
            self._create_csr_matrix,
            axis=1,
            reduce=True,
            csr_matrix_shape=csr_matrix_shape,
        )

    @staticmethod
    def _create_csr_matrix(
        df_row: Series, csr_matrix_shape: Tuple[int, int]
    ) -> csr_matrix:
        rows = df_row["rowIndexes"]
        cols = df_row["colIndexes"]
        data = np.ones_like(rows)
        csr = csr_matrix((data, (rows, cols)), shape=csr_matrix_shape)
        return csr

    @staticmethod
    def _find_subjects_with_many_outcomes(cohort: Cohort) -> Cohort:
        subjects_w_many_outcomes = (
            cohort.events.groupby("patientId")
            .count()
            .where(sf.col("count") > 1)
            .select("patientId")
            .drop_duplicates()
        )
        # between returns false when col is null
        invalid_events = cohort.events.join(subjects_w_many_outcomes, "patientId").sort(
            "patientId"
        )

        return Cohort(
            cohort.name + "_inconsistent_w_single_outcome_constraint",
            "events showing there are more than one outcome per patient",
            subjects_w_many_outcomes,
            invalid_events,
        )
