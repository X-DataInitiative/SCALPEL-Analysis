from copy import copy
from datetime import datetime
from typing import Tuple, List
from warnings import warn

import numpy as np
import pyspark.sql.functions as sf
from pandas.core.series import Series
from pyspark.sql import DataFrame
from scipy.sparse import csr_matrix

from src.exploration.core.cohort import Cohort
from src.exploration.loaders.base import BaseLoader

# TODO: (later) Implement trunc strategy for bucket rounding
# TODO later: add an option to keep only patients whose followup lasts at least a
#      minimum number of periods


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
        If bucket_rounding is equal to 'trunc', then this bucket is removed and
        so are associated events. This last option might result in subjects loss
        (not implemented yet).
    :param  run_checks: `bool`, default=True Automated checks are performed on cohorts
        passed to the loaders. If you don't want these checks to be ran, set this option
        to False. Disabling the checks might increase performance, but use at your own
        risks!
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
            np.ceil(n_buckets) if (bucket_rounding == "ceil") else np.floor(n_buckets)
        )
        # Cohorts
        self._exposures = None
        self._outcomes = None
        self._final_cohort = None
        self.exposures = exposures
        self.outcomes = outcomes
        # output
        self._features = None
        self._labels = None
        self._censoring = None

    def load(self) -> Tuple[List[csr_matrix], List[np.ndarray], np.ndarray]:
        if len(self.features) != len(self.labels):
            raise AssertionError(
                "Number of feature matrices does not match "
                "number of label matrices. You might want to"
                " investigate this"
            )
        if len(self.features) != len(self.censoring):
            raise AssertionError(
                "Number of feature matrices does not match "
                "number of censoring values. You might want to"
                " investigate this"
            )
        return self.features, self.labels, self.censoring

    @property
    def bucket_rounding(self) -> str:
        return self._bucket_rounding

    @bucket_rounding.setter
    def bucket_rounding(self, value: str) -> None:
        if value not in ["ceil", "floor", "trunc"]:
            raise ValueError(
                "bucket_rounding should be equal to either 'ceil',"
                "'floor', or 'trunc'"
            )
        if value == "trunc":
            raise NotImplementedError("Not implemented yet :( ")
        self._bucket_rounding = value

    @property
    def exposures(self) -> Cohort:
        return self._exposures

    @exposures.setter
    def exposures(self, value: Cohort) -> None:
        if self.run_checks:
            self._check_event_dates_consistency_w_followup_bounds(value)
        self._exposures = value

    @property
    def outcomes(self) -> Cohort:
        return self._outcomes

    @outcomes.setter
    def outcomes(self, value: Cohort) -> None:
        if self.run_checks:
            self._check_event_dates_consistency_w_followup_bounds(value)
        warn(
            "At this moment, ConvSccsLoader considers only the first outcome for each "
            "subject, per category and groupID. Latter outcomes are ignored."
        )
        duplicate = copy(value)
        duplicate.events = (
            duplicate.events.groupby(
                "patientID", "category", "groupID", "value", "weight"
            )
            .agg(sf.min("start").alias("start"), sf.min("end").alias("end"))
            .select(
                "patientID", "start", "end", "category", "groupID", "value", "weight"
            )
        )
        self._outcomes = duplicate

    @property
    def final_cohort(self) -> Cohort:
        if self._final_cohort is None:
            final_cohort = self.base_population.intersect_all(
                [self.followups, self.exposures, self.outcomes]
            )
            final_cohort.add_subject_information(
                self.base_population, missing_patients="omit_all"
            )
            if final_cohort.subjects.count() == 0:
                raise AssertionError(
                    "Final cohort is empty, please check that "
                    "the intersection of the provided cohorts "
                    "is nonempty"
                )
            self._final_cohort = final_cohort
        return self._final_cohort

    @final_cohort.setter
    def final_cohort(self, value: Cohort) -> None:
        raise NotImplementedError(
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
        raise NotImplementedError(
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
        raise NotImplementedError(
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
        raise NotImplementedError(
            "censoring should not be set manually,"
            "it is computed from initial cohorts."
        )

    def _load_features(self) -> List[csr_matrix]:
        cohort = self.exposures.intersection(self.final_cohort)
        cohort.add_subject_information(self.final_cohort)

        events = self._discretize_start_end(cohort.events).select(
            "patientID", "value", "startBucket", "endBucket"
        )
        events, n_cols, mapping = self._index_string_column(
            events, "value", "valueIndex"
        )
        self._feature_mapping = mapping

        features_ = events.select(
            "patientID",
            sf.col("startBucket").alias("rowIndex"),
            sf.col("valueIndex").alias("colIndex"),
        )

        if self.age_groups is not None:
            age_features, mapping = self._compute_longitudinal_age_groups(
                cohort, n_cols
            )
            self._feature_mapping.extend(mapping)
            features_ = age_features.union(features_)

        # Remove events which are not in fup ; should be done only for age events
        # It should not be a problem for exposures as exposure events are checked
        # to be consistent with fups
        fup_events = self.followups.intersection(self.final_cohort).events
        fup_events = self._discretize_start_end(fup_events)
        fup_events = self._rename_columns(fup_events, prefix="fup_")
        features_columns = features_.columns
        features_ = features_.join(fup_events, on="patientID")
        features_ = features_.where(
            sf.col("rowIndex").between(
                sf.col("fup_startBucket"), sf.col("fup_endBucket")
            )
        )
        features_ = features_.select(*features_columns)

        feature_matrix_shape = (int(self.n_buckets), int(n_cols + self.n_age_groups))

        features_matrices = self._get_csr_matrices(features_, feature_matrix_shape)

        return features_matrices

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
        if not cohort.has_subject_information():
            raise AssertionError(
                "Cohort subjects should have gender and birthdate information"
            )

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

        assert n_age_groups == self.n_age_groups

        age_features = subjects.select(
            sf.col("patientID"),
            sf.col("bucket").alias("rowIndex"),
            (sf.col("longitudinalAgeBucket") + col_offset).alias("colIndex"),
        )

        return age_features, mapping

    def _load_labels(self) -> List[np.ndarray]:
        cases = self.outcomes.intersection(self.final_cohort)
        events = self._discretize_start_end(cases.events)

        events, n_cols, _ = self._index_string_column(events, "value", "valueIndex")
        events = events.select(
            "patientID",
            sf.col("startBucket").alias("rowIndex"),
            sf.col("valueIndex").alias("colIndex"),
        )

        labels_matrix_shape = (int(self.n_buckets), int(n_cols))

        labels_ = [
            l.toarray() for l in self._get_csr_matrices(events, labels_matrix_shape)
        ]

        return labels_

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
