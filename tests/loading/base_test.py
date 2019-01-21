import pandas as pd
import pytz
import numpy as np
from copy import copy

from src.exploration.loaders.base import BaseLoader
from src.exploration.core.cohort import Cohort
from tests.exploration.core.pyspark_tests import PySparkTest
from src.exploration.core.util import data_frame_equality


class TestBaseLoader(PySparkTest):
    def setUp(self):
        super().setUp()
        self.study_start = pytz.datetime.datetime(2010, 2, 5, tzinfo=pytz.UTC)
        self.study_end = pytz.datetime.datetime(2013, 10, 12, tzinfo=pytz.UTC)
        self.age_reference_date = pytz.datetime.datetime(2011, 9, 21, tzinfo=pytz.UTC)
        self.age_groups = [65, 60, 75, 70, 80, np.Inf]
        self.sorted_age_groups = sorted(self.age_groups)

        self.patients = {
            "patientID": ["0", "1", "2", "3", "4"],  # uuid
            "gender": [1, 2, 2, 2, 1],  # in {1, 2}
            "birthDate": [
                pytz.datetime.datetime(1934, 7, 27, tzinfo=pytz.UTC),
                pytz.datetime.datetime(1951, 5, 1, tzinfo=pytz.UTC),
                pytz.datetime.datetime(1942, 1, 12, tzinfo=pytz.UTC),
                pytz.datetime.datetime(1933, 10, 3, tzinfo=pytz.UTC),
                pytz.datetime.datetime(1937, 12, 31, tzinfo=pytz.UTC),
            ],
            "deathDate": [
                None,
                None,
                None,
                pytz.datetime.datetime(2011, 6, 20, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2012, 12, 10, tzinfo=pytz.UTC),
            ],  # can be null
        }

        self.followup_events = {
            "patientID": ["0", "3", "4", "2"],  # uuid
            "start": [
                pytz.datetime.datetime(2010, 2, 5, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2011, 3, 27, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2011, 7, 2, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2010, 11, 21, tzinfo=pytz.UTC),
            ],
            "stop": [
                pytz.datetime.datetime(2013, 10, 12, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2011, 6, 20, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2012, 7, 3, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2013, 10, 12, tzinfo=pytz.UTC),
            ],
            "endReason": ["ObservationEnd", "Trackloss", "Death", "ObservationEnd"],
        }

        self.patients_df, _ = self.create_spark_df(self.patients)
        self.fup_events_df, _ = self.create_spark_df(self.followup_events)

        self.base_population = Cohort(
            "base_population", "base_population", self.patients_df, None
        )

        self.followups = Cohort(
            "followups",
            "followups",
            self.fup_events_df.select("patientID").distinct(),
            self.fup_events_df,
        )

        self.kwargs = {
            "base_population": self.base_population,
            "followups": self.followups,
            "study_start": self.study_start,
            "study_end": self.study_end,
            "age_reference_date": self.age_reference_date,
            "age_groups": self.sorted_age_groups,
            "run_checks": False,
        }

    def test_bucketize_age_column(self):
        # Expected
        pandas_df = pd.DataFrame(self.patients)
        bucketized = pd.cut(
            pandas_df.birthDate.apply(
                lambda date: (self.age_reference_date - date).days // 365.25
            ),
            self.sorted_age_groups,
            right=False,
        )
        expected = bucketized.cat.codes.values
        expected_map = [cat.__str__() for cat in bucketized.cat.categories.values]
        expected_n = len(expected_map)

        # Loader
        self.base_population.add_age_information(self.age_reference_date)
        loader = BaseLoader(**self.kwargs)
        df, n, mapping = loader._bucketize_age_column(
            self.base_population.subjects, "age", "bucketizedAge"
        )
        result = df.select("bucketizedAge").toPandas().values

        self.assertListEqual(result.ravel().tolist(), expected.ravel().tolist())
        self.assertListEqual(mapping, expected_map)
        self.assertEqual(expected_n, n)

    # Test checks
    def test_check_consistency_with_study_dates(self):
        corrupted_events_pre_study = {
            "patientID": ["0", "1"],  # uuid
            "start": [
                pytz.datetime.datetime(1934, 7, 27, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2012, 7, 27, tzinfo=pytz.UTC),
            ],
            "stop": [
                pytz.datetime.datetime(2012, 10, 12, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2006, 6, 20, tzinfo=pytz.UTC),
            ],
        }

        corrupted_events_post_study = {
            "patientID": ["0", "1"],  # uuid
            "start": [
                pytz.datetime.datetime(2017, 7, 27, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2012, 7, 27, tzinfo=pytz.UTC),
            ],
            "stop": [
                pytz.datetime.datetime(2012, 10, 12, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2014, 6, 20, tzinfo=pytz.UTC),
            ],
        }

        corrupted_pre, _ = self.create_spark_df(corrupted_events_pre_study)
        corrupted_post, _ = self.create_spark_df(corrupted_events_post_study)

        loader = BaseLoader(**self.kwargs)
        with self.assertRaises(Exception) as context:
            loader._check_consistency_with_study_dates(
                Cohort("", "", corrupted_pre.select("patientID"), corrupted_pre),
                ["start", "stop"],
            )

        self.assertTrue("Found date < study_start." in str(context.exception))

        with self.assertRaises(Exception) as context:
            loader._check_consistency_with_study_dates(
                Cohort("", "", corrupted_post.select("patientID"), corrupted_post),
                ["start", "stop"],
            )

        self.assertTrue("Found date > study_end." in str(context.exception))

        # The following should not raise an error
        loader._check_consistency_with_study_dates(self.followups, ["start", "stop"])

    def test_is_before_study_start(self):
        amsterdam = pytz.timezone("Europe/Amsterdam")
        date_ok = pytz.datetime.datetime(2012, 5, 1)
        date_ok_weird_tz = pytz.datetime.datetime(2012, 5, 1, tzinfo=amsterdam)
        date_error = pytz.datetime.datetime(2009, 5, 1)
        date_error_weird_tz = pytz.datetime.datetime(2009, 5, 1, tzinfo=amsterdam)
        loader = BaseLoader(**self.kwargs)
        self.assertFalse(loader._is_before_study_start(date_ok))
        self.assertFalse(loader._is_before_study_start(date_ok_weird_tz))
        self.assertTrue(loader._is_before_study_start(date_error))
        self.assertTrue(loader._is_before_study_start(date_error_weird_tz))

    def test_is_after_study_end(self):
        amsterdam = pytz.timezone("Europe/Amsterdam")
        date_ok = pytz.datetime.datetime(2012, 5, 1)
        date_ok_weird_tz = pytz.datetime.datetime(2012, 5, 1, tzinfo=amsterdam)
        date_error = pytz.datetime.datetime(2015, 5, 1)
        date_error_weird_tz = pytz.datetime.datetime(2015, 5, 1, tzinfo=amsterdam)
        loader = BaseLoader(**self.kwargs)
        self.assertFalse(loader._is_after_study_end(date_ok))
        self.assertFalse(loader._is_after_study_end(date_ok_weird_tz))
        self.assertTrue(loader._is_after_study_end(date_error))
        self.assertTrue(loader._is_after_study_end(date_error_weird_tz))

    def test_compare_with_followup_bounds(self):
        data_ok = {
            "patientID": ["0", "6"],  # uuid
            "start": [
                pytz.datetime.datetime(2011, 7, 1, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2012, 7, 27, tzinfo=pytz.UTC),
            ],
            "end": [
                pytz.datetime.datetime(2012, 10, 12, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2006, 6, 20, tzinfo=pytz.UTC),
            ],
        }
        # The second value contains invalid dates, but patient 6
        # does not have a fup anyway. This check function only checks validity
        # for patients who have a fup. Raise an error if that is not the case?

        data_error = {
            "patientID": ["0", "3", "5"],  # uuid
            "start": [
                pytz.datetime.datetime(2017, 7, 27, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2011, 4, 12, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2006, 7, 27, tzinfo=pytz.UTC),
            ],
            "end": [
                pytz.datetime.datetime(2012, 10, 12, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2014, 6, 20, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2015, 7, 27, tzinfo=pytz.UTC),
            ],
        }

        df, _ = self.create_spark_df(data_ok)
        df_error, _ = self.create_spark_df(data_error)
        cohort = Cohort("", "", df.select("patientID").distinct(), df)
        loader = BaseLoader(**self.kwargs)
        loader._check_event_dates_consistency_w_followup_bounds(cohort)

        with self.assertRaises(Exception) as context:
            cohort_error = Cohort(
                "", "", df_error.select("patientID").distinct(), df_error
            )
            loader._check_event_dates_consistency_w_followup_bounds(cohort_error)

        self.assertTrue(
            "Cohort contains 2 events (concerns "
            "2 patients) which are not between "
            "followup start and followup end." in str(context.exception)
        )

    def test_has_timezone(self):
        with_tz = pytz.datetime.datetime(
            1890, 1, 29, 3, 20, 15, tzinfo=pytz.timezone("US/Eastern")
        )
        without_tz = pytz.datetime.datetime(1890, 1, 29, 3, 20, 15)
        self.assertTrue(BaseLoader._has_timezone(with_tz))
        self.assertFalse(BaseLoader._has_timezone(without_tz))

    def test_rename_columns(self):
        fup_events = self.followups.events
        cols = fup_events.columns

        res = BaseLoader._rename_columns(fup_events, prefix="pre_")
        expected = ["pre_" + c if c != "patientID" else c for c in cols]
        self.assertListEqual(res.columns, expected)

        res = BaseLoader._rename_columns(fup_events, suffix="_suf")
        expected = [c + "_suf" if c != "patientID" else c for c in cols]
        self.assertListEqual(res.columns, expected)

        expected = "the quick brown".split(" ")
        res = BaseLoader._rename_columns(fup_events, new_names=expected)
        expected.insert(0, "patientID")
        self.assertListEqual(res.columns, expected)

    def test_index_string_column(self):
        some_events = {
            "patientID": ["0", "1", "1"],  # uuid
            "start": [
                pytz.datetime.datetime(1934, 7, 27, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2012, 7, 27, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2012, 8, 26, tzinfo=pytz.UTC),
            ],
            "stop": [
                pytz.datetime.datetime(2012, 10, 12, tzinfo=pytz.UTC),
                None,
                pytz.datetime.datetime(2012, 8, 30, tzinfo=pytz.UTC),
            ],
            "value": ["foo", "bar", "foo"],
        }

        data, _ = self.create_spark_df(some_events)
        expected_mapping = list(sorted(set(some_events["value"])))
        expected_index = [1, 0, 1]
        expected_n_categories = len(expected_mapping)

        indexed, n_cat, mapping = BaseLoader._index_string_column(
            data, "value", "valueIndex"
        )
        index = indexed.toPandas().valueIndex.values.tolist()
        self.assertEqual(expected_n_categories, n_cat)
        self.assertListEqual(expected_mapping, mapping)
        self.assertListEqual(expected_index, index)

    def test_load(self):
        loader = BaseLoader(**self.kwargs)
        with self.assertRaises(NotImplementedError) as context:
            loader.load()

        self.assertTrue(
            "load method is not implemented in" " BaseLoader." in str(context.exception)
        )

    def test_properties(self):
        kwargs = copy(self.kwargs)
        kwargs["run_checks"] = True
        loader = BaseLoader(**kwargs)
        # Study start / end
        self.assertEqual(self.study_start, loader.study_start)
        self.assertEqual(self.study_end, loader.study_end)

        with self.assertRaises(PermissionError) as context:
            loader.study_start = self.study_start

        self.assertTrue(
            "study_start should not be modified after loader initialisation"
            in str(context.exception)
        )

        with self.assertRaises(PermissionError) as context:
            loader.study_end = self.study_end

        self.assertTrue(
            "study_end should not be modified after loader initialisation"
            in str(context.exception)
        )

        with self.assertRaises(ValueError) as context:
            kwargs["study_start"] = self.study_end
            kwargs["study_end"] = self.study_start
            loader = BaseLoader(**kwargs)

        self.assertTrue("study_start should be < study_end" in str(context.exception))

        with self.assertRaises(ValueError) as context:
            kwargs["study_start"] = self.study_end
            loader = BaseLoader(**kwargs)

        self.assertTrue("study_start should be < study_end" in str(context.exception))

        # Age groups
        self.assertListEqual(self.sorted_age_groups, loader.age_groups)

        with self.assertRaises(ValueError) as context:
            loader.age_groups = self.age_groups

        self.assertTrue("age_groups bounds should be sorted." in str(context.exception))

        # feature_mapping
        with self.assertRaises(NotImplementedError) as context:
            loader.feature_mapping = ["foo", "bar", "baz"]

        self.assertTrue(
            "feature_mapping should not be set manually." in str(context.exception)
        )

        loader._feature_mapping = ["foo", "bar", "baz"]
        self.assertListEqual(loader.feature_mapping, ["foo", "bar", "baz"])

        # base_population
        self.assertTrue(
            data_frame_equality(
                self.base_population.subjects, loader.base_population.subjects
            )
        )
        self.assertTrue(loader.base_population.events is None)

        loader = BaseLoader(
            self.base_population,
            self.followups,
            self.study_start,
            self.study_end,
            self.age_reference_date,
            self.sorted_age_groups,
            run_checks=True,
        )

        other = Cohort(
            "other_population", "other_population", self.patients_df.limit(2), None
        )  # Note the None here

        loader.base_population = other
        self.assertTrue(
            data_frame_equality(other.subjects, loader.base_population.subjects)
        )
        self.assertTrue(loader.base_population.events is None)  # Hence this one

        # followups
        self.assertTrue(
            data_frame_equality(self.followups.subjects, loader.followups.subjects)
        )
        self.assertTrue(
            data_frame_equality(self.followups.events, loader.followups.events)
        )

        other = Cohort(
            "other_fups",
            "other_fups",
            self.fup_events_df.limit(3).select("patientID").distinct(),
            self.fup_events_df.limit(3),
        )

        loader.followups = other

        self.assertTrue(data_frame_equality(other.subjects, loader.followups.subjects))

        self.assertTrue(data_frame_equality(other.events, loader.followups.events))

        # is_using_longitudinal_age_groups
        self.assertFalse(loader.is_using_longitudinal_age_groups)

        with self.assertRaises(NotImplementedError) as context:
            loader.is_using_longitudinal_age_groups = True

        self.assertTrue(
            "is_using_longitudinal_age_groups should not be set "
            "manually." in str(context.exception)
        )

    def test_check_followups_start_end_ordering_exception(self):
        valid_events = {
            "patientID": ["0", "1"],  # uuid
            "start": [
                pytz.datetime.datetime(2011, 7, 2, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2012, 9, 30, tzinfo=pytz.UTC),
            ],
            "end": [
                pytz.datetime.datetime(2012, 10, 12, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2013, 6, 20, tzinfo=pytz.UTC),
            ],
        }

        invalid_events = {
            "patientID": ["0", "1"],  # uuid
            "start": [
                pytz.datetime.datetime(2011, 7, 2, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2012, 9, 30, tzinfo=pytz.UTC),
            ],
            "end": [
                pytz.datetime.datetime(2012, 10, 12, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2011, 6, 20, tzinfo=pytz.UTC),
            ],
        }

        valid_df, _ = self.create_spark_df(valid_events)
        invalid_df, _ = self.create_spark_df(invalid_events)

        BaseLoader._check_followups_start_end_ordering(
            Cohort("", "", valid_df.select("patientID"), valid_df)
        )

        with self.assertRaises(Exception) as context:
            BaseLoader._check_followups_start_end_ordering(
                Cohort("", "", invalid_df.select("patientID"), invalid_df)
            )

        self.assertTrue(
            (
                "Cohort contains 1 followups "
                "(concerns 1 patients) for which"
                "followup start >= followup end."
            )
            in str(context.exception)
        )

    def test_check_subjects_age_consistency_w_age_groups(self):
        cohort_wo_subject_info = Cohort(
            "base_population",
            "base_population",
            self.patients_df.select("patientID"),
            None,
        )
        loader = BaseLoader(**self.kwargs)

        with self.assertRaises(Exception) as context:
            loader._check_subjects_age_consistency_w_age_groups(cohort_wo_subject_info)

        self.assertTrue(
            "Cohort should have subject information." in str(context.exception)
        )

        too_young_patients, _ = self.create_spark_df(
            {
                "patientID": ["0", "1"],  # uuid
                "gender": [1, 2],  # in {1, 2}
                "birthDate": [
                    pytz.datetime.datetime(1934, 7, 27, tzinfo=pytz.UTC),
                    pytz.datetime.datetime(1999, 5, 1, tzinfo=pytz.UTC),
                ],
                "deathDate": [
                    None,
                    pytz.datetime.datetime(2012, 12, 10, tzinfo=pytz.UTC),
                ],  # can be null
            }
        )

        too_young_cohort = Cohort(
            "base_population", "base_population", too_young_patients, None
        )

        too_old_patients, _ = self.create_spark_df(
            {
                "patientID": ["0", "1"],  # uuid
                "gender": [1, 2],  # in {1, 2}
                "birthDate": [
                    pytz.datetime.datetime(1900, 7, 27, tzinfo=pytz.UTC),
                    pytz.datetime.datetime(1950, 5, 1, tzinfo=pytz.UTC),
                ],
                "deathDate": [
                    None,
                    pytz.datetime.datetime(2012, 12, 10, tzinfo=pytz.UTC),
                ],  # can be null
            }
        )

        too_old_cohort = Cohort(
            "base_population", "base_population", too_old_patients, None
        )

        patients, _ = self.create_spark_df(
            {
                "patientID": ["0", "1"],  # uuid
                "gender": [1, 2],  # in {1, 2}
                "birthDate": [
                    pytz.datetime.datetime(1940, 7, 27, tzinfo=pytz.UTC),
                    pytz.datetime.datetime(1950, 5, 1, tzinfo=pytz.UTC),
                ],
                "deathDate": [
                    None,
                    pytz.datetime.datetime(2012, 12, 10, tzinfo=pytz.UTC),
                ],  # can be null
            }
        )

        patients_cohort = Cohort("base_population", "base_population", patients, None)

        age_groups = [60, 65, 70, 75, 80]
        age_reference_date = pytz.datetime.datetime(2011, 9, 21, tzinfo=pytz.UTC)
        study_start = pytz.datetime.datetime(2011, 1, 1, tzinfo=pytz.UTC)
        study_end = pytz.datetime.datetime(2013, 12, 31, tzinfo=pytz.UTC)

        # Case 1: no longitudinal age groups
        loader = BaseLoader(
            patients_cohort,
            self.followups,
            study_start,
            study_end,
            age_reference_date,
            age_groups,
            run_checks=False,
        )

        loader._check_subjects_age_consistency_w_age_groups(patients_cohort)

        with self.assertRaises(Exception) as context:
            loader._check_subjects_age_consistency_w_age_groups(too_young_cohort)

        self.assertTrue(
            "Found patients whose age is < min(age_group)" in str(context.exception)
        )

        with self.assertRaises(Exception) as context:
            loader._check_subjects_age_consistency_w_age_groups(too_old_cohort)

        self.assertTrue(
            "Found patients whose age is > max(age_group)."
            "Not that max(age_group) is corrected using "
            "study_length when working with longitudinal"
            "age groups." in str(context.exception)
        )

        loader._is_using_longitudinal_age_groups = True

        with self.assertRaises(Exception) as context:
            loader._check_subjects_age_consistency_w_age_groups(too_young_cohort)

        self.assertTrue(
            "Found patients whose age is < min(age_group)" in str(context.exception)
        )

        with self.assertRaises(Exception) as context:
            loader._check_subjects_age_consistency_w_age_groups(too_old_cohort)

        self.assertTrue(
            "Found patients whose age is > max(age_group)."
            "Not that max(age_group) is corrected using "
            "study_length when working with longitudinal"
            "age groups." in str(context.exception)
        )
