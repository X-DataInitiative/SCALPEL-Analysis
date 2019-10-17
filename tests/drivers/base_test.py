# License: BSD 3 clause

import pandas as pd
import pytz
import numpy as np
from copy import copy
from unittest.mock import patch, MagicMock

from scalpel.drivers.base import BaseFeatureDriver
from scalpel.core.cohort import Cohort
from tests.core.pyspark_tests import PySparkTest
from scalpel.core.util import data_frame_equality


class TestBaseFeatureDriver(PySparkTest):
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
            "end": [
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

        self.mock_dataframe = MagicMock()
        self.mock_dataframe.take = lambda x: True
        self.mock_cohort = MagicMock()
        self.mock_cohort.subjects = self.mock_dataframe
        self.mock_cohort.events = self.mock_dataframe
        self.mock_empty_df = MagicMock()
        self.mock_empty_df.take = lambda x: []
        self.mock_empty_cohort = MagicMock()
        self.mock_empty_cohort.subjects = self.mock_empty_df
        self.mock_empty_cohort.events = self.mock_empty_df

    def test_bucketize_age_column(self):
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

        self.base_population.add_age_information(self.age_reference_date)
        loader = BaseFeatureDriver(**self.kwargs)
        df, n, mapping = loader._bucketize_age_column(
            self.base_population.subjects, "age", "bucketizedAge"
        )
        result = df.select("bucketizedAge").toPandas().values

        self.assertListEqual(result.ravel().tolist(), expected.ravel().tolist())
        self.assertListEqual(mapping, expected_map)
        self.assertEqual(expected_n, n)

    def test_find_events_not_in_study_dates(self):
        invalid_events = {
            "patientID": ["0", "1", "2", "2", "3"],  # uuid
            "start": [
                pytz.datetime.datetime(1934, 7, 27, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2012, 7, 27, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2017, 7, 27, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2012, 7, 27, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2011, 7, 27, tzinfo=pytz.UTC),
            ],
            "end": [
                pytz.datetime.datetime(2012, 10, 12, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2006, 6, 20, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2012, 10, 12, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2014, 6, 20, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2012, 7, 27, tzinfo=pytz.UTC),
            ],
            "value": [0, 1, 2, 3, 4],
        }

        invalid_df, _ = self.create_spark_df(invalid_events)
        invalid_cohort = Cohort(
            "some_cohort", "Some cohort", invalid_df.select("patientID"), invalid_df
        )

        loader = BaseFeatureDriver(**self.kwargs)
        invalid = loader._find_events_not_in_study_dates(invalid_cohort)

        self.assertEqual(invalid.name, "some_cohort_inconsistent_w_study_dates")
        self.assertEqual(
            invalid.describe(),
            "Events are some_cohort_inconsistent_w_study_dates. Events contain only "
            "events inconsistent with study dates.",
        )
        self.assertListEqual(
            sorted(invalid.subjects.toPandas().values.ravel().tolist()),
            sorted(["0", "1", "2"]),
        )
        self.assertListEqual(
            sorted(invalid.events.toPandas().value.values.ravel().tolist()),
            sorted([0, 1, 2, 3]),
        )

    def test_find_events_not_in_followup_bounds(self):
        data = {
            "patientID": ["0", "6", "2", "3", "5"],  # uuid
            "start": [
                pytz.datetime.datetime(2011, 7, 1, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2012, 7, 27, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2017, 7, 27, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2011, 4, 12, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2006, 7, 27, tzinfo=pytz.UTC),
            ],
            "end": [
                pytz.datetime.datetime(2012, 10, 12, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2006, 6, 20, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2012, 10, 12, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2014, 6, 20, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2015, 7, 27, tzinfo=pytz.UTC),
            ],
            "value": [0, 1, 2, 3, 4],
        }

        df, _ = self.create_spark_df(data)
        cohort = Cohort(
            "some_cohort", "Some cohort", df.select("patientID").distinct(), df
        )
        loader = BaseFeatureDriver(**self.kwargs)
        invalid = loader._find_events_not_in_followup_bounds(cohort)
        self.assertEqual(invalid.name, "some_cohort_inconsistent_w_followup_bounds")
        self.assertEqual(
            invalid.describe(),
            "Events are some_cohort_inconsistent_w_followup_bounds. Events contain "
            "only events inconsistent with followup bounds.",
        )
        self.assertListEqual(
            sorted(invalid.subjects.toPandas().values.ravel().tolist()),
            sorted(["2", "3"]),
        )
        self.assertListEqual(
            sorted(invalid.events.toPandas().value.values.ravel().tolist()),
            sorted([2, 3]),
        )

    def test_find_inconsistent_start_end_ordering(self):
        events = {
            "patientID": ["0", "1", "2", "2"],  # uuid
            "start": [
                pytz.datetime.datetime(2011, 7, 2, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2012, 9, 30, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2011, 7, 2, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2012, 9, 30, tzinfo=pytz.UTC),
            ],
            "end": [
                pytz.datetime.datetime(2012, 10, 12, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2013, 6, 20, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2012, 10, 12, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2011, 6, 20, tzinfo=pytz.UTC),
            ],
            "value": [0, 1, 2, 3],
        }

        df, _ = self.create_spark_df(events)
        cohort = Cohort(
            "some_cohort", "Some cohort", df.select("patientID").distinct(), df
        )

        invalid = BaseFeatureDriver._find_inconsistent_start_end_ordering(cohort)

        self.assertEqual(invalid.name, "some_cohort_inconsistent_w_start_end_ordering")
        self.assertEqual(
            invalid.describe(),
            "Events are some_cohort_inconsistent_w_start_end_ordering. Events contain "
            "only events where start >= end dates are inconsistent.",
        )
        self.assertListEqual(
            sorted(invalid.subjects.toPandas().values.ravel().tolist()), sorted(["2"])
        )
        self.assertListEqual(
            sorted(invalid.events.toPandas().value.values.ravel().tolist()), sorted([3])
        )

    def test_find_subjects_with_age_inconsistent_w_age_groups(self):
        cohort_wo_subject_info = Cohort(
            "base_population",
            "base_population",
            self.patients_df.select("patientID"),
            None,
        )

        with self.assertRaises(ValueError) as context:
            loader = BaseFeatureDriver(**self.kwargs)
            _ = loader._find_subjects_with_age_inconsistent_w_age_groups(
                cohort_wo_subject_info
            )

        self.assertTrue(
            "Cohort should have subject information." in str(context.exception)
        )

        data = {
            "patientID": ["0", "1", "2", "3"],  # uuid
            "gender": [1, 2, 1, 2],  # in {1, 2}
            "birthDate": [
                pytz.datetime.datetime(1934, 7, 27, tzinfo=pytz.UTC),
                pytz.datetime.datetime(1999, 5, 1, tzinfo=pytz.UTC),
                pytz.datetime.datetime(1900, 7, 27, tzinfo=pytz.UTC),
                pytz.datetime.datetime(1950, 5, 1, tzinfo=pytz.UTC),
            ],
            "deathDate": [
                None,
                pytz.datetime.datetime(2012, 12, 10, tzinfo=pytz.UTC),
                None,
                pytz.datetime.datetime(2012, 12, 10, tzinfo=pytz.UTC),
            ],  # can be null
        }
        df, _ = self.create_spark_df(data)
        cohort = Cohort("some_cohort", "Some cohort", df.distinct(), None)

        kwargs = copy(self.kwargs)
        kwargs["study_start"] = pytz.datetime.datetime(2011, 1, 1, tzinfo=pytz.UTC)
        kwargs["study_end"] = pytz.datetime.datetime(2015, 12, 31, tzinfo=pytz.UTC)
        kwargs["age_groups"] = [60, 65, 70, 75, 80]
        loader = BaseFeatureDriver(**kwargs)

        # Case 1: no longitudinal age groups
        invalid = loader._find_subjects_with_age_inconsistent_w_age_groups(cohort)

        self.assertEqual(invalid.name, "some_cohort_inconsistent_w_ages_and_age_groups")
        self.assertEqual(
            invalid.describe(),
            "This a subject cohort, no event needed. Subjects are from operation "
            "some_cohort_inconsistent_w_ages_and_age_groups.",
        )
        self.assertListEqual(
            sorted(invalid.subjects.toPandas().patientID.values.ravel().tolist()),
            sorted(["1", "2"]),
        )
        self.assertIsNone(invalid.events)

        # Case 2: longitudinal age groups
        loader._is_using_longitudinal_age_groups = True
        invalid = loader._find_subjects_with_age_inconsistent_w_age_groups(cohort)

        self.assertEqual(invalid.name, "some_cohort_inconsistent_w_ages_and_age_groups")
        self.assertEqual(
            invalid.describe(),
            "This a subject cohort, no event needed. Subjects are from operation"
            " some_cohort_inconsistent_w_ages_and_age_groups.",
        )
        self.assertListEqual(
            sorted(invalid.subjects.toPandas().patientID.values.ravel().tolist()),
            sorted(["0", "1", "2"]),
        )
        self.assertIsNone(invalid.events)

    #
    # def test_log_invalid_events_cohort(self):
    #     events = OrderedDict(
    #         [
    #             ("patientID", ["0", "1", "2", "2"]),  # uuid
    #             (
    #                 "start",
    #                 [
    #                     pytz.datetime.datetime(2011, 7, 2, tzinfo=pytz.UTC),
    #                     pytz.datetime.datetime(2012, 9, 30, tzinfo=pytz.UTC),
    #                     pytz.datetime.datetime(2011, 7, 2, tzinfo=pytz.UTC),
    #                     pytz.datetime.datetime(2012, 9, 30, tzinfo=pytz.UTC),
    #                 ],
    #             ),
    #             (
    #                 "end",
    #                 [
    #                     pytz.datetime.datetime(2012, 10, 12, tzinfo=pytz.UTC),
    #                     pytz.datetime.datetime(2013, 6, 20, tzinfo=pytz.UTC),
    #                     pytz.datetime.datetime(2012, 10, 12, tzinfo=pytz.UTC),
    #                     pytz.datetime.datetime(2011, 6, 20, tzinfo=pytz.UTC),
    #                 ],
    #             ),
    #             ("value", [0, 1, 2, 3]),
    #         ]
    #     )
    #
    #     df, _ = self.create_spark_df(events)
    #     invalid = Cohort(
    #         "some_cohort_inconsistent_w_start_end_ordering",
    #         "events where start >= end dates are inconsistent",
    #         df.select("patientID").distinct().orderBy("patientID"),
    #         df.orderBy("patientID", "start"),
    #     )
    #
    #     msg = BaseLoader._log_invalid_events_cohort(invalid)
    #     self.assertEqual(
    #         msg,
    #         "Found 3 subjects in cohort some_cohort inconsistent "
    #         "with start_end_ordering.\n",
    #     )
    #     msg = BaseLoader._log_invalid_events_cohort(invalid, log_invalid_events=True)
    #     self.assertEqual(
    #         msg,
    #         "Found 3 subjects in cohort some_cohort inconsistent with "
    #         "start_end_ordering.\n"
    #         "Showing first 10 invalid events below:\n"
    #         "patientID      start        end  value\n"
    #         "        0 2011-07-02 2012-10-12      0\n"
    #         "        1 2012-09-30 2013-06-20      1\n"
    #         "        2 2011-07-02 2012-10-12      2\n"
    #         "        2 2012-09-30 2011-06-20      3\n",
    #     )
    #     msg = BaseLoader._log_invalid_events_cohort(invalid, log_invalid_subjects=True)
    #     self.assertEqual(
    #         msg,
    #         "Found 3 subjects in cohort some_cohort inconsistent with "
    #         "start_end_ordering.\n"
    #         "Showing first 10 invalid subjects below:\n"
    #         "patientID\n"
    #         "        0\n"
    #         "        1\n"
    #         "        2\n",
    #     )
    #     msg = BaseLoader._log_invalid_events_cohort(
    #         invalid, log_invalid_events=True, log_invalid_subjects=True
    #     )
    #     self.assertEqual(
    #         msg,
    #         "Found 3 subjects in cohort some_cohort inconsistent with "
    #         "start_end_ordering.\n"
    #         "Showing first 10 invalid events below:\n"
    #         "patientID      start        end  value\n"
    #         "       0 2011-07-02 2012-10-12      0\n"
    #         "       1 2012-09-30 2013-06-20      1\n"
    #         "       2 2011-07-02 2012-10-12      2\n"
    #         "       2 2012-09-30 2011-06-20      3\n"
    #         "Showing first 10 invalid subjects below:\n"
    #         "patientID\n"
    #         "       0\n"
    #         "       1\n"
    #         "       2\n",
    #     )
    #
    def test_has_timezone(self):
        with_tz = pytz.datetime.datetime(
            1890, 1, 29, 3, 20, 15, tzinfo=pytz.timezone("US/Eastern")
        )
        without_tz = pytz.datetime.datetime(1890, 1, 29, 3, 20, 15)
        self.assertTrue(BaseFeatureDriver._has_timezone(with_tz))
        self.assertFalse(BaseFeatureDriver._has_timezone(without_tz))

    def test_load(self):
        loader = BaseFeatureDriver(**self.kwargs)
        with self.assertRaises(NotImplementedError) as context:
            loader.load()

        self.assertTrue(
            "load method is not implemented in BaseLoader." in str(context.exception)
        )

    def test_properties_study_start_end(self):
        loader = BaseFeatureDriver(**self.kwargs)
        self.assertEqual(self.study_start, loader.study_start)
        self.assertEqual(self.study_end, loader.study_end)

        with self.assertRaises(PermissionError) as context:
            loader.study_start = self.study_start
        self.assertTrue(
            "study_start should not be updated after loader initialisation"
            in str(context.exception)
        )

        with self.assertRaises(PermissionError) as context:
            loader.study_end = self.study_end
        self.assertTrue(
            "study_end should not be updated after loader initialisation"
            in str(context.exception)
        )

        with self.assertRaises(ValueError) as context:
            kwargs_ = copy(self.kwargs)
            kwargs_["study_start"] = self.study_end
            kwargs_["study_end"] = self.study_start
            _ = BaseFeatureDriver(**kwargs_)
        self.assertTrue("study_start should be < study_end" in str(context.exception))

        with self.assertRaises(ValueError) as context:
            kwargs_ = copy(self.kwargs)
            kwargs_["study_start"] = self.study_end
            _ = BaseFeatureDriver(**kwargs_)
        self.assertTrue("study_start should be < study_end" in str(context.exception))

        with self.assertRaises(ValueError) as context:
            kwargs_ = copy(self.kwargs)
            kwargs_["study_start"] = pytz.datetime.datetime(2011, 1, 1)
            _ = BaseFeatureDriver(**kwargs_)
        self.assertTrue(
            "study_start should have a timezone. Please use pytz."
            in str(context.exception)
        )

        with self.assertRaises(ValueError) as context:
            kwargs_ = copy(self.kwargs)
            kwargs_["study_end"] = pytz.datetime.datetime(2017, 1, 1)
            _ = BaseFeatureDriver(**kwargs_)
        self.assertTrue(
            "study_end should have a timezone. Please use pytz."
            in str(context.exception)
        )

    def test_properties_age_reference_date(self):
        with self.assertRaises(ValueError) as context:
            kwargs_ = copy(self.kwargs)
            kwargs_["age_reference_date"] = pytz.datetime.datetime(2012, 1, 1)
            _ = BaseFeatureDriver(**kwargs_)
        self.assertTrue(
            "age_reference_date should have a timezone. Please use pytz."
            in str(context.exception)
        )

        with self.assertRaises(ValueError) as context:
            kwargs_ = copy(self.kwargs)
            kwargs_["age_reference_date"] = pytz.datetime.datetime(
                2000, 1, 1, tzinfo=pytz.UTC
            )
            _ = BaseFeatureDriver(**kwargs_)
        self.assertTrue(
            "age_reference_date should be >= study_start." in str(context.exception)
        )

        with self.assertRaises(PermissionError) as context:
            loader = BaseFeatureDriver(**self.kwargs)
            loader.age_reference_date = pytz.datetime.date(2006, 1, 1)
        self.assertTrue(
            "age_reference_date should not be updated after loader initialisation."
            in str(context.exception)
        )

    def test_properties_age_groups(self):
        loader = BaseFeatureDriver(**self.kwargs)
        self.assertListEqual(self.sorted_age_groups, loader.age_groups)

        with self.assertRaises(ValueError) as context:
            loader.age_groups = self.age_groups
        self.assertTrue("age_groups bounds should be sorted." in str(context.exception))

    def test_properties_base_population(self):
        loader = BaseFeatureDriver(**self.kwargs)
        loader.run_checks = True

        self.assertTrue(
            data_frame_equality(
                self.base_population.subjects, loader.base_population.subjects
            )
        )
        self.assertTrue(loader.base_population.events is None)

        with patch.object(
            BaseFeatureDriver,
            "_find_subjects_with_age_inconsistent_w_age_groups",
            return_value=self.mock_cohort,
        ) as mock_find_subjects:
            with patch.object(
                BaseFeatureDriver,
                "_log_invalid_events_cohort",
                return_value="Ooops, error here!",
            ) as mock_log_invalid:
                loader = BaseFeatureDriver(**self.kwargs)
                loader.run_checks = True
                with self.assertRaises(ValueError) as context:
                    loader.base_population = self.mock_cohort
                mock_find_subjects.assert_called_once_with(self.mock_cohort)
                mock_log_invalid.assert_called_once_with(
                    self.mock_cohort, log_invalid_subjects=True
                )
                self.assertTrue("Ooops, error here!" == str(context.exception))

    def test_properties_followups(self):
        loader = BaseFeatureDriver(**self.kwargs)
        loader.run_checks = True
        self.assertTrue(
            data_frame_equality(self.followups.subjects, loader.followups.subjects)
        )
        self.assertTrue(
            data_frame_equality(self.followups.events, loader.followups.events)
        )

        with patch.object(
            BaseFeatureDriver,
            "_log_invalid_events_cohort",
            return_value="Ooops, error here!",
        ) as mock_log_invalid:
            with patch.object(
                BaseFeatureDriver,
                "_find_events_not_in_study_dates",
                return_value=self.mock_cohort,
            ) as mock_find_events_study_dates:
                loader = BaseFeatureDriver(**self.kwargs)
                loader.run_checks = True
                with self.assertRaises(ValueError) as context:
                    loader.followups = self.mock_cohort
                mock_find_events_study_dates.assert_called_once_with(self.mock_cohort)
                mock_log_invalid.assert_called_once_with(
                    self.mock_cohort, log_invalid_events=True
                )
                self.assertTrue("Ooops, error here!" == str(context.exception))

        with patch.object(
            BaseFeatureDriver,
            "_log_invalid_events_cohort",
            return_value="Ooops, error here!",
        ) as mock_log_invalid:
            with patch.object(
                BaseFeatureDriver,
                "_find_events_not_in_study_dates",
                return_value=self.mock_empty_cohort,
            ) as mock_did_not_find_events_study_dates:
                with patch.object(
                    BaseFeatureDriver,
                    "_find_inconsistent_start_end_ordering",
                    return_value=self.mock_cohort,
                ) as mock_find_inconsistent_ordering:
                    loader = BaseFeatureDriver(**self.kwargs)
                    loader.run_checks = True
                    with self.assertRaises(ValueError) as context:
                        loader.followups = self.mock_cohort
                    mock_did_not_find_events_study_dates.assert_called_once_with(
                        self.mock_cohort
                    )
                    mock_find_inconsistent_ordering.assert_called_once_with(
                        self.mock_cohort
                    )
                    mock_log_invalid.assert_called_once_with(
                        self.mock_cohort, log_invalid_events=True
                    )
                    self.assertTrue("Ooops, error here!" == str(context.exception))

    def test_properties_is_using_longitudinal_age_groups(self):
        loader = BaseFeatureDriver(**self.kwargs)
        self.assertFalse(loader.is_using_longitudinal_age_groups)

        with self.assertRaises(PermissionError) as context:
            loader.is_using_longitudinal_age_groups = True
        self.assertTrue(
            "is_using_longitudinal_age_groups should not be set manually."
            in str(context.exception)
        )
