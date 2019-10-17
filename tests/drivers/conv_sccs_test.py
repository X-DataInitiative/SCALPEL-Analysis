# License: BSD 3 clause

import pandas as pd
import pytz
import numpy as np
from scipy.sparse import csr_matrix

from scalpel.drivers import ConvSccsFeatureDriver
from scalpel.core.cohort import Cohort
from tests.core.pyspark_tests import PySparkTest
from scalpel.core.util import data_frame_equality
from unittest.mock import patch, PropertyMock, MagicMock
import pyspark.sql.functions as sf
from pyspark.sql import Column
from copy import copy


class TestConvSccsFeatureDriver(PySparkTest):
    def setUp(self):
        super().setUp()
        self.study_start = pytz.datetime.datetime(2010, 2, 5, tzinfo=pytz.UTC)
        self.study_end = pytz.datetime.datetime(2013, 10, 12, tzinfo=pytz.UTC)
        self.age_reference_date = pytz.datetime.datetime(2011, 9, 21, tzinfo=pytz.UTC)
        self.age_groups = [55, 65, 60, 75, 70, 80, 85]
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
                pytz.datetime.datetime(2010, 6, 5, tzinfo=pytz.UTC),
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

        self.exposure_events = {
            "patientID": ["0", "3", "4", "2"],  # uuid
            "start": [
                pytz.datetime.datetime(2010, 6, 7, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2011, 3, 28, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2011, 7, 3, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2010, 11, 22, tzinfo=pytz.UTC),
            ],
            "end": [
                None,
                None,
                None,
                pytz.datetime.datetime(2011, 11, 22, tzinfo=pytz.UTC),
            ],
            "value": ["foo"] * 4,
            "category": ["exposure"] * 4,
            "groupID": [0] * 4,
            "weight": [1] * 4,
        }

        self.outcome_events = {
            "patientID": ["0", "3", "4", "2"],  # uuid
            "start": [
                pytz.datetime.datetime(2010, 6, 8, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2011, 3, 29, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2011, 7, 4, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2010, 11, 23, tzinfo=pytz.UTC),
            ],
            "end": [
                None,
                None,
                None,
                pytz.datetime.datetime(2010, 11, 24, tzinfo=pytz.UTC),
            ],
            "value": ["bar"] * 4,
            "category": ["outcome"] * 4,
            "groupID": [0] * 4,
            "weight": [1] * 4,
        }

        patients_df, _ = self.create_spark_df(self.patients)
        fup_events_df, _ = self.create_spark_df(self.followup_events)
        exp_events_df, _ = self.create_spark_df(self.exposure_events)
        out_events_df, _ = self.create_spark_df(self.outcome_events)

        self.base_population = Cohort(
            "base_population", "base_population", patients_df, None
        )

        self.followups = Cohort(
            "followups",
            "followups",
            fup_events_df.select("patientID").distinct(),
            fup_events_df,
        )

        self.exposures = Cohort(
            "exposures",
            "exposures",
            exp_events_df.select("patientID").distinct(),
            exp_events_df,
        )

        self.outcomes = Cohort(
            "outcomes",
            "outcomes",
            out_events_df.select("patientID").distinct(),
            out_events_df,
        )

        self.bucket_size = 30

        self.kwargs = {
            "base_population": self.base_population,
            "followups": self.followups,
            "exposures": self.exposures,
            "outcomes": self.outcomes,
            "bucket_size": self.bucket_size,
            "study_start": self.study_start,
            "study_end": self.study_end,
            "age_reference_date": self.age_reference_date,
            "age_groups": self.sorted_age_groups,
            "bucket_rounding": "ceil",
            "run_checks": False,
        }

    def test_load(self):
        mock_features = PropertyMock(return_value=[1, 2, 3])
        mock_labels = PropertyMock(return_value=[1, 2, 3])
        mock_censoring = PropertyMock(return_value=[1, 2, 3])
        with patch.object(
            ConvSccsFeatureDriver, "features", new_callable=mock_features
        ):
            with patch.object(
                ConvSccsFeatureDriver, "labels", new_callable=mock_labels
            ):
                with patch.object(
                    ConvSccsFeatureDriver, "censoring", new_callable=mock_censoring
                ):
                    loader_ = ConvSccsFeatureDriver(**self.kwargs)
                    loader_.load()
                    mock_features.assert_called_with()
                    mock_labels.assert_called_once_with()
                    mock_censoring.assert_called_once_with()

        mock_missing_labels = PropertyMock(return_value=[1, 2])
        mock_missing_censoring = PropertyMock(return_value=[1, 2])
        with patch.object(
            ConvSccsFeatureDriver, "features", new_callable=mock_features
        ):
            with patch.object(
                ConvSccsFeatureDriver, "labels", new_callable=mock_missing_labels
            ):
                with patch.object(
                    ConvSccsFeatureDriver, "censoring", new_callable=mock_censoring
                ):
                    loader_ = ConvSccsFeatureDriver(**self.kwargs)
                    with self.assertRaises(AssertionError) as context:
                        loader_.load()
                    self.assertTrue(
                        "Number of feature matrices does not match "
                        "number of label matrices. You might want to"
                        " investigate this" in str(context.exception)
                    )

        with patch.object(
            ConvSccsFeatureDriver, "features", new_callable=mock_features
        ):
            with patch.object(
                ConvSccsFeatureDriver, "labels", new_callable=mock_labels
            ):
                with patch.object(
                    ConvSccsFeatureDriver,
                    "censoring",
                    new_callable=mock_missing_censoring,
                ):
                    loader_ = ConvSccsFeatureDriver(**self.kwargs)
                    with self.assertRaises(AssertionError) as context:
                        loader_.load()
                    self.assertTrue(
                        "Number of feature matrices does not match "
                        "number of censoring values. You might want to"
                        " investigate this" in str(context.exception)
                    )

    def test_properties_bucket_rounding(self):
        loader = ConvSccsFeatureDriver(**self.kwargs)

        self.assertEqual(loader.bucket_rounding, self.kwargs["bucket_rounding"])

        with self.assertRaises(ValueError) as context:
            loader.bucket_rounding = "foo"
        self.assertTrue(
            "bucket_rounding should be equal to either 'ceil' or 'floor'"
            in str(context.exception)
        )

    def test_properties_exposures(self):
        loader = ConvSccsFeatureDriver(**self.kwargs)
        exposures_ = loader.exposures
        self.assertTrue(
            data_frame_equality(exposures_.subjects, self.exposures.subjects)
        )
        self.assertTrue(data_frame_equality(exposures_.events, self.exposures.events))

        mock_dataframe = MagicMock()
        mock_dataframe.take = lambda x: True
        mock_cohort = MagicMock()
        mock_cohort.subjects = mock_dataframe
        mock_cohort.events = mock_dataframe
        with patch.object(
            ConvSccsFeatureDriver,
            "_find_events_not_in_followup_bounds",
            return_value=mock_cohort,
        ) as mock_find_events:
            with patch.object(
                ConvSccsFeatureDriver,
                "_log_invalid_events_cohort",
                return_value="Ooops, error here!",
            ) as mock_log_invalid:
                loader_ = ConvSccsFeatureDriver(**self.kwargs)
                loader_.run_checks = True
                with self.assertRaises(ValueError) as context:
                    loader_.exposures = mock_cohort
                mock_find_events.assert_called_once_with(mock_cohort)
                mock_log_invalid.assert_called_once_with(
                    mock_cohort, log_invalid_events=True
                )
                self.assertTrue("Ooops, error here!" == str(context.exception))

    def test_properties_outcomes(self):
        loader = ConvSccsFeatureDriver(**self.kwargs)
        outcomes_ = loader.outcomes
        self.assertTrue(data_frame_equality(outcomes_.subjects, self.outcomes.subjects))
        self.assertTrue(data_frame_equality(outcomes_.events, self.outcomes.events))

        loader_ = ConvSccsFeatureDriver(**self.kwargs)
        loader_.run_checks = True

        bad_outcomes_df, _ = self.create_spark_df(
            {
                "patientID": ["0", "4"],  # uuid
                "start": [
                    pytz.datetime.datetime(2010, 6, 8, tzinfo=pytz.UTC),
                    pytz.datetime.datetime(2011, 3, 29, tzinfo=pytz.UTC),
                ],
                "end": [None, pytz.datetime.datetime(2010, 11, 24, tzinfo=pytz.UTC)],
                "value": ["bar", "baz"],
                "category": ["outcome"] * 2,
                "groupID": [0] * 2,
                "weight": [1] * 2,
            }
        )
        bad_outcomes_cohort = Cohort(
            "", "", bad_outcomes_df.select("patientID").distinct(), bad_outcomes_df
        )

        with self.assertRaises(AssertionError) as context:
            loader_.outcomes = bad_outcomes_cohort

        self.assertTrue(
            "There are more than one type of outcomes, check the 'value' field of "
            "outcomes cohort events." in str(context.exception)
        )

        mock_dataframe = MagicMock()
        mock_dataframe.take = lambda x: True
        mock_cohort = MagicMock()
        mock_cohort.subjects = mock_dataframe
        mock_cohort.events = mock_dataframe
        mock_empty_df = MagicMock()
        mock_empty_df.take = lambda x: []
        mock_empty_cohort = MagicMock
        mock_empty_cohort.subjects = mock_empty_df
        mock_empty_cohort.events = mock_empty_df
        with patch.object(
            ConvSccsFeatureDriver,
            "_log_invalid_events_cohort",
            return_value="Ooops, error here!",
        ) as mock_log_invalid:
            with patch.object(
                ConvSccsFeatureDriver,
                "_find_events_not_in_followup_bounds",
                return_value=mock_cohort,
            ) as mock_find_events_outcome_bounds:
                loader = ConvSccsFeatureDriver(**self.kwargs)
                loader.run_checks = True
                with self.assertRaises(ValueError) as context:
                    loader.outcomes = self.outcomes
                mock_find_events_outcome_bounds.assert_called_once_with(self.outcomes)
                mock_log_invalid.assert_called_once_with(
                    mock_cohort, log_invalid_events=True
                )
                self.assertTrue("Ooops, error here!" == str(context.exception))
        with patch.object(
            ConvSccsFeatureDriver,
            "_log_invalid_events_cohort",
            return_value="Ooops, error here!",
        ) as mock_log_invalid:
            with patch.object(
                ConvSccsFeatureDriver,
                "_find_events_not_in_followup_bounds",
                return_value=mock_empty_cohort,
            ) as mock_did_not_find_outcome_bounds:
                with patch.object(
                    ConvSccsFeatureDriver,
                    "_find_subjects_with_many_outcomes",
                    return_value=mock_cohort,
                ) as mock_find_many_outcomes:
                    loader = ConvSccsFeatureDriver(**self.kwargs)
                    loader.run_checks = True
                    with self.assertRaises(ValueError) as context:
                        loader.outcomes = self.outcomes
                    mock_did_not_find_outcome_bounds.assert_called_once_with(
                        self.outcomes
                    )
                    mock_find_many_outcomes.assert_called_once_with(self.outcomes)
                    mock_log_invalid.assert_called_once_with(
                        mock_cohort, log_invalid_subjects=True
                    )
                    self.assertTrue("Ooops, error here!" == str(context.exception))

    def test_properties_features(self):
        with patch.object(
            ConvSccsFeatureDriver, "_load_features", return_value="some features"
        ) as mocked_method:
            loader_ = ConvSccsFeatureDriver(**self.kwargs)
            result = loader_.features
            mocked_method.assert_called_once_with()
            self.assertEqual(result, "some features")

            with self.assertRaises(PermissionError) as context:
                loader_.features = "some value"
            self.assertTrue(
                "features should not be set manually,"
                "they are computed from initial cohorts." in str(context.exception)
            )

    def test_properties_labels(self):
        with patch.object(
            ConvSccsFeatureDriver, "_load_labels", return_value="some labels"
        ) as mocked_method:
            loader_ = ConvSccsFeatureDriver(**self.kwargs)
            self.assertEqual(loader_.labels, "some labels")
            mocked_method.assert_called_once_with()

            with self.assertRaises(PermissionError) as context:
                loader_.labels = "some value"

            self.assertTrue(
                "labels should not be set manually,"
                "they are computed from initial cohorts." in str(context.exception)
            )

    def test_properties_censoring(self):
        with patch.object(
            ConvSccsFeatureDriver, "_load_censoring", return_value="some censoring"
        ) as mocked_method:
            loader_ = ConvSccsFeatureDriver(**self.kwargs)
            self.assertEqual(loader_.censoring, "some censoring")
            mocked_method.assert_called_once_with()

            with self.assertRaises(PermissionError) as context:
                loader_.censoring = "some value"
            self.assertTrue(
                "censoring should not be set manually,"
                "it is computed from initial cohorts." in str(context.exception)
            )

    def test_properties_mapping(self):
        loader = ConvSccsFeatureDriver(**self.kwargs)
        loader._feature_mapping = ["feature 1", "feature 2"]
        loader._outcome_mapping = ["outcome 1"]
        self.assertListEqual(loader.mappings[0], ["feature 1", "feature 2"])
        self.assertListEqual(loader.mappings[1], ["outcome 1"])

        with self.assertRaises(PermissionError) as context:
            loader.mappings = "some value"
        self.assertTrue(
            "mappings should not be set manually,"
            "they are computed from initial cohorts." in str(context.exception)
        )

    def test_properties_final_cohort(self):
        loader = ConvSccsFeatureDriver(**self.kwargs)
        with self.assertRaises(PermissionError) as context:
            loader.final_cohort = "some value"
        self.assertTrue(
            "final_cohort should not be set manually,"
            "it is computed from initial cohorts." in str(context.exception)
        )

        with self.assertRaises(AssertionError) as context:
            patients_wo_events, _ = self.create_spark_df(self.patients)
            patients_wo_events = patients_wo_events.select(
                (sf.col("patientID") + 1000).alias("patientID"),
                sf.col("gender"),
                sf.col("birthDate"),
                sf.col("deathDate"),
            )
            loader.base_population = Cohort(
                "base_population", "base_population", patients_wo_events, None
            )
            loader.final_cohort.subjects.count()
        self.assertTrue(
            "Final cohort is empty, please check that "
            "the intersection of the provided cohorts "
            "is nonempty" in str(context.exception)
        )

    def test_properties_exposures_split_column(self):
        loader = ConvSccsFeatureDriver(**self.kwargs)
        self.assertEqual(loader.exposures_split_column, "value")
        with self.assertRaises(ValueError) as context:
            loader.exposures_split_column = "foo"
        self.assertTrue(
            "exposures_split_column should be either 'category', 'groupID', or 'value'"
            in str(context.exception)
        )

    def test_properties_outcomes_split_column(self):
        loader = ConvSccsFeatureDriver(**self.kwargs)
        self.assertEqual(loader.outcomes_split_column, "value")
        with self.assertRaises(ValueError) as context:
            loader.outcomes_split_column = "foo"
        self.assertTrue(
            "outcomes_split_column should be either 'category', 'groupID', or 'value'"
            in str(context.exception)
        )

    def test_get_bucketized_events(self):
        kwargs = copy(self.kwargs)
        kwargs["bucket_size"] = 365
        loader = ConvSccsFeatureDriver(**kwargs)
        features_, n_cols, mapping = loader._get_bucketized_events(
            loader.exposures, "value"
        )

        expected = np.array([[0, 0, 0], [3, 1, 0], [4, 1, 0], [2, 0, 0]]).astype("int")

        self.assertEqual(n_cols, 1)
        self.assertListEqual(mapping, ["foo"])
        np.testing.assert_array_equal(
            features_.toPandas().values.astype("int"), expected
        )

    def test_compute_longitudinal_age_groups(self):
        kwargs = copy(self.kwargs)
        kwargs["bucket_size"] = 365
        loader = ConvSccsFeatureDriver(**kwargs)

        with self.assertRaises(AssertionError) as context:
            bad_cohort = Cohort(
                "base_population",
                "base_population",
                self.base_population.subjects.select("patientID"),
                None,
            )
            _ = loader._compute_longitudinal_age_groups(bad_cohort, col_offset=int(2))
        self.assertTrue(
            "Cohort subjects should have gender and birthdate information"
            in str(context.exception)
        )

        features, mapping = loader._compute_longitudinal_age_groups(
            self.base_population, col_offset=int(2)
        )
        expected_mapping = [
            "[55.0, 60.0)",
            "[60.0, 65.0)",
            "[65.0, 70.0)",
            "[70.0, 75.0)",
            "[75.0, 80.0)",
            "[80.0, 85.0)",
        ]
        expected_data = np.array(
            [
                [3, 1, 6],
                [0, 0, 6],
                [0, 1, 6],
                [0, 2, 6],
                [0, 3, 7],
                [4, 1, 5],
                [4, 2, 6],
                [2, 0, 4],
                [2, 1, 5],
                [2, 2, 5],
                [2, 3, 5],
            ]
        ).astype("int")
        self.assertListEqual(mapping, expected_mapping)
        np.testing.assert_array_equal(
            features.toPandas().values.astype("int"), expected_data
        )

    def test_load_features(self):
        kwargs = copy(self.kwargs)
        kwargs["bucket_size"] = 365
        loader = ConvSccsFeatureDriver(**kwargs)
        result = loader._load_features()
        result = [res.toarray() for res in result]
        expected = [
            np.array(
                [
                    [1, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                ],
                dtype=np.int64,
            ),
            np.array(
                [
                    [1, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                ],
                dtype=np.int64,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int64,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int64,
            ),
        ]
        for i, exp in enumerate(expected):
            np.testing.assert_array_equal(result[i], exp)

    def test_load_labels(self):
        kwargs = copy(self.kwargs)
        kwargs["bucket_size"] = 365
        loader = ConvSccsFeatureDriver(**kwargs)
        result = loader._load_labels()
        expected = [
            np.array([[1], [0], [0], [0]], dtype=np.int64),
            np.array([[1], [0], [0], [0]], dtype=np.int64),
            np.array([[0], [1], [0], [0]], dtype=np.int64),
            np.array([[0], [1], [0], [0]], dtype=np.int64),
        ]
        for i, exp in enumerate(expected):
            np.testing.assert_array_equal(result[i], exp)

    def test_load_censoring(self):
        kwargs = copy(self.kwargs)
        kwargs["bucket_size"] = 365
        loader = ConvSccsFeatureDriver(**kwargs)
        result = loader._load_censoring()
        expected = np.array([[3], [3], [1], [2]])
        np.testing.assert_array_equal(result, expected)

    def test_discretize_time(self):
        loader = ConvSccsFeatureDriver(**self.kwargs)
        result = loader._discretize_start_end(self.exposures.events).toPandas()
        np.testing.assert_array_equal(
            result.startBucket.values, np.array([4, 13, 17, 9])
        )
        np.testing.assert_array_equal(
            result.endBucket.values, np.array([np.nan, np.nan, np.nan, 21])
        )

        kwargs = copy(self.kwargs)
        kwargs["bucket_rounding"] = "floor"
        kwargs["study_end"] = pytz.datetime.datetime(2011, 7, 5, tzinfo=pytz.UTC)
        loader_floor = ConvSccsFeatureDriver(**kwargs)
        some_events = {
            "patientID": ["0", "3", "4", "2"],  # uuid
            "start": [
                pytz.datetime.datetime(2010, 6, 8, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2011, 3, 29, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2011, 7, 4, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2010, 11, 23, tzinfo=pytz.UTC),
            ],
            "end": [
                None,
                None,
                None,
                pytz.datetime.datetime(2010, 11, 24, tzinfo=pytz.UTC),
            ],
        }
        data, _ = self.create_spark_df(some_events)
        result = loader_floor._discretize_start_end(data).toPandas()
        np.testing.assert_array_equal(
            result.startBucket.values, np.array([4, 13, 16, 9])
        )
        np.testing.assert_array_equal(
            result.endBucket.values, np.array([np.nan, np.nan, np.nan, 9])
        )

    def test_discretize_start_end(self):
        with patch.object(
            ConvSccsFeatureDriver,
            "_discretize_time",
            return_value=sf.lit("mocked_value"),
        ) as mocked_method:
            loader_ = ConvSccsFeatureDriver(**self.kwargs)
            loader_._discretize_start_end(self.exposures.events)
            colnames = ["start", "end"]
            # Check that '_distcretize_time' has been called on the right cols
            for i, call in enumerate(mocked_method.call_args_list):
                expected_name = colnames[i]
                result = call[0][0]
                self.assertEqual(expected_name, result._jc.toString())
                self.assertTrue(isinstance(result, Column))

    def test_get_csr_matrices(self):
        loader = ConvSccsFeatureDriver(**self.kwargs)
        no_row_index_df, _ = self.create_spark_df({"blah": [1, 2, 3]})
        no_col_index_df, _ = self.create_spark_df(
            {"rowIndex": [1, 2, 3], "blah": [1, 2, 3]}
        )
        csr_shape = (2, 5)

        with self.assertRaises(ValueError) as context:
            loader._get_csr_matrices(no_row_index_df, csr_shape)
        self.assertTrue(
            "rowIndex should be in events columns." in str(context.exception)
        )

        with self.assertRaises(ValueError) as context:
            loader._get_csr_matrices(no_col_index_df, csr_shape)
        self.assertTrue(
            "colIndex should be in events columns." in str(context.exception)
        )

        valid_df, _ = self.create_spark_df(
            {
                "patientID": ["Alice", "Bob", "Alice", "Alice", "Bob"],
                "rowIndex": [0, 0, 0, 1, 1],
                "colIndex": [0, 1, 2, 3, 4],
            }
        )
        result = loader._get_csr_matrices(valid_df, csr_shape)
        result = [res.toarray() for res in result]
        expected = [
            np.array([[1, 0, 1, 0, 0], [0, 0, 0, 1, 0]], dtype=np.int64),
            np.array([[0, 1, 0, 0, 0], [0, 0, 0, 0, 1]], dtype=np.int64),
        ]
        for i, exp in enumerate(expected):
            np.testing.assert_array_equal(result[i], exp)

    def test_create_csr_matrix(self):
        df_row = pd.Series({"rowIndexes": [0, 1, 2, 2], "colIndexes": [0, 1, 0, 1]})
        loader = ConvSccsFeatureDriver(**self.kwargs)
        m = loader._create_csr_matrix(df_row, csr_matrix_shape=(5, 2))
        expected = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [0, 0]])
        self.assertTrue(np.array_equal(m.toarray(), expected))
        self.assertTrue(m.shape == (5, 2))
        self.assertTrue(type(m) == csr_matrix)

    def test_find_subjects_with_many_outcomes(self):
        invalid_events = {
            "patientID": ["0", "0", "1", "1", "2"],  # uuid
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

        loader = ConvSccsFeatureDriver(**self.kwargs)
        invalid = loader._find_subjects_with_many_outcomes(invalid_cohort)

        self.assertEqual(
            invalid.name, "some_cohort_inconsistent_w_single_outcome_constraint"
        )
        self.assertEqual(
            invalid.describe(),
            "Events are some_cohort_inconsistent_w_single_outcome_constraint. Events "
            "contain only events showing there are more than one outcome per patient.",
        )
        self.assertListEqual(
            sorted(invalid.subjects.toPandas().values.ravel().tolist()),
            sorted(["0", "1"]),
        )
        self.assertListEqual(
            sorted(invalid.events.toPandas().value.values.ravel().tolist()),
            sorted([0, 1, 2, 3]),
        )
