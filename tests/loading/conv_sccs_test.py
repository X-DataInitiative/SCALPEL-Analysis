import pandas as pd
import pytz
import numpy as np
from scipy.sparse import csr_matrix

from src.exploration.loaders import ConvSccsLoader
from src.exploration.core.cohort import Cohort
from tests.exploration.core.pyspark_tests import PySparkTest
from src.exploration.core.util import data_frame_equality
from unittest.mock import patch, PropertyMock
import pyspark.sql.functions as sf
from pyspark.sql import Column
from copy import copy
from pandas.util.testing import assert_frame_equal


class TestConvSccsLoader(PySparkTest):
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
            "stop": [
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
        out_events_df, _ = self.create_spark_df(self.exposure_events)

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
        with patch.object(ConvSccsLoader, "features", new_callable=mock_features):
            with patch.object(ConvSccsLoader, "labels", new_callable=mock_labels):
                with patch.object(
                    ConvSccsLoader, "censoring", new_callable=mock_censoring
                ):
                    loader_ = ConvSccsLoader(**self.kwargs)
                    loader_.load()
                    mock_features.assert_called_with()
                    mock_labels.assert_called_once_with()
                    mock_censoring.assert_called_once_with()

        mock_missing_labels = PropertyMock(return_value=[1, 2])
        mock_missing_censoring = PropertyMock(return_value=[1, 2])
        with patch.object(ConvSccsLoader, "features", new_callable=mock_features):
            with patch.object(
                ConvSccsLoader, "labels", new_callable=mock_missing_labels
            ):
                with patch.object(
                    ConvSccsLoader, "censoring", new_callable=mock_censoring
                ):
                    loader_ = ConvSccsLoader(**self.kwargs)
                    with self.assertRaises(AssertionError) as context:
                        loader_.load()
                    self.assertTrue(
                        "Number of feature matrices does not match "
                        "number of label matrices. You might want to"
                        " investigate this" in str(context.exception)
                    )

        with patch.object(ConvSccsLoader, "features", new_callable=mock_features):
            with patch.object(ConvSccsLoader, "labels", new_callable=mock_labels):
                with patch.object(
                    ConvSccsLoader, "censoring", new_callable=mock_missing_censoring
                ):
                    loader_ = ConvSccsLoader(**self.kwargs)
                    with self.assertRaises(AssertionError) as context:
                        loader_.load()
                    self.assertTrue(
                        "Number of feature matrices does not match "
                        "number of censoring values. You might want to"
                        " investigate this" in str(context.exception)
                    )

    def test_properties(self):
        loader = ConvSccsLoader(**self.kwargs)

        # bucket_rounding
        self.assertEqual(loader.bucket_rounding, self.kwargs["bucket_rounding"])

        with self.assertRaises(ValueError) as context:
            loader.bucket_rounding = "foo"

        self.assertTrue(
            "bucket_rounding should be equal to either 'ceil',"
            "'floor', or 'trunc'" in str(context.exception)
        )

        with self.assertRaises(NotImplementedError) as context:
            loader.bucket_rounding = "trunc"

        self.assertTrue("Not implemented yet :( " in str(context.exception))

        # exposures
        exposures_ = loader.exposures
        self.assertTrue(
            data_frame_equality(exposures_.subjects, self.exposures.subjects)
        )
        self.assertTrue(data_frame_equality(exposures_.events, self.exposures.events))

        with patch.object(
            ConvSccsLoader, "_check_event_dates_consistency_w_followup_bounds"
        ) as mocked_method:
            loader_ = ConvSccsLoader(**self.kwargs)
            loader_.run_checks = True
            loader_.exposures = "foo"  # Just use some other cohort
            self.assertEqual(loader_.exposures, "foo")
            mocked_method.assert_called_once_with("foo")

        # outcomes
        outcomes_ = loader.outcomes
        self.assertTrue(data_frame_equality(outcomes_.subjects, self.outcomes.subjects))
        self.assertTrue(
            data_frame_equality(
                outcomes_.events,
                self.outcomes.events.groupby(
                    "patientID", "category", "groupID", "value", "weight"
                )
                .agg(sf.min("start").alias("start"), sf.min("end").alias("end"))
                .select(
                    "patientID",
                    "start",
                    "end",
                    "category",
                    "groupID",
                    "value",
                    "weight",
                ),
            )
        )

        with patch.object(
            ConvSccsLoader, "_check_event_dates_consistency_w_followup_bounds"
        ) as mocked_method:
            loader_ = ConvSccsLoader(**self.kwargs)
            loader_.run_checks = True
            loader_.outcomes = self.outcomes
            mocked_method.assert_called_once_with(self.outcomes)

            new_outcomes = copy(self.outcomes)
            new_outcomes.events = (
                new_outcomes.events.groupby(
                    "patientID", "category", "groupID", "value", "weight"
                )
                .agg(sf.min("start").alias("start"), sf.min("end").alias("end"))
                .select(
                    "patientID",
                    "start",
                    "end",
                    "category",
                    "groupID",
                    "value",
                    "weight",
                )
            )

            self.assertTrue(
                data_frame_equality(loader_.outcomes.subjects, new_outcomes.subjects)
            )
            self.assertTrue(
                data_frame_equality(loader_.outcomes.events, new_outcomes.events)
            )

        # features
        with patch.object(
            ConvSccsLoader, "_load_features", return_value="some features"
        ) as mocked_method:
            loader_ = ConvSccsLoader(**self.kwargs)
            self.assertEqual(loader_.features, "some features")
            mocked_method.assert_called_once()

            with self.assertRaises(NotImplementedError) as context:
                loader_.features = "some value"

            self.assertTrue(
                "features should not be set manually,"
                "they are computed from initial cohorts." in str(context.exception)
            )

        # labels
        with patch.object(
            ConvSccsLoader, "_load_labels", return_value="some labels"
        ) as mocked_method:
            loader_ = ConvSccsLoader(**self.kwargs)
            self.assertEqual(loader_.labels, "some labels")
            mocked_method.assert_called_once()

            with self.assertRaises(NotImplementedError) as context:
                loader_.labels = "some value"

            self.assertTrue(
                "labels should not be set manually,"
                "they are computed from initial cohorts." in str(context.exception)
            )

        # censoring
        with patch.object(
            ConvSccsLoader, "_load_censoring", return_value="some censoring"
        ) as mocked_method:
            loader_ = ConvSccsLoader(**self.kwargs)
            self.assertEqual(loader_.censoring, "some censoring")
            mocked_method.assert_called_once()

            with self.assertRaises(NotImplementedError) as context:
                loader_.censoring = "some value"

            self.assertTrue(
                "censoring should not be set manually,"
                "it is computed from initial cohorts." in str(context.exception)
            )

        # final_cohort
        with self.assertRaises(NotImplementedError) as context:
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

    def test_compute_longitudinal_age_groups(self):
        kwargs = copy(self.kwargs)
        kwargs["bucket_size"] = 365
        loader = ConvSccsLoader(**kwargs)

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
        expected_data = [
            [0, 0, 6],
            [0, 1, 6],
            [0, 2, 6],
            [0, 3, 7],
            [1, 0, 3],
            [1, 1, 3],
            [1, 2, 3],
            [1, 3, 3],
            [2, 0, 4],
            [2, 1, 5],
            [2, 2, 5],
            [2, 3, 5],
            [3, 0, 6],
            [3, 1, 6],
            [3, 2, 6],
            [3, 3, 7],
            [4, 0, 5],
            [4, 1, 5],
            [4, 2, 6],
            [4, 3, 6],
        ]

        expected_df = pd.DataFrame(
            expected_data, columns=["patientID", "rowIndex", "colIndex"], dtype="int"
        )
        self.assertListEqual(mapping, expected_mapping)
        assert_frame_equal(features.toPandas().astype("int"), expected_df)

    def test_load_features(self):
        kwargs = copy(self.kwargs)
        kwargs["bucket_size"] = 365
        loader = ConvSccsLoader(**kwargs)
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
        loader = ConvSccsLoader(**kwargs)
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
        loader = ConvSccsLoader(**kwargs)
        result = loader._load_censoring()
        expected = np.array([[3], [3], [1], [2]])
        np.testing.assert_array_equal(result, expected)

    def test_discretize_time(self):
        loader = ConvSccsLoader(**self.kwargs)
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
        loader_floor = ConvSccsLoader(**kwargs)
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
            ConvSccsLoader, "_discretize_time", return_value=sf.lit("mocked_value")
        ) as mocked_method:
            loader_ = ConvSccsLoader(**self.kwargs)
            loader_._discretize_start_end(self.exposures.events)
            colnames = ["start", "end"]
            # Check that '_distcretize_time' has been called on the right cols
            for i, call in enumerate(mocked_method.call_args_list):
                expected_name = colnames[i]
                result = call[0][0]
                self.assertEqual(expected_name, result._jc.toString())
                self.assertTrue(isinstance(result, Column))

    def test_get_csr_matrices(self):
        loader = ConvSccsLoader(**self.kwargs)
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
        loader = ConvSccsLoader(**self.kwargs)
        m = loader._create_csr_matrix(df_row, csr_matrix_shape=(5, 2))
        expected = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [0, 0]])
        self.assertTrue(np.array_equal(m.toarray(), expected))
        self.assertTrue(m.shape == (5, 2))
        self.assertTrue(type(m) == csr_matrix)
