import pytz
from src.exploration.core.util import (
    data_frame_equality,
    fold_right,
    rename_df_columns,
    index_string_column,
)
from .pyspark_tests import PySparkTest


class TestUtil(PySparkTest):
    def test_fold_right(self):
        input_data = [1, 2, 3]

        def add(a, b):
            return a + b

        result = fold_right(add, input_data)
        self.assertEqual(result, 6)

    def test_data_frame_equality(self):
        df, _ = self.create_spark_df({"patientID": [1, 2, 3]})

        self.assertTrue(data_frame_equality(df, df))
        self.assertFalse(data_frame_equality(df, None))

    def test_rename_columns(self):
        some_events = {
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

        some_events_df, _ = self.create_spark_df(some_events)
        cols = some_events_df.columns

        res = rename_df_columns(some_events_df, prefix="pre_")
        expected = ["pre_" + c if c != "patientID" else c for c in cols]
        self.assertListEqual(sorted(res.columns), sorted(expected))

        res = rename_df_columns(some_events_df, suffix="_suf")
        expected = [c + "_suf" if c != "patientID" else c for c in cols]
        self.assertListEqual(sorted(res.columns), sorted(expected))

        expected = "the quick brown".split(" ")
        res = rename_df_columns(some_events_df, new_names=expected)
        expected.insert(0, "patientID")
        self.assertListEqual(sorted(res.columns), sorted(expected))

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

        indexed, n_cat, mapping = index_string_column(data, "value", "valueIndex")
        index = indexed.toPandas().valueIndex.values.tolist()
        self.assertEqual(expected_n_categories, n_cat)
        self.assertListEqual(expected_mapping, mapping)
        self.assertListEqual(expected_index, index)
