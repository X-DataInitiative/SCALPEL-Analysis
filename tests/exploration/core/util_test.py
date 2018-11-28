from src.exploration.core.util import data_frame_equality, fold_right
from .pyspark_tests import PySparkTest


class TestUtil(PySparkTest):
    def test_fold_right(self):
        input = [1, 2, 3]

        def add(a, b):
            return a + b

        result = fold_right(add, input)
        self.assertEqual(result, 6)

    def test_data_frame_equality(self):
        df, _ = self.create_spark_df({"patientID": [1, 2, 3]})

        self.assertTrue(data_frame_equality(df, df))
        self.assertFalse(data_frame_equality(df, None))
