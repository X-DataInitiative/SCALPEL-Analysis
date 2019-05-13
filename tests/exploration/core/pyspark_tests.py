import logging
import unittest
from typing import Dict

import pandas as pd
from pyspark.sql import SparkSession


class PySparkTest(unittest.TestCase):
    @classmethod
    def suppress_py4j_logging(cls):
        logger = logging.getLogger("py4j")
        logger.setLevel(logging.WARN)

    @classmethod
    def create_testing_pyspark_session(cls):
        return (
            SparkSession.builder.master("local[2]")
            .appName("my-local-testing-pyspark-context")
            .getOrCreate()
        )

    @classmethod
    def setUpClass(cls):
        cls.suppress_py4j_logging()
        cls.spark = cls.create_testing_pyspark_session()
        cls.spark.conf.set("spark.sql.session.timeZone", "UTC")

    def create_spark_df(self, data: Dict):
        # Warning: if you want to be sure of colums ordering, use and OrderedDict
        df = pd.DataFrame(data, columns=data.keys())
        # use of dict.keys() to ensure consistent column ordering
        return self.spark.createDataFrame(df), df
