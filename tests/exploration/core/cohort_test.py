import pandas as pd

from src.exploration.core.cohort import Cohort
from .pyspark_tests import PySparkTest


class TestCohort(PySparkTest):

    def test_union(self):
        patients_pd = pd.DataFrame({"patientID": [1, 2, 3]})
        events_pd = pd.DataFrame({"patientID": [1, 2, 3], "value": ["DP", "DAS", "DR"]})

        patients = self.spark.createDataFrame(patients_pd)

        events = self.spark.createDataFrame(events_pd)
        cohort1 = Cohort("liberal_fractures", "liberal_fractures",
                         patients, events)

        cohort2 = Cohort("hospit_fractures", "hospit_fractures",
                         patients, events)

        result = cohort1.union(cohort2)

        expected_patients = self.spark.createDataFrame(pd.concat([patients_pd, patients_pd]))
        expected_events = self.spark.createDataFrame(pd.concat([events_pd, events_pd]))
        expected = Cohort("result", "result", expected_patients, expected_events)
        self.assertEqual(result, expected)

    def test_union_all(self):
        patients_pd = pd.DataFrame({"patientID": [1, 2, 3]})
        events_pd = pd.DataFrame({"patientID": [1, 2, 3], "value": ["DP", "DAS", "DR"]})

        patients = self.spark.createDataFrame(patients_pd)

        events = self.spark.createDataFrame(events_pd)
        cohort1 = Cohort("liberal_fractures", "liberal_fractures",
                         patients, events)

        cohort2 = Cohort("hospit_fractures", "hospit_fractures",
                         patients, events)

        cohort3 = Cohort("fractures", "fractures",
                         patients, events)

        result = Cohort.union_all([cohort1, cohort2, cohort3])

        expected_patients = self.spark.createDataFrame(
            pd.concat([patients_pd]*3))
        expected_events = self.spark.createDataFrame(pd.concat([events_pd]*3))
        expected = Cohort("result", "result", expected_patients, expected_events)
        self.assertEqual(result, expected)

    def test_intersect(self):
        patients_1, patients_pd_1 = self.create_spark_df({"patientID": [1, 2]})
        events_1, events_pd_1 = self.create_spark_df({"patientID": [1, 2], "value": ["DP", "DAS"]})

        patients_2, patients_pd_2 = self.create_spark_df({"patientID": [1]})
        events_2, events_pd_2 = self.create_spark_df({"patientID": [1], "value": ["DP"]})

        cohort1 = Cohort("liberal_fractures", "liberal_fractures",
                         patients_1, events_1)

        cohort2 = Cohort("hospit_fractures", "hospit_fractures",
                         patients_2, events_2)

        result = cohort1.intersection(cohort2)

        expected = cohort2
        self.assertEqual(result, expected)

    def test_intersect_all(self):
        patients_1, patients_pd_1 = self.create_spark_df({"patientID": [1, 2]})
        events_1, events_pd_1 = self.create_spark_df({"patientID": [1, 2], "value": ["DP", "DAS"]})

        patients_2, patients_pd_2 = self.create_spark_df({"patientID": [1]})
        events_2, events_pd_2 = self.create_spark_df({"patientID": [1], "value": ["DP"]})

        patients_3, patients_pd_3 = self.create_spark_df({"patientID": [1, 3]})
        events_3, events_pd_3 = self.create_spark_df({"patientID": [1, 3], "value": ["DP", "DR"]})

        cohort1 = Cohort("liberal_fractures", "liberal_fractures",
                         patients_1, events_1)

        cohort2 = Cohort("hospit_fractures", "hospit_fractures",
                         patients_2, events_2)

        cohort3 = Cohort("imb_fractures", "imb_fractures",
                         patients_3, events_3)

        result = Cohort.intersect_all([cohort1, cohort2, cohort3])

        expected = cohort2
        self.assertEqual(result, expected)
