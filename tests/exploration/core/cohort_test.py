from datetime import datetime

import pandas as pd
from pyspark.sql.types import IntegerType, StructField, StructType, TimestampType

from src.exploration.core.cohort import Cohort
from src.exploration.core.util import data_frame_equality
from .pyspark_tests import PySparkTest


class TestCohort(PySparkTest):
    def test_union(self):
        patients_pd = pd.DataFrame({"patientID": [1, 2, 3]})
        events_pd = pd.DataFrame({"patientID": [1, 2, 3], "value": ["DP", "DAS", "DR"]})

        patients = self.spark.createDataFrame(patients_pd)

        events = self.spark.createDataFrame(events_pd)
        cohort1 = Cohort("liberal_fractures", "liberal_fractures", patients, events)

        cohort2 = Cohort("hospit_fractures", "hospit_fractures", patients, events)

        result = cohort1.union(cohort2)

        expected_patients = self.spark.createDataFrame(
            pd.concat([patients_pd, patients_pd])
        )
        expected_events = self.spark.createDataFrame(pd.concat([events_pd, events_pd]))
        expected = Cohort("result", "result", expected_patients, expected_events)
        self.assertEqual(result, expected)

    def test_union_all(self):
        patients_pd = pd.DataFrame({"patientID": [1, 2, 3]})
        events_pd = pd.DataFrame({"patientID": [1, 2, 3], "value": ["DP", "DAS", "DR"]})

        patients = self.spark.createDataFrame(patients_pd)

        events = self.spark.createDataFrame(events_pd)
        cohort1 = Cohort("liberal_fractures", "liberal_fractures", patients, events)

        cohort2 = Cohort("hospit_fractures", "hospit_fractures", patients, events)

        cohort3 = Cohort("fractures", "fractures", patients, None)

        result = Cohort.union_all([cohort1, cohort2, cohort3])

        expected_patients = self.spark.createDataFrame(pd.concat([patients_pd] * 3))
        expected = Cohort("result", "result", expected_patients, None)
        self.assertEqual(result, expected)

    def test_intersect(self):
        patients_1, patients_pd_1 = self.create_spark_df({"patientID": [1, 2]})
        events_1, events_pd_1 = self.create_spark_df(
            {"patientID": [1, 2], "value": ["DP", "DAS"]}
        )

        patients_2, patients_pd_2 = self.create_spark_df({"patientID": [1]})
        events_2, events_pd_2 = self.create_spark_df(
            {"patientID": [1], "value": ["DP"]}
        )

        cohort1 = Cohort("liberal_fractures", "liberal_fractures", patients_1, events_1)

        cohort2 = Cohort("hospit_fractures", "hospit_fractures", patients_2, events_2)

        result = cohort1.intersection(cohort2)

        expected = cohort2
        self.assertEqual(result, expected)

    def test_intersect_all(self):
        patients_1, _ = self.create_spark_df({"patientID": [1, 2]})
        events_1, _ = self.create_spark_df(
            {"patientID": [1, 2], "value": ["DP", "DAS"]}
        )

        patients_2, _ = self.create_spark_df({"patientID": [1]})
        events_2, _ = self.create_spark_df({"patientID": [1], "value": ["DP"]})

        patients_3, _ = self.create_spark_df({"patientID": [1, 3]})

        cohort1 = Cohort("liberal_fractures", "liberal_fractures", patients_1, events_1)

        cohort2 = Cohort("hospit_fractures", "hospit_fractures", patients_2, events_2)

        cohort3 = Cohort("imb_fractures", "imb_fractures", patients_3, None)

        result = Cohort.intersect_all([cohort1, cohort2, cohort3])

        expected = Cohort("hospit_fractures", "hospit_fractures", patients_2, None)
        self.assertEqual(result, expected)

    def test_difference(self):
        patients_1, _ = self.create_spark_df({"patientID": [1, 2]})
        events_1, _ = self.create_spark_df(
            {"patientID": [1, 2], "value": ["DP", "DAS"]}
        )

        patients_2, _ = self.create_spark_df({"patientID": [1]})
        events_2, _ = self.create_spark_df({"patientID": [1], "value": ["DP"]})

        cohort1 = Cohort("liberal_fractures", "liberal_fractures", patients_1, events_1)

        cohort2 = Cohort("hospit_fractures", "hospit_fractures", patients_2, events_2)

        result = cohort1.difference(cohort2)

        patients_3, _ = self.create_spark_df({"patientID": [2]})
        events_3, _ = self.create_spark_df({"patientID": [2], "value": ["DAS"]})
        expected = Cohort("hospit_fractures", "hospit_fractures", patients_3, events_3)
        self.assertEqual(result, expected)

    def test_difference_all(self):
        patients_1, patients_pd_1 = self.create_spark_df({"patientID": [1, 2]})
        events_1, events_pd_1 = self.create_spark_df(
            {"patientID": [1, 2], "value": ["DP", "DAS"]}
        )

        patients_2, patients_pd_2 = self.create_spark_df({"patientID": [1]})
        events_2, events_pd_2 = self.create_spark_df(
            {"patientID": [1], "value": ["DP"]}
        )

        patients_3, patients_pd_3 = self.create_spark_df({"patientID": [1, 3]})

        cohort1 = Cohort("liberal_fractures", "liberal_fractures", patients_1, events_1)

        cohort2 = Cohort("hospit_fractures", "hospit_fractures", patients_2, events_2)

        cohort3 = Cohort("imb_fractures", "imb_fractures", patients_3, None)

        result = Cohort.difference_all([cohort1, cohort2, cohort3])

        patients_4, _ = self.create_spark_df({"patientID": [2]})
        expected = Cohort("hospit_fractures", "hospit_fractures", patients_4, None)
        self.assertEqual(result, expected)

    def test_has_subject_information(self):
        patients_1, _ = self.create_spark_df({"patientID": [1, 2]})
        cohort1 = Cohort("liberal_fractures", "liberal_fractures", patients_1)
        patients_2, _ = self.create_spark_df(
            {
                "patientID": [1, 2],
                "gender": [1, 1],
                "birthDate": [
                    pd.to_datetime("1993-10-09"),
                    pd.to_datetime("1992-03-14"),
                ],
                "deathDate": [
                    pd.to_datetime("1993-10-09"),
                    pd.to_datetime("1992-03-14"),
                ],
            }
        )

        cohort2 = Cohort("liberal_fractures", "liberal_fractures", patients_2, None)

        self.assertFalse(cohort1.has_subject_information())
        self.assertTrue(cohort2.has_subject_information())

    def test_add_subject_information(self):
        patients_1, _ = self.create_spark_df({"patientID": [1, 2]})
        events_1, _ = self.create_spark_df(
            {"patientID": [1, 2], "value": ["fracture", "fracture"]}
        )
        input1 = Cohort("liberal_fractures", "liberal_fractures", patients_1, events_1)
        patients_2, _ = self.create_spark_df(
            {
                "patientID": [1, 2],
                "gender": [1, 1],
                "birthDate": [
                    pd.to_datetime("1993-10-09"),
                    pd.to_datetime("1992-03-14"),
                ],
                "deathDate": [
                    pd.to_datetime("1993-10-09"),
                    pd.to_datetime("1992-03-14"),
                ],
            }
        )

        base_cohort1 = Cohort("patients", "patients", patients_2)
        input1.add_subject_information(base_cohort1, "error")

        self.assertTrue(
            input1.has_subject_information() and input1.subjects.count() == 2
        )

        patients_3, _ = self.create_spark_df(
            {
                "patientID": [1],
                "gender": [1],
                "birthDate": [pd.to_datetime("1993-10-09")],
                "deathDate": [pd.to_datetime("1993-10-09")],
            }
        )
        base_cohort2 = Cohort(
            "liberal_fractures", "liberal_fractures", patients_3, None
        )
        input2 = Cohort("liberal_fractures", "liberal_fractures", patients_1, events_1)
        input2.add_subject_information(base_cohort2, "omit")
        self.assertTrue(
            input2.has_subject_information()
            and input2.subjects.count() == 1
            and input2.events.count() == 2
        )

        input3 = Cohort("liberal_fractures", "liberal_fractures", patients_1, events_1)
        input3.add_subject_information(base_cohort2, "omit_all")
        self.assertTrue(
            input3.has_subject_information()
            and input3.subjects.count() == 1
            and input3.events.count() == 1
        )

    def test_add_age_information(self):
        subjects, df = self.create_spark_df(
            {"birthDate": [datetime(1993, 10, 9), datetime(1992, 3, 14)]}
        )

        input = Cohort("liberal_fractures", "liberal_fractures", subjects, None)

        input.add_age_information(datetime(2013, 1, 1))
        result = input
        expected_subjects, _ = self.create_spark_df({"age": [19, 20]})
        expected = Cohort(
            "liberal_fractures", "liberal_fractures", expected_subjects, None
        )
        self.assertTrue(
            data_frame_equality(
                result.subjects.select("age"), expected.subjects.select("age")
            )
        )

    def test_is_duration_events(self):
        schema = StructType(
            [
                StructField("patientID", IntegerType(), True),
                StructField("start", TimestampType(), True),
                StructField("end", TimestampType(), True),
            ]
        )

        patients_pd = pd.DataFrame({"patientID": [1, 2, 3]})
        patients = self.spark.createDataFrame(patients_pd)

        cohort1 = Cohort("patients", "patients", patients, None)
        self.assertFalse(cohort1.is_duration_events())

        data = [(1, datetime(1993, 10, 9), datetime(1993, 10, 9))]

        events = self.spark.createDataFrame(data=data, schema=schema)

        cohort2 = Cohort("patients", "patients", patients, events)
        self.assertTrue(cohort2.is_duration_events())

        data = [(1, datetime(1993, 10, 9), None), (2, datetime(1993, 10, 9), None)]

        events = self.spark.createDataFrame(data=data, schema=schema)

        cohort2 = Cohort("patients", "patients", patients, events)
        self.assertFalse(cohort2.is_duration_events())
