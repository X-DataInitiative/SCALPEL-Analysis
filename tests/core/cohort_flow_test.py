# License: BSD 3 clause

from scalpel.core.cohort import Cohort
from scalpel.core.cohort_flow import get_steps, cohort_collection_from_cohort_flow
from scalpel.core.cohort_collection import CohortCollection
from scalpel.core.cohort_flow import CohortFlow
from .pyspark_tests import PySparkTest
import pytz


class TestCohortFlow(PySparkTest):
    def test_get_steps(self):
        """Test that the parsing of cohorts."""

        input = """
        {
            "intermediate_operations": {
                "operation": {
                    "type": "union",
                    "name": "outcome",
                    "parents": ["liberal_fractures", "hospit_fractures"]
                }
            },
            "cohorts": [
                "extract_patients",
                "exposures",
                "filter_patients",
                "outcome"
            ]
        }
        """
        result = get_steps(input)
        expected = ["extract_patients", "exposures", "filter_patients", "outcome"]

        self.assertSequenceEqual(result, expected)

    def test_cohort_collection_from_cohort_flow(self):
        input = """
        {
            "intermediate_operations": {
                "operation": {
                    "type": "union",
                    "name": "outcome",
                    "parents": ["liberal_fractures", "hospit_fractures"]
                }
            },
            "cohorts": [
                "extract_patients",
                "exposures",
                "filter_patients",
                "outcome"
            ]
        }
        """

        df, _ = self.create_spark_df({"patientID": [1, 2, 3]})

        cc = CohortCollection(
            {
                "liberal_fractures": Cohort(
                    "liberal_fractures", "liberal_fractures", df, None
                ),
                "hospit_fractures": Cohort(
                    "hospit_fractures", "hospit_fractures", df, None
                ),
            }
        )

        result = cohort_collection_from_cohort_flow(cc, input)

        self.assertSetEqual(
            set(result.cohorts.keys()),
            {"liberal_fractures", "hospit_fractures", "outcome"},
        )

    def test_steps_flowchart(self):
        patients = {
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

        exposure_events = {
            "patientID": ["0", "10", "4", "2"],  # uuid
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

        outcome_events = {
            "patientID": ["0", "3", "4", "22"],  # uuid
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
                pytz.datetime.datetime(2011, 11, 22, tzinfo=pytz.UTC),
            ],
            "value": ["bar"] * 4,
            "category": ["outcome"] * 4,
            "groupID": [0] * 4,
            "weight": [1] * 4,
        }

        patients_df, _ = self.create_spark_df(patients)
        exp_events_df, _ = self.create_spark_df(exposure_events)
        out_events_df, _ = self.create_spark_df(outcome_events)
        base_population = Cohort(
            "base_population", "base_population", patients_df, None
        )

        exposures = Cohort(
            "exposures",
            "exposures",
            exp_events_df.select("patientID").distinct(),
            exp_events_df,
        )

        outcomes = Cohort(
            "outcomes",
            "outcomes",
            out_events_df.select("patientID").distinct(),
            out_events_df,
        )

        flow = CohortFlow([base_population, exposures, outcomes])

        expected_step_1 = {
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

        expected_step_2 = {
            "patientID": ["0", "2", "4"],  # uuid
            "gender": [1, 2, 1],  # in {1, 2}
            "birthDate": [
                pytz.datetime.datetime(1934, 7, 27, tzinfo=pytz.UTC),
                pytz.datetime.datetime(1942, 1, 12, tzinfo=pytz.UTC),
                pytz.datetime.datetime(1937, 12, 31, tzinfo=pytz.UTC),
            ],
            "deathDate": [
                None,
                None,
                pytz.datetime.datetime(2012, 12, 10, tzinfo=pytz.UTC),
            ],  # can be null
        }

        expected_step_3 = {
            "patientID": ["0", "4"],  # uuid
            "gender": [1, 1],  # in {1, 2}
            "birthDate": [
                pytz.datetime.datetime(1934, 7, 27, tzinfo=pytz.UTC),
                pytz.datetime.datetime(1937, 12, 31, tzinfo=pytz.UTC),
            ],
            "deathDate": [
                None,
                pytz.datetime.datetime(2012, 12, 10, tzinfo=pytz.UTC),
            ],  # can be null
        }

        step1_df, _ = self.create_spark_df(expected_step_1)
        step2_df, _ = self.create_spark_df(expected_step_2)
        step3_df, _ = self.create_spark_df(expected_step_3)
        step_1_cohort = Cohort("step1", "step1", step1_df, None)

        step_2_cohort = Cohort("step2", "step2", step2_df, None)

        step_3_cohort = Cohort("step3", "step3", step3_df, None)

        for result, expected in zip(
            flow, [step_1_cohort, step_2_cohort, step_3_cohort]
        ):
            self.assertEqual(result, expected)

        # Case where the Flowchart has only one element
        flow_2 = CohortFlow([base_population])
        for result, expected in zip(flow_2, [step_1_cohort]):
            self.assertEqual(result, expected)

        # Case where the Flowchart is empty
        with self.assertWarns(Warning):
            CohortFlow([])

    def test_prepend_cohort_flowchart(self):
        patients = {
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

        exposure_events = {
            "patientID": ["0", "10", "4", "2"],  # uuid
            "start": [
                pytz.datetime.datetime(2010, 6, 7, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2011, 3, 28, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2011, 7, 3, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2010, 11, 22, tzinfo=pytz.UTC),
            ],
            "end": [
                pytz.datetime.datetime(1934, 7, 27, tzinfo=pytz.UTC),
                None,
                None,
                pytz.datetime.datetime(2011, 11, 22, tzinfo=pytz.UTC),
            ],
            "value": ["foo"] * 4,
            "category": ["exposure"] * 4,
            "groupID": [0] * 4,
            "weight": [1] * 4,
        }

        outcome_events = {
            "patientID": ["0", "3", "4", "22"],  # uuid
            "start": [
                pytz.datetime.datetime(2010, 6, 8, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2011, 3, 29, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2011, 7, 4, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2010, 11, 23, tzinfo=pytz.UTC),
            ],
            "end": [
                pytz.datetime.datetime(1934, 7, 27, tzinfo=pytz.UTC),
                None,
                None,
                pytz.datetime.datetime(2011, 11, 22, tzinfo=pytz.UTC),
            ],
            "value": ["bar"] * 4,
            "category": ["outcome"] * 4,
            "groupID": [0] * 4,
            "weight": [1] * 4,
        }

        patients_df, _ = self.create_spark_df(patients)
        exp_events_df, _ = self.create_spark_df(exposure_events)
        out_events_df, _ = self.create_spark_df(outcome_events)
        base_population = Cohort(
            "base_population", "base_population", patients_df, None
        )

        exposures = Cohort(
            "exposures",
            "exposures",
            exp_events_df.select("patientID").distinct(),
            exp_events_df,
        )

        outcomes = Cohort(
            "outcomes",
            "outcomes",
            out_events_df.select("patientID").distinct(),
            out_events_df,
        )

        flow = CohortFlow([base_population, exposures])

        expected_step_1 = outcome_events

        expected_step_2 = {
            "patientID": ["0", "3", "4"],  # uuid
            "start": [
                pytz.datetime.datetime(2010, 6, 8, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2011, 3, 29, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2011, 7, 4, tzinfo=pytz.UTC),
            ],
            "end": [pytz.datetime.datetime(1934, 7, 27, tzinfo=pytz.UTC), None, None],
            "value": ["bar"] * 3,
            "category": ["outcome"] * 3,
            "groupID": [0] * 3,
            "weight": [1] * 3,
        }

        expected_step_3 = {
            "patientID": ["0", "4"],  # uuid
            "start": [
                pytz.datetime.datetime(2010, 6, 8, tzinfo=pytz.UTC),
                pytz.datetime.datetime(2011, 7, 4, tzinfo=pytz.UTC),
            ],
            "end": [pytz.datetime.datetime(1934, 7, 27, tzinfo=pytz.UTC), None],
            "value": ["bar"] * 2,
            "category": ["outcome"] * 2,
            "groupID": [0] * 2,
            "weight": [1] * 2,
        }

        step1_df, _ = self.create_spark_df(expected_step_1)
        step2_df, _ = self.create_spark_df(expected_step_2)
        step3_df, _ = self.create_spark_df(expected_step_3)
        step_1_cohort = Cohort(
            "step1", "step1", step1_df.select("patientID").distinct(), step1_df
        )

        step_2_cohort = Cohort(
            "step2", "step2", step2_df.select("patientID").distinct(), step2_df
        )

        step_3_cohort = Cohort(
            "step3", "step3", step3_df.select("patientID").distinct(), step3_df
        )

        for result, expected in zip(
            flow.prepend_cohort(outcomes),
            [step_1_cohort, step_2_cohort, step_3_cohort],
        ):
            self.assertEqual(result, expected)
