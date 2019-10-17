# License: BSD 3 clause

from unittest.mock import patch

from scalpel.core.cohort import Cohort
from scalpel.core.cohort_collection import CohortCollection
from .pyspark_tests import PySparkTest


class TestCohortCollection(PySparkTest):
    maxDiff = None

    @patch("scalpel.core.cohort.read_data_frame")
    def test_from_json(self, mock_read_data_frame):
        mock_read_data_frame.return_value = self.create_spark_df({"patientID": [1, 2]})
        metadata = {
            "class_name": "fr.polytechnique.cmap.cnam.study."
            "pioglitazone.PioglitazoneMain$",
            "start_timestamp": "2018-07-25T10:13:10Z",
            "end_timestamp": "2018-07-25T10:45:52Z",
            "operations": [
                {
                    "name": "extract_patients",
                    "inputs": ["DCIR", "MCO", "IR_BEN_R"],
                    "output_type": "patients",
                    "output_path": "/some/path/to/extract_patients/data",
                },
                {
                    "name": "drug_purchases",
                    "inputs": ["DCIR"],
                    "output_type": "dispensations",
                    "output_path": "/some/path/to/drug_purchases/data",
                    "population_path": "/some/path/to/drug_purchases/patients",
                },
                {
                    "name": "diagnoses",
                    "inputs": ["MCO", "IR_IMB_R"],
                    "output_type": "diagnosis",
                    "output_path": "/some/path/to/diagnoses/data",
                    "population_path": "/some/path/to/diagnoses/patients",
                },
                {
                    "name": "acts",
                    "inputs": ["DCIR", "MCO", "MCO_CE"],
                    "output_type": "acts",
                    "output_path": "/some/path/to/acts/data",
                    "population_path": "/some/path/to/acts/patients",
                },
                {
                    "name": "outcomes",
                    "inputs": ["acts", "diagnoses"],
                    "output_type": "outcomes",
                    "output_path": "/some/path/to/outcomes/data",
                    "population_path": "/some/path/to/outcomes/patients",
                },
                {
                    "name": "exposures",
                    "inputs": ["drug_purchases", "followup"],
                    "output_type": "exposures",
                    "output_path": "/some/path/to/exposures/data",
                    "population_path": "/some/path/to/exposures/patients",
                },
            ],
        }
        result = CohortCollection.load(metadata)

        expected_cohorts = {
            "extract_patients",
            "drug_purchases",
            "diagnoses",
            "acts",
            "outcomes",
            "exposures",
        }
        self.assertSetEqual(expected_cohorts, result.cohorts_names)

    @patch("scalpel.core.cohort_collection.Cohort")
    def test_union(self, mock_Cohort):
        cc1 = CohortCollection({"extract_patients": mock_Cohort, "acts": mock_Cohort})
        cc2 = CohortCollection(
            {
                "exposures": mock_Cohort,
                "outcomes": mock_Cohort,
                "extract_patients": mock_Cohort,
            }
        )

        result = cc1.union(cc2)
        expected_cohorts = {"extract_patients", "acts", "outcomes", "exposures"}
        self.assertSetEqual(expected_cohorts, result.cohorts_names)

    @patch("scalpel.core.cohort_collection.Cohort")
    def test_intersect(self, mock_Cohort):
        cc1 = CohortCollection({"extract_patients": mock_Cohort, "acts": mock_Cohort})
        cc2 = CohortCollection(
            {
                "exposures": mock_Cohort,
                "outcomes": mock_Cohort,
                "extract_patients": mock_Cohort,
            }
        )

        result = cc1.intersection(cc2)
        expected_cohorts = {"extract_patients"}
        self.assertSetEqual(expected_cohorts, result.cohorts_names)

    @patch("scalpel.core.cohort_collection.Cohort")
    def test_difference(self, mock_Cohort):
        cc1 = CohortCollection({"extract_patients": mock_Cohort, "acts": mock_Cohort})
        cc2 = CohortCollection(
            {
                "exposures": mock_Cohort,
                "outcomes": mock_Cohort,
                "extract_patients": mock_Cohort,
            }
        )

        result = cc1.difference(cc2)
        expected_cohorts = {"acts"}
        self.assertSetEqual(expected_cohorts, result.cohorts_names)

    @patch("scalpel.core.cohort_collection.Cohort")
    def test_union_all(self, mock_Cohort):
        cc1 = CohortCollection({"extract_patients": mock_Cohort, "acts": mock_Cohort})
        cc2 = CohortCollection(
            {
                "exposures": mock_Cohort,
                "outcomes": mock_Cohort,
                "extract_patients": mock_Cohort,
            }
        )
        cc3 = CohortCollection(
            {
                "diagnoses": mock_Cohort,
                "outcomes": mock_Cohort,
                "extract_patients": mock_Cohort,
            }
        )

        result = CohortCollection.union_all([cc1, cc2, cc3])
        expected_cohorts = {
            "extract_patients",
            "acts",
            "outcomes",
            "exposures",
            "diagnoses",
        }
        self.assertSetEqual(expected_cohorts, result.cohorts_names)

    @patch("scalpel.core.cohort_collection.Cohort")
    def test_intersect_all(self, mock_Cohort):
        cc1 = CohortCollection({"extract_patients": mock_Cohort, "acts": mock_Cohort})
        cc2 = CohortCollection(
            {
                "exposures": mock_Cohort,
                "outcomes": mock_Cohort,
                "extract_patients": mock_Cohort,
            }
        )
        cc3 = CohortCollection(
            {
                "diagnoses": mock_Cohort,
                "outcomes": mock_Cohort,
                "extract_patients": mock_Cohort,
            }
        )

        result = CohortCollection.intersect_all([cc1, cc2, cc3])
        expected_cohorts = {"extract_patients"}
        self.assertSetEqual(expected_cohorts, result.cohorts_names)

    @patch("scalpel.core.io.write_data_frame", return_value=None)
    def test_dump_metadata(self, mock_writing):
        df, _ = self.create_spark_df({"patientID": [1, 2]})
        cohort_1 = Cohort("test", "test", df, None)
        df_events, _ = self.create_spark_df(
            {"patientID": [1, 2], "category": ["test", "test"]}
        )

        cohort_2 = Cohort("events", "events", df, df_events)

        cc = CohortCollection({"test": cohort_1, "events": cohort_2})
        expected = sorted(
            {
                "operations": [
                    {
                        "output_type": "events",
                        "name": "events",
                        "output_path": "../../output/events/data",
                        "population_path": "../../output/events/subjects",
                    },
                    {
                        "output_type": "patients",
                        "output_path": "../../output/test/subjects",
                        "name": "test",
                    },
                ]
            }
        )

        result = sorted(cc.save("../../output"))
        self.assertEqual(expected, result)

    def test_eq(self):
        df, _ = self.create_spark_df({"patientID": [1, 2]})
        cohort_1 = Cohort("test", "test", df, None)
        df_events, _ = self.create_spark_df(
            {"patientID": [1, 2], "category": ["test", "test"]}
        )

        cohort_2 = Cohort("events", "events", df, df_events)

        cc1 = CohortCollection({"test": cohort_1, "events": cohort_2})
        cc2 = CohortCollection({"test": cohort_1, "events": cohort_2})
        self.assertEqual(cc1, cc2)

        cc3 = CohortCollection({"test1": cohort_1, "events": cohort_2})
        self.assertNotEqual(cc1, cc3)

        df, _ = self.create_spark_df({"patientID": [1, 45]})
        cohort_3 = Cohort("test", "test", df, None)
        cc4 = CohortCollection({"test": cohort_3, "events": cohort_2})

        self.assertNotEqual(cc1, cc4)

        self.assertNotEqual(cc1, df)
