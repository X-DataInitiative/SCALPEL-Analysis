from unittest.mock import patch

from src.exploration.core.metadata import Metadata
from .pyspark_tests import PySparkTest


class TestMetadata(PySparkTest):

    @patch('src.exploration.core.cohort.read_data_frame')
    def test_from_json(self, mock_read_data_frame):
        mock_read_data_frame.return_value = self.create_spark_df({"patientID": [1, 2]})
        input = '''
        {
  "class_name" : "fr.polytechnique.cmap.cnam.study.pioglitazone.PioglitazoneMain$",
  "start_timestamp" : "2018-07-25T10:13:10Z",
  "end_timestamp" : "2018-07-25T10:45:52Z",
  "operations" : [ {
    "name" : "extract_patients",
    "inputs" : [ "DCIR", "MCO", "IR_BEN_R" ],
    "output_type" : "patients",
    "output_path" : "/shared/Observapur/staging/pio/extract_patients/data"
  }, {
    "name" : "drug_purchases",
    "inputs" : [ "DCIR" ],
    "output_type" : "dispensations",
    "output_path" : "/shared/Observapur/staging/pio/drug_purchases/data",
    "population_path" : "/shared/Observapur/staging/pio/drug_purchases/patients"
  }, {
    "name" : "diagnoses",
    "inputs" : [ "MCO", "IR_IMB_R" ],
    "output_type" : "diagnosis",
    "output_path" : "/shared/Observapur/staging/pio/diagnoses/data",
    "population_path" : "/shared/Observapur/staging/pio/diagnoses/patients"
  }, {
    "name" : "acts",
    "inputs" : [ "DCIR", "MCO", "MCO_CE" ],
    "output_type" : "acts",
    "output_path" : "/shared/Observapur/staging/pio/acts/data",
    "population_path" : "/shared/Observapur/staging/pio/acts/patients"
  }, {
    "name" : "outcomes",
    "inputs" : [ "acts", "diagnoses" ],
    "output_type" : "outcomes",
    "output_path" : "/shared/Observapur/staging/pio/outcomes/data",
    "population_path" : "/shared/Observapur/staging/pio/outcomes/patients"
  }, {
    "name" : "exposures",
    "inputs" : [ "drug_purchases", "followup" ],
    "output_type" : "exposures",
    "output_path" : "/shared/Observapur/staging/pio/exposures/data",
    "population_path" : "/shared/Observapur/staging/pio/exposures/patients"
  } ]
}
        '''
        result = Metadata.from_json(input)

        expected_cohorts = {"extract_patients", "drug_purchases",
                            "diagnoses", "acts", "outcomes", "exposures"}
        self.assertSetEqual(expected_cohorts, result.cohorts_names())

    @patch('src.exploration.core.metadata.Cohort')
    def test_union(self, mock_Cohort):

        meta1 = Metadata({"extract_patients": mock_Cohort, "acts": mock_Cohort})
        meta2 = Metadata({"exposures": mock_Cohort, "outcomes": mock_Cohort,
                          "extract_patients": mock_Cohort})

        result = meta1.union(meta2)
        expected_cohorts = {"extract_patients", "acts", "outcomes", "exposures"}
        self.assertSetEqual(expected_cohorts, result.cohorts_names())

    @patch('src.exploration.core.metadata.Cohort')
    def test_intersect(self, mock_Cohort):
        meta1 = Metadata({"extract_patients": mock_Cohort, "acts": mock_Cohort})
        meta2 = Metadata({"exposures": mock_Cohort, "outcomes": mock_Cohort,
                          "extract_patients": mock_Cohort})

        result = meta1.intersection(meta2)
        expected_cohorts = {"extract_patients"}
        self.assertSetEqual(expected_cohorts, result.cohorts_names())

    @patch('src.exploration.core.metadata.Cohort')
    def test_difference(self, mock_Cohort):
        meta1 = Metadata({"extract_patients": mock_Cohort, "acts": mock_Cohort})
        meta2 = Metadata({"exposures": mock_Cohort, "outcomes": mock_Cohort,
                          "extract_patients": mock_Cohort})

        result = meta1.difference(meta2)
        expected_cohorts = {"acts"}
        self.assertSetEqual(expected_cohorts, result.cohorts_names())
