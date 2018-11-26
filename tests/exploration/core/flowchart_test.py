import unittest

from src.exploration.core.cohort import Cohort
from src.exploration.core.flowchart import metadata_from_flowchart
from src.exploration.core.metadata import Metadata
from .pyspark_tests import PySparkTest
from src.exploration.core.flowchart import get_steps


class TestFlowchart(PySparkTest):

    def test_get_steps(self):
        """Test that the parsing of steps."""

        input = '''
        {
            "intermediate_operations": {
                "operation": {
                    "type": "union",
                    "name": "outcome",
                    "parents": ["liberal_fractures", "hospit_fractures"]
                }
            },
            "steps": [
                "extract_patients",
                "exposures",
                "filter_patients",
                "outcome"
            ]
        }
        '''
        result = get_steps(input)
        expected = ["extract_patients",
                    "exposures",
                    "filter_patients",
                    "outcome"]

        self.assertSequenceEqual(result, expected)

    def test_metadata_from_flowchart(self):
        input = '''
        {
            "intermediate_operations": {
                "operation": {
                    "type": "union",
                    "name": "outcome",
                    "parents": ["liberal_fractures", "hospit_fractures"]
                }
            },
            "steps": [
                "extract_patients",
                "exposures",
                "filter_patients",
                "outcome"
            ]
        }
        '''

        df, _ = self.create_spark_df({"patientID": [1, 2, 3]})

        metadata = Metadata(
            {"liberal_fractures": Cohort("liberal_fractures", "liberal_fractures",
                                         df, None),
             "hospit_fractures": Cohort("hospit_fractures", "hospit_fractures",
                                        df, None)})

        result = metadata_from_flowchart(metadata, input)

        self.assertSetEqual(set(result.cohorts.keys()), {"liberal_fractures",
                                                         "hospit_fractures",
                                                         "outcome"})
