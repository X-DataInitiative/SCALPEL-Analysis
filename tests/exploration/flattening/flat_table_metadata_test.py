from unittest.mock import patch

import pandas as pd

from src.exploration.flattening.flat_table_metadata import FlatTableMetadata
from tests.exploration.core.pyspark_tests import PySparkTest


class TestFlatTableMetadata(PySparkTest):
    @patch("src.exploration.flattening.flat_table.read_data_frame")
    def testFromJson(self, mock_read_data_frame):
        mock_read_data_frame.return_value = self.spark.createDataFrame(
            pd.DataFrame(
                {
                    "NUM_ENQ": ["1", "2", "3"],
                    "EXE_SOI_DTD": ["01/01/2015", "01/02/2016", "01/03/2017"],
                }
            )
        )
        json = """
        {
          "class_name" : "fr.polytechnique.cmap.cnam.flattening.FlatteningMain$",
          "flat_tables" : [
          {
            "name" : "DCIR",
            "path" : "/user/ds/CNAM378/flattening/2019_02_25/flat_table/DCIR",
            "join_keys" : ["NUM_ENQ", "EXE_SOI_DTD"]
          },
          {
            "name" : "MCO",
            "path" : "/user/ds/CNAM378/flattening/2019_02_25/flat_table/DCIR",
            "join_keys" : ["NUM_ENQ", "EXE_SOI_DTD"]
          }]
        }
        """
        repo = FlatTableMetadata.from_json(json)
        expected_names = {"MCO", "DCIR"}
        self.assertEqual(expected_names, repo.flat_table_names())

    @patch("src.exploration.flattening.flat_table_metadata.FlatTable")
    def test_exists(self, mock_flat_table):
        repo = FlatTableMetadata({"MCO": mock_flat_table, "DCIR": mock_flat_table})
        self.assertTrue(repo.exists("MCO"))

    @patch("src.exploration.flattening.flat_table_metadata.FlatTable")
    def test_get(self, mock_flat_table):
        repo = FlatTableMetadata({"MCO": mock_flat_table, "DCIR": mock_flat_table})
        self.assertEquals(repo.get("MCO"), mock_flat_table)
        self.assertRaises(KeyError, repo.get, "MCO_CE")

    @patch("src.exploration.flattening.flat_table_metadata.FlatTable")
    def test_add_flat_table(self, mock_flat_table):
        repo = FlatTableMetadata({"MCO": mock_flat_table, "DCIR": mock_flat_table})
        repo.add_flat_table("MCO_CE", mock_flat_table)
        expected_names = {"MCO", "DCIR", "MCO_CE"}
        self.assertEqual(expected_names, repo.flat_table_names())

    @patch("src.exploration.flattening.flat_table_metadata.FlatTable")
    def test_union(self, mock_flat_table):
        repo1 = FlatTableMetadata({"MCO": mock_flat_table})
        repo2 = FlatTableMetadata({"DCIR": mock_flat_table})
        repo = repo1.union(repo2)
        expected_names = {"MCO", "DCIR"}
        self.assertEqual(expected_names, repo.flat_table_names())

    @patch("src.exploration.flattening.flat_table_metadata.FlatTable")
    def test_union_all(self, mock_flat_table):
        repo1 = FlatTableMetadata({"MCO": mock_flat_table})
        repo2 = FlatTableMetadata({"DCIR": mock_flat_table})
        repo3 = FlatTableMetadata({"MCO_CE": mock_flat_table})
        repo = FlatTableMetadata.union_all([repo1, repo2, repo3])
        expected_names = {"MCO", "DCIR", "MCO_CE"}
        self.assertEqual(expected_names, repo.flat_table_names())

    @patch("src.exploration.flattening.flat_table_metadata.FlatTable")
    def test_intersection(self, mock_flat_table):
        repo1 = FlatTableMetadata({"MCO": mock_flat_table, "DCIR": mock_flat_table})
        repo2 = FlatTableMetadata({"DCIR": mock_flat_table})
        repo = repo1.intersection(repo2)
        expected_names = {"DCIR"}
        self.assertEqual(expected_names, repo.flat_table_names())

    @patch("src.exploration.flattening.flat_table_metadata.FlatTable")
    def test_intersection_all(self, mock_flat_table):
        repo1 = FlatTableMetadata({"MCO": mock_flat_table, "DCIR": mock_flat_table})
        repo2 = FlatTableMetadata({"DCIR": mock_flat_table})
        repo3 = FlatTableMetadata({"DCIR": mock_flat_table, "MCO_CE": mock_flat_table})
        repo = FlatTableMetadata.intersection_all([repo1, repo2, repo3])
        expected_names = {"DCIR"}
        self.assertEqual(expected_names, repo.flat_table_names())

    @patch("src.exploration.flattening.flat_table_metadata.FlatTable")
    def test_difference(self, mock_flat_table):
        repo1 = FlatTableMetadata({"MCO": mock_flat_table, "DCIR": mock_flat_table})
        repo2 = FlatTableMetadata({"DCIR": mock_flat_table})
        repo = repo1.difference(repo2)
        expected_names = {"MCO"}
        self.assertEqual(expected_names, repo.flat_table_names())

    @patch("src.exploration.flattening.flat_table_metadata.FlatTable")
    def test_difference_all(self, mock_flat_table):
        repo1 = FlatTableMetadata({"MCO": mock_flat_table, "DCIR": mock_flat_table})
        repo2 = FlatTableMetadata({"DCIR": mock_flat_table})
        repo3 = FlatTableMetadata({"DCIR": mock_flat_table, "MCO_CE": mock_flat_table})
        repo = FlatTableMetadata.difference_all([repo1, repo2, repo3])
        expected_names = {"MCO"}
        self.assertEqual(expected_names, repo.flat_table_names())
