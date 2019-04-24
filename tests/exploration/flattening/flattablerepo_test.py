from unittest.mock import patch

import pandas as pd

from src.exploration.flattening.flattablerepo import FlatTableRepo
from tests.exploration.core.pyspark_tests import PySparkTest


class TestFlatTableRepo(PySparkTest):
    @patch("src.exploration.flattening.flattable.read_data_frame")
    def testFromJson(self, mock_read_data_frame):
        mock_read_data_frame.return_value = self.spark.createDataFrame(
            pd.DataFrame({'NUM_ENQ': ['1', '2', '3'], 'EXE_SOI_DTD': ['01/01/2015', '01/02/2016', '01/03/2017']}))
        json = '''
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
        '''
        repo = FlatTableRepo.from_json(json)
        expected_names = {"MCO", "DCIR"}
        self.assertEqual(expected_names, repo.flat_table_names())

    @patch("src.exploration.flattening.flattablerepo.FlatTable")
    def test_union(self, mock_flat_table):
        repo1 = FlatTableRepo({'MCO': mock_flat_table})
        repo2 = FlatTableRepo({'DCIR': mock_flat_table})
        repo = repo1.union(repo2)
        expected_names = {"MCO", "DCIR"}
        self.assertEqual(expected_names, repo.flat_table_names())

    @patch("src.exploration.flattening.flattablerepo.FlatTable")
    def test_union_all(self, mock_flat_table):
        repo1 = FlatTableRepo({'MCO': mock_flat_table})
        repo2 = FlatTableRepo({'DCIR': mock_flat_table})
        repo3 = FlatTableRepo({'MCO_CE': mock_flat_table})
        repo = FlatTableRepo.union_all([repo1, repo2, repo3])
        expected_names = {"MCO", "DCIR", "MCO_CE"}
        self.assertEqual(expected_names, repo.flat_table_names())

    @patch("src.exploration.flattening.flattablerepo.FlatTable")
    def test_intersection(self, mock_flat_table):
        repo1 = FlatTableRepo({'MCO': mock_flat_table, 'DCIR': mock_flat_table})
        repo2 = FlatTableRepo({'DCIR': mock_flat_table})
        repo = repo1.intersection(repo2)
        expected_names = {"DCIR"}
        self.assertEqual(expected_names, repo.flat_table_names())

    @patch("src.exploration.flattening.flattablerepo.FlatTable")
    def test_intersection_all(self, mock_flat_table):
        repo1 = FlatTableRepo({'MCO': mock_flat_table, 'DCIR': mock_flat_table})
        repo2 = FlatTableRepo({'DCIR': mock_flat_table})
        repo3 = FlatTableRepo({'DCIR': mock_flat_table, 'MCO_CE': mock_flat_table})
        repo = FlatTableRepo.intersection_all([repo1, repo2, repo3])
        expected_names = {"DCIR"}
        self.assertEqual(expected_names, repo.flat_table_names())

    @patch("src.exploration.flattening.flattablerepo.FlatTable")
    def test_difference(self, mock_flat_table):
        repo1 = FlatTableRepo({'MCO': mock_flat_table, 'DCIR': mock_flat_table})
        repo2 = FlatTableRepo({'DCIR': mock_flat_table})
        repo = repo1.difference(repo2)
        expected_names = {"MCO"}
        self.assertEqual(expected_names, repo.flat_table_names())

    @patch("src.exploration.flattening.flattablerepo.FlatTable")
    def test_difference_all(self, mock_flat_table):
        repo1 = FlatTableRepo({'MCO': mock_flat_table, 'DCIR': mock_flat_table})
        repo2 = FlatTableRepo({'DCIR': mock_flat_table})
        repo3 = FlatTableRepo({'DCIR': mock_flat_table, 'MCO_CE': mock_flat_table})
        repo = FlatTableRepo.difference_all([repo1, repo2, repo3])
        expected_names = {"MCO"}
        self.assertEqual(expected_names, repo.flat_table_names())
