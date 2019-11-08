# License: BSD 3 clause

from unittest.mock import patch

import pandas as pd

from scalpel.flattening.flat_table_collection import FlatTableCollection
from tests.core.pyspark_tests import PySparkTest


class TestFlatTableCollection(PySparkTest):
    @patch("scalpel.flattening.flat_table.read_data_frame")
    @patch("scalpel.flattening.single_table.read_data_frame")
    def testFromJson(
        self, mock_flat_table_read_data_frame, mock_single_table_read_data_frame
    ):
        mock_flat_table_read_data_frame.return_value = self.spark.createDataFrame(
            pd.DataFrame(
                {
                    "NUM_ENQ": ["1", "2", "3"],
                    "EXE_SOI_DTD": ["01/01/2015", "01/02/2016", "01/03/2017"],
                }
            )
        )
        mock_single_table_read_data_frame.return_value = self.spark.createDataFrame(
            pd.DataFrame(
                {
                    "NUM_ENQ": ["1", "2", "3"],
                    "EXE_SOI_DTD": ["01/01/2015", "01/02/2016", "01/03/2017"],
                }
            )
        )
        json = """
{
  "class_name" : "fr.polytechnique.cmap.cnam.flattening.FlatteningMainJoin$",
  "start_timestamp" : "2019-09-26T13:30:24Z",
  "end_timestamp" : "2019-09-26T17:02:25Z",
  "operations" : [{
    "output_table" : "ER_UCD_F",
    "output_path" : "/user/ds/CNAM243/flattening/single_table",
    "output_type" : "single_table",
    "sources" : ["/shared/Observapur/raw_data/DCIR_2010/ER_UCD_F_2010.CSV",
     "/shared/Observapur/raw_data/DCIR_2011/ER_UCD_F_2011.CSV",
     "/shared/Observapur/raw_data/DCIR_2012/ER_UCD_F_2012.CSV",
     "/shared/Observapur/raw_data/DCIR_2013/ER_UCD_F_2013.CSV",
     "/shared/Observapur/raw_data/DCIR_2014/ER_UCD_F_2014.CSV"],
    "join_keys" : []
  }, {
    "output_table" : "ER_ETE_F",
    "output_path" : "/user/ds/CNAM243/flattening/single_table",
    "output_type" : "single_table",
    "sources" : ["/shared/Observapur/raw_data/DCIR_2010/ER_ETE_F_2010.CSV",
    "/shared/Observapur/raw_data/DCIR_2011/ER_ETE_F_2011.CSV",
    "/shared/Observapur/raw_data/DCIR_2012/ER_ETE_F_2012.CSV",
    "/shared/Observapur/raw_data/DCIR_2013/ER_ETE_F_2013.CSV",
    "/shared/Observapur/raw_data/DCIR_2014/ER_ETE_F_2014.CSV"],
    "join_keys" : []
  }, {
    "output_table" : "ER_PHA_F",
    "output_path" : "/user/ds/CNAM243/flattening/single_table",
    "output_type" : "single_table",
    "sources" : ["/shared/Observapur/raw_data/DCIR_2010/ER_PHA_F_2010.CSV",
    "/shared/Observapur/raw_data/DCIR_2011/ER_PHA_F_2011.CSV",
    "/shared/Observapur/raw_data/DCIR_2012/ER_PHA_F_2012.CSV",
    "/shared/Observapur/raw_data/DCIR_2013/ER_PHA_F_2013.CSV",
    "/shared/Observapur/raw_data/DCIR_2014/ER_PHA_F_2014.CSV"],
    "join_keys" : []
  }, {
    "output_table" : "ER_PRS_F",
    "output_path" : "/user/ds/CNAM243/flattening/single_table",
    "output_type" : "single_table",
    "sources" : ["/shared/Observapur/raw_data/DCIR_2010/ER_PRS_F_2010.CSV",
    "/shared/Observapur/raw_data/DCIR_2011/ER_PRS_F_2011.CSV",
    "/shared/Observapur/raw_data/DCIR_2012/ER_PRS_F_2012.CSV",
    "/shared/Observapur/raw_data/DCIR_2013/ER_PRS_F_2013.CSV",
    "/shared/Observapur/raw_data/DCIR_2014/ER_PRS_F_2014.CSV"],
    "join_keys" : []
  }, {
    "output_table" : "ER_CAM_F",
    "output_path" : "/user/ds/CNAM243/flattening/single_table",
    "output_type" : "single_table",
    "sources" : ["/shared/Observapur/raw_data/DCIR_2010/ER_CAM_F_2010.CSV",
    "/shared/Observapur/raw_data/DCIR_2011/ER_CAM_F_2011.CSV",
    "/shared/Observapur/raw_data/DCIR_2012/ER_CAM_F_2012.CSV",
    "/shared/Observapur/raw_data/DCIR_2013/ER_CAM_F_2013.CSV",
    "/shared/Observapur/raw_data/DCIR_2014/ER_CAM_F_2014.CSV"],
    "join_keys" : []
  }, {
    "output_table" : "DCIR",
    "output_path" : "/user/ds/CNAM243/flattening/flat_table",
    "output_type" : "flat_table",
    "sources" : ["ER_PRS_F", "ER_UCD_F", "ER_CAM_F", "ER_ETE_F", "ER_PHA_F"],
    "join_keys" : ["DCT_ORD_NUM", "FLX_DIS_DTD", "FLX_EMT_NUM", "FLX_EMT_ORD",
    "FLX_EMT_TYP", "FLX_TRT_DTD", "ORG_CLE_NUM", "PRS_ORD_NUM", "REM_TYP_AFF"]
  }]
}
        """
        repo = FlatTableCollection.from_json(json)
        expected_flat_names = {"DCIR"}
        expected_single_names = {
            "ER_PRS_F",
            "ER_UCD_F",
            "ER_CAM_F",
            "ER_ETE_F",
            "ER_PHA_F",
        }
        self.assertEqual(expected_flat_names, repo.flat_table_names())
        self.assertEqual(
            expected_single_names, repo.single_table_names_from_flat_table("DCIR")
        )
        self.assertEqual(set(), repo.single_table_names_from_flat_table("MCO"))
        dcir = repo.get("DCIR")
        self.assertEqual("ER_PRS_F", dcir.single_tables.get("ER_PRS_F").name)
        self.assertEqual("ER_PRS_F", dcir.single_tables.get("ER_PRS_F").characteristics)

    @patch("scalpel.flattening.flat_table_collection.FlatTable")
    def test_exists(self, mock_flat_table):
        repo = FlatTableCollection({"MCO": mock_flat_table, "DCIR": mock_flat_table})
        self.assertTrue(repo.exists("MCO"))

    @patch("scalpel.flattening.flat_table_collection.FlatTable")
    def test_get(self, mock_flat_table):
        repo = FlatTableCollection({"MCO": mock_flat_table, "DCIR": mock_flat_table})
        self.assertEquals(repo.get("MCO"), mock_flat_table)
        self.assertRaises(KeyError, repo.get, "MCO_CE")

    @patch("scalpel.flattening.flat_table_collection.FlatTable")
    def test_add_flat_table(self, mock_flat_table):
        repo = FlatTableCollection({"MCO": mock_flat_table, "DCIR": mock_flat_table})
        repo.add_flat_table("MCO_CE", mock_flat_table)
        expected_names = {"MCO", "DCIR", "MCO_CE"}
        self.assertEqual(expected_names, repo.flat_table_names())

    @patch("scalpel.flattening.flat_table_collection.FlatTable")
    def test_union(self, mock_flat_table):
        repo1 = FlatTableCollection({"MCO": mock_flat_table})
        repo2 = FlatTableCollection({"DCIR": mock_flat_table})
        repo = repo1.union(repo2)
        expected_names = {"MCO", "DCIR"}
        self.assertEqual(expected_names, repo.flat_table_names())

    @patch("scalpel.flattening.flat_table_collection.FlatTable")
    def test_union_all(self, mock_flat_table):
        repo1 = FlatTableCollection({"MCO": mock_flat_table})
        repo2 = FlatTableCollection({"DCIR": mock_flat_table})
        repo3 = FlatTableCollection({"MCO_CE": mock_flat_table})
        repo = FlatTableCollection.union_all([repo1, repo2, repo3])
        expected_names = {"MCO", "DCIR", "MCO_CE"}
        self.assertEqual(expected_names, repo.flat_table_names())

    @patch("scalpel.flattening.flat_table_collection.FlatTable")
    def test_intersection(self, mock_flat_table):
        repo1 = FlatTableCollection({"MCO": mock_flat_table, "DCIR": mock_flat_table})
        repo2 = FlatTableCollection({"DCIR": mock_flat_table})
        repo = repo1.intersection(repo2)
        expected_names = {"DCIR"}
        self.assertEqual(expected_names, repo.flat_table_names())

    @patch("scalpel.flattening.flat_table_collection.FlatTable")
    def test_intersection_all(self, mock_flat_table):
        repo1 = FlatTableCollection({"MCO": mock_flat_table, "DCIR": mock_flat_table})
        repo2 = FlatTableCollection({"DCIR": mock_flat_table})
        repo3 = FlatTableCollection(
            {"DCIR": mock_flat_table, "MCO_CE": mock_flat_table}
        )
        repo = FlatTableCollection.intersection_all([repo1, repo2, repo3])
        expected_names = {"DCIR"}
        self.assertEqual(expected_names, repo.flat_table_names())

    @patch("scalpel.flattening.flat_table_collection.FlatTable")
    def test_difference(self, mock_flat_table):
        repo1 = FlatTableCollection({"MCO": mock_flat_table, "DCIR": mock_flat_table})
        repo2 = FlatTableCollection({"DCIR": mock_flat_table})
        repo = repo1.difference(repo2)
        expected_names = {"MCO"}
        self.assertEqual(expected_names, repo.flat_table_names())

    @patch("scalpel.flattening.flat_table_collection.FlatTable")
    def test_difference_all(self, mock_flat_table):
        repo1 = FlatTableCollection({"MCO": mock_flat_table, "DCIR": mock_flat_table})
        repo2 = FlatTableCollection({"DCIR": mock_flat_table})
        repo3 = FlatTableCollection(
            {"DCIR": mock_flat_table, "MCO_CE": mock_flat_table}
        )
        repo = FlatTableCollection.difference_all([repo1, repo2, repo3])
        expected_names = {"MCO"}
        self.assertEqual(expected_names, repo.flat_table_names())
