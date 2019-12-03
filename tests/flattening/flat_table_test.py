# License: BSD 3 clause

import copy
from collections import OrderedDict

import pandas as pd

from scalpel.flattening.flat_table import FlatTable, HistoryTable
from scalpel.flattening.single_table import SingleTable
from tests.core.pyspark_tests import PySparkTest


class TestFlatTable(PySparkTest):
    def test_setter_validation(self):
        df1 = self.spark.createDataFrame(
            pd.DataFrame(
                {
                    "NUM_ENQ": ["1", "2", "3"],
                    "EXE_SOI_DTD": ["01/01/2015", "01/02/2016", "01/03/2017"],
                }
            )
        )
        ft1 = FlatTable("FT1", df1, "FT1", ["NUM_ENQ", "EXE_SOI_DTD"], {})
        with self.assertRaises(TypeError) as context:
            ft1.name = None
        self.assertTrue("Expected a string" in str(context.exception))
        with self.assertRaises(TypeError) as context:
            ft1.source = None
        self.assertTrue("Expected a Spark DataFrame" in str(context.exception))
        with self.assertRaises(TypeError) as context:
            ft1.characteristics = None
        self.assertTrue("Expected a string" in str(context.exception))
        with self.assertRaises(TypeError) as context:
            ft1.join_keys = None
        self.assertTrue("Expected a List" in str(context.exception))
        with self.assertRaises(TypeError) as context:
            ft1.single_tables = None

    def test_eq(self):
        df1 = self.spark.createDataFrame(
            pd.DataFrame(
                {
                    "NUM_ENQ": ["1", "2", "3"],
                    "EXE_SOI_DTD": ["01/01/2015", "01/02/2016", "01/03/2017"],
                }
            )
        )
        df2 = self.spark.createDataFrame(
            pd.DataFrame(
                {
                    "NUM_ENQ": ["3", "2", "1"],
                    "EXE_SOI_DTD": ["01/03/2017", "01/02/2016", "01/01/2015"],
                }
            )
        )
        df3 = self.spark.createDataFrame(
            pd.DataFrame(
                {
                    "NUM_ENQ": ["3", "2", "1"],
                    "EXE_SOI_DTF": ["01/03/2017", "01/02/2016", "01/01/2015"],
                }
            )
        )
        ft1 = FlatTable("FT1", df1, "FT1", ["NUM_ENQ", "EXE_SOI_DTD"], {})
        ft2 = copy.copy(ft1)
        ft3 = copy.copy(ft1)
        ft3.name = "FT2"
        ft3.source = df2
        ft3.characteristics = "FT2"
        ft3.join_keys = ["NUM_ENQ", "EXE_SOI_DTF"]
        ft3 = copy.copy(ft1)
        ft3.name = "FT3"
        ft3.source = df3
        ft3.characteristics = "FT3"
        ft3.join_keys = ["NUM_ENQ", "EXE_SOI_DTF"]
        self.assertEqual(ft1, ft2)
        self.assertNotEqual(ft1, ft3)

    def test_in(self):
        df1 = self.spark.createDataFrame(
            pd.DataFrame(
                {
                    "NUM_ENQ": ["1", "2", "3"],
                    "EXE_SOI_DTD": ["01/01/2015", "01/02/2016", "01/03/2017"],
                }
            )
        )
        df2 = self.spark.createDataFrame(
            pd.DataFrame(
                {
                    "NUM_ENQ": ["3", "2", "2"],
                    "EXE_SOI_DTD": ["01/03/2017", "01/02/2016", "01/02/2016"],
                }
            )
        )
        df3 = self.spark.createDataFrame(
            pd.DataFrame(
                {"NUM_ENQ": ["3", "2"], "EXE_SOI_DTF": ["01/03/2017", "01/02/2016"]}
            )
        )
        ft1 = FlatTable("FT1", df1, "FT1", ["NUM_ENQ", "EXE_SOI_DTD"], {})
        ft2 = FlatTable("FT2", df2, "FT2", ["NUM_ENQ", "EXE_SOI_DTD"], {})
        ft3 = FlatTable("FT3", df3, "FT3", ["NUM_ENQ", "EXE_SOI_DTF"], {})
        self.assertIn(ft2, ft1)
        self.assertNotIn(ft1, ft2)
        self.assertNotIn(ft3, ft1)

    def test_getitem(self):
        df1 = self.spark.createDataFrame(
            pd.DataFrame(
                {
                    "NUM_ENQ": ["1", "2", "3"],
                    "EXE_SOI_DTD": ["01/01/2015", "01/02/2016", "01/03/2017"],
                }
            )
        )
        ft1 = FlatTable("FT1", df1, "FT1", ["NUM_ENQ", "EXE_SOI_DTD"], {})
        ft2 = FlatTable("FT2", ft1["NUM_ENQ"], "FT2", ["NUM_ENQ"], {})
        df3 = self.spark.createDataFrame(pd.DataFrame({"NUM_ENQ": ["1", "2", "3"]}))
        ft3 = FlatTable("FT3", df3, "FT3", ["NUM_ENQ"], {})
        self.assertEqual(ft3, ft2)
        with self.assertRaises(TypeError) as context:
            ft1[1]
        self.assertTrue("Expected a str" in str(context.exception))

    def test_union(self):
        df1 = self.spark.createDataFrame(
            pd.DataFrame(
                {
                    "NUM_ENQ": ["1", "2", "3"],
                    "EXE_SOI_DTD": ["01/01/2015", "01/02/2016", "01/03/2017"],
                }
            )
        )
        df2 = self.spark.createDataFrame(
            pd.DataFrame(
                {"NUM_ENQ": ["3", "2"], "EXE_SOI_DTD": ["01/03/2017", "01/02/2016"]}
            )
        )
        df3 = self.spark.createDataFrame(
            pd.DataFrame(
                {"NUM_ENQ": ["3", "2"], "EXE_SOI_DTF": ["01/03/2017", "01/02/2016"]}
            )
        )
        single_1 = SingleTable("ST1", df1, "ST1")
        single_2 = SingleTable("ST2", df2, "ST2")
        ft1 = FlatTable(
            "FT1", df1, "FT1", ["NUM_ENQ", "EXE_SOI_DTD"], {"ST1": single_1}
        )
        ft2 = FlatTable(
            "FT2", df2, "FT2", ["NUM_ENQ", "EXE_SOI_DTD"], {"ST2": single_2}
        )
        ft3 = FlatTable("FT3", df3, "FT3", ["NUM_ENQ", "EXE_SOI_DTF"], {})
        ft = ft1.union(ft2)
        df4 = self.spark.createDataFrame(
            pd.DataFrame(
                {
                    "NUM_ENQ": ["1", "2", "3"],
                    "EXE_SOI_DTD": ["01/01/2015", "01/02/2016", "01/03/2017"],
                }
            )
        )
        expected_ft = FlatTable(
            "result",
            df4,
            "result",
            ["NUM_ENQ", "EXE_SOI_DTD"],
            {"ST1": single_1, "ST2": single_2},
        )
        self.assertEqual(expected_ft, ft)
        self.assertRaises(ValueError, ft1.union, ft3)

    def test_intersection(self):
        df1 = self.spark.createDataFrame(
            pd.DataFrame(
                {"NUM_ENQ": ["1", "2"], "EXE_SOI_DTD": ["01/01/2015", "01/02/2016"]}
            )
        )
        df2 = self.spark.createDataFrame(
            pd.DataFrame(
                {"NUM_ENQ": ["3", "2"], "EXE_SOI_DTD": ["01/03/2017", "01/02/2016"]}
            )
        )
        df3 = self.spark.createDataFrame(
            pd.DataFrame(
                {"NUM_ENQ": ["3", "2"], "EXE_SOI_DTF": ["01/03/2017", "01/02/2016"]}
            )
        )
        ft1 = FlatTable("FT1", df1, "FT1", ["NUM_ENQ", "EXE_SOI_DTD"], {})
        ft2 = FlatTable("FT2", df2, "FT2", ["NUM_ENQ", "EXE_SOI_DTD"], {})
        ft3 = FlatTable("FT3", df3, "FT3", ["NUM_ENQ", "EXE_SOI_DTF"], {})
        ft = ft1.intersection(ft2)
        df4 = self.spark.createDataFrame(
            pd.DataFrame(
                OrderedDict([("NUM_ENQ", ["2"]), ("EXE_SOI_DTD", ["01/02/2016"])])
            )
        )
        expected_ft = FlatTable("result", df4, "result", ["NUM_ENQ", "EXE_SOI_DTD"], {})
        self.assertEqual(expected_ft, ft)
        self.assertRaises(ValueError, ft1.intersection, ft3)

    def test_difference(self):
        df1 = self.spark.createDataFrame(
            pd.DataFrame(
                {"NUM_ENQ": ["1", "2"], "EXE_SOI_DTD": ["01/01/2015", "01/02/2016"]}
            )
        )
        df2 = self.spark.createDataFrame(
            pd.DataFrame(
                {"NUM_ENQ": ["3", "2"], "EXE_SOI_DTD": ["01/03/2017", "01/02/2016"]}
            )
        )
        df3 = self.spark.createDataFrame(
            pd.DataFrame(
                {"NUM_ENQ": ["3", "2"], "EXE_SOI_DTF": ["01/03/2017", "01/02/2016"]}
            )
        )
        ft1 = FlatTable("FT1", df1, "FT1", ["NUM_ENQ", "EXE_SOI_DTD"], {})
        ft2 = FlatTable("FT2", df2, "FT2", ["NUM_ENQ", "EXE_SOI_DTD"], {})
        ft3 = FlatTable("FT3", df3, "FT3", ["NUM_ENQ", "EXE_SOI_DTF"], {})
        ft = ft1.difference(ft2)
        df4 = self.spark.createDataFrame(
            pd.DataFrame(
                OrderedDict([("NUM_ENQ", ["1"]), ("EXE_SOI_DTD", ["01/01/2015"])])
            )
        )
        expected_ft = FlatTable("result", df4, "result", ["NUM_ENQ", "EXE_SOI_DTD"], {})
        self.assertEqual(expected_ft, ft)
        self.assertRaises(ValueError, ft1.difference, ft3)

    def test_union_all(self):
        df1 = self.spark.createDataFrame(
            pd.DataFrame({"NUM_ENQ": ["1"], "EXE_SOI_DTD": ["01/01/2015"]})
        )
        df2 = self.spark.createDataFrame(
            pd.DataFrame({"NUM_ENQ": ["2"], "EXE_SOI_DTD": ["01/02/2016"]})
        )
        df3 = self.spark.createDataFrame(
            pd.DataFrame({"NUM_ENQ": ["3"], "EXE_SOI_DTD": ["01/03/2017"]})
        )
        single_1 = SingleTable("ST1", df1, "ST1")
        single_2 = SingleTable("ST2", df2, "ST2")
        ft1 = FlatTable(
            "FT1", df1, "FT1", ["NUM_ENQ", "EXE_SOI_DTD"], {"ST1": single_1}
        )
        ft2 = FlatTable(
            "FT2", df2, "FT2", ["NUM_ENQ", "EXE_SOI_DTD"], {"ST2": single_2}
        )
        ft3 = FlatTable("FT3", df3, "FT3", ["NUM_ENQ", "EXE_SOI_DTD"], {})
        ft = FlatTable.union_all([ft1, ft2, ft3])
        df4 = self.spark.createDataFrame(
            pd.DataFrame(
                {
                    "NUM_ENQ": ["1", "2", "3"],
                    "EXE_SOI_DTD": ["01/01/2015", "01/02/2016", "01/03/2017"],
                }
            )
        )
        expected_ft = FlatTable(
            "result",
            df4,
            "result",
            ["NUM_ENQ", "EXE_SOI_DTD"],
            {"ST1": single_1, "ST2": single_2},
        )
        self.assertEqual(expected_ft, ft)

    def test_intersection_all(self):
        df1 = self.spark.createDataFrame(
            pd.DataFrame(
                {"NUM_ENQ": ["1", "2"], "EXE_SOI_DTD": ["01/01/2015", "01/02/2016"]}
            )
        )
        df2 = self.spark.createDataFrame(
            pd.DataFrame(
                {"NUM_ENQ": ["3", "2"], "EXE_SOI_DTD": ["01/03/2017", "01/02/2016"]}
            )
        )
        df3 = self.spark.createDataFrame(
            pd.DataFrame(
                {
                    "NUM_ENQ": ["1", "2", "3"],
                    "EXE_SOI_DTD": ["01/01/2015", "01/02/2016", "01/03/2017"],
                }
            )
        )
        ft1 = FlatTable("FT1", df1, "FT1", ["NUM_ENQ", "EXE_SOI_DTD"], {})
        ft2 = FlatTable("FT2", df2, "FT2", ["NUM_ENQ", "EXE_SOI_DTD"], {})
        ft3 = FlatTable("FT3", df3, "FT3", ["NUM_ENQ", "EXE_SOI_DTD"], {})
        ft = FlatTable.intersection_all([ft1, ft2, ft3], ["NUM_ENQ", "EXE_SOI_DTD"])
        df4 = self.spark.createDataFrame(
            pd.DataFrame(
                OrderedDict([("NUM_ENQ", ["2"]), ("EXE_SOI_DTD", ["01/02/2016"])])
            )
        )
        expected_ft = FlatTable("result", df4, "result", ["NUM_ENQ", "EXE_SOI_DTD"], {})
        self.assertEqual(expected_ft, ft)

    def test_different_all(self):
        df1 = self.spark.createDataFrame(
            pd.DataFrame(
                {
                    "NUM_ENQ": ["1", "2", "3"],
                    "EXE_SOI_DTD": ["01/01/2015", "01/02/2016", "01/03/2017"],
                }
            )
        )
        df2 = self.spark.createDataFrame(
            pd.DataFrame({"NUM_ENQ": ["1"], "EXE_SOI_DTD": ["01/01/2015"]})
        )
        df3 = self.spark.createDataFrame(
            pd.DataFrame({"NUM_ENQ": ["2"], "EXE_SOI_DTD": ["01/02/2016"]})
        )
        ft1 = FlatTable("FT1", df1, "FT1", ["NUM_ENQ", "EXE_SOI_DTD"], {})
        ft2 = FlatTable("FT2", df2, "FT2", ["NUM_ENQ", "EXE_SOI_DTD"], {})
        ft3 = FlatTable("FT3", df3, "FT3", ["NUM_ENQ", "EXE_SOI_DTD"], {})
        ft = FlatTable.difference_all([ft1, ft2, ft3], ["NUM_ENQ", "EXE_SOI_DTD"])
        df4 = self.spark.createDataFrame(
            pd.DataFrame(
                OrderedDict([("NUM_ENQ", ["3"]), ("EXE_SOI_DTD", ["01/03/2017"])])
            )
        )
        expected_ft = FlatTable("result", df4, "result", ["NUM_ENQ", "EXE_SOI_DTD"], {})
        self.assertEqual(expected_ft, ft)

    def test_build_history_table(self):
        df1 = self.spark.createDataFrame(
            pd.DataFrame(
                {"NUM_ENQ": ["1", "2"], "EXE_SOI_DTD": ["01/01/2015", "01/01/2016"]}
            )
        )

        df2 = self.spark.createDataFrame(
            pd.DataFrame(
                {
                    "NUM_ENQ": ["1", "2"],
                    "EXE_SOI_DTD": ["01/01/2015", "01/01/2016"],
                    "history": ["2018", "2018"],
                }
            )
        )

        df3 = self.spark.createDataFrame(
            pd.DataFrame(
                OrderedDict(
                    [
                        ("NUM_ENQ", ["1", "2", "1", "2", "1", "2"]),
                        (
                            "EXE_SOI_DTD",
                            [
                                "01/01/2015",
                                "01/01/2016",
                                "01/01/2015",
                                "01/01/2016",
                                "01/01/2015",
                                "01/01/2016",
                            ],
                        ),
                        ("history", ["2016", "2016", "2017", "2017", "2018", "2018"]),
                    ]
                )
            )
        )

        data = {"2016": df1, "2017": df1, "2018": df2}

        table = HistoryTable.build("2016 2017 2018", "2016 2017 2018", data)
        expected = HistoryTable("result", df3, "result")

        self.assertEqual(expected, table)
