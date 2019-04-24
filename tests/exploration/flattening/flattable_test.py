from collections import OrderedDict

import pandas as pd

from src.exploration.flattening.flattable import FlatTable
from tests.exploration.core.pyspark_tests import PySparkTest


class TestFlatTable(PySparkTest):
    def test_eq(self):
        df1 = self.spark.createDataFrame(
            pd.DataFrame({'NUM_ENQ': ['1', '2', '3'], 'EXE_SOI_DTD': ['01/01/2015', '01/02/2016', '01/03/2017']}))
        df2 = self.spark.createDataFrame(
            pd.DataFrame({'NUM_ENQ': ['3', '2', '1'], 'EXE_SOI_DTD': ['01/03/2017', '01/02/2016', '01/01/2015']}))
        ft1 = FlatTable('FT1', df1, 'FT1', ['NUM_ENQ', 'EXE_SOI_DTD'])
        ft2 = FlatTable('FT2', df2, 'FT2', ['NUM_ENQ', 'EXE_SOI_DTD'])
        self.assertEqual(ft1, ft2)

    def test_in(self):
        df1 = self.spark.createDataFrame(
            pd.DataFrame({'NUM_ENQ': ['1', '2', '3'], 'EXE_SOI_DTD': ['01/01/2015', '01/02/2016', '01/03/2017']}))
        df2 = self.spark.createDataFrame(
            pd.DataFrame({'NUM_ENQ': ['3', '2'], 'EXE_SOI_DTD': ['01/03/2017', '01/02/2016']}))
        ft1 = FlatTable('FT1', df1, 'FT1', ['NUM_ENQ', 'EXE_SOI_DTD'])
        ft2 = FlatTable('FT2', df2, 'FT2', ['NUM_ENQ', 'EXE_SOI_DTD'])
        self.assertIn(ft2, ft1)

    def test_union(self):
        df1 = self.spark.createDataFrame(
            pd.DataFrame({'NUM_ENQ': ['1', '2', '3'], 'EXE_SOI_DTD': ['01/01/2015', '01/02/2016', '01/03/2017']}))
        df2 = self.spark.createDataFrame(
            pd.DataFrame({'NUM_ENQ': ['3', '2'], 'EXE_SOI_DTD': ['01/03/2017', '01/02/2016']}))
        ft1 = FlatTable('FT1', df1, 'FT1', ['NUM_ENQ', 'EXE_SOI_DTD'])
        ft2 = FlatTable('FT2', df2, 'FT2', ['NUM_ENQ', 'EXE_SOI_DTD'])
        ft = ft1.union(ft2)
        df3 = self.spark.createDataFrame(
            pd.DataFrame({'NUM_ENQ': ['1', '2', '3'], 'EXE_SOI_DTD': ['01/01/2015', '01/02/2016', '01/03/2017']}))
        expected_ft = FlatTable('result', df3, 'result', ['NUM_ENQ', 'EXE_SOI_DTD'])
        self.assertEqual(expected_ft, ft)

    def test_intersection(self):
        df1 = self.spark.createDataFrame(
            pd.DataFrame({'NUM_ENQ': ['1', '2'], 'EXE_SOI_DTD': ['01/01/2015', '01/02/2016']}))
        df2 = self.spark.createDataFrame(
            pd.DataFrame({'NUM_ENQ': ['3', '2'], 'EXE_SOI_DTD': ['01/03/2017', '01/02/2016']}))
        ft1 = FlatTable('FT1', df1, 'FT1', ['NUM_ENQ', 'EXE_SOI_DTD'])
        ft2 = FlatTable('FT2', df2, 'FT2', ['NUM_ENQ', 'EXE_SOI_DTD'])
        ft = ft1.intersection(ft2)
        df3 = self.spark.createDataFrame(
            pd.DataFrame(OrderedDict([('NUM_ENQ', ['2']), ('EXE_SOI_DTD', ['01/02/2016'])])))
        expected_ft = FlatTable('result', df3, 'result', ['NUM_ENQ', 'EXE_SOI_DTD'])
        self.assertEqual(expected_ft, ft)

    def test_difference(self):
        df1 = self.spark.createDataFrame(
            pd.DataFrame({'NUM_ENQ': ['1', '2'], 'EXE_SOI_DTD': ['01/01/2015', '01/02/2016']}))
        df2 = self.spark.createDataFrame(
            pd.DataFrame({'NUM_ENQ': ['3', '2'], 'EXE_SOI_DTD': ['01/03/2017', '01/02/2016']}))
        ft1 = FlatTable('FT1', df1, 'FT1', ['NUM_ENQ', 'EXE_SOI_DTD'])
        ft2 = FlatTable('FT2', df2, 'FT2', ['NUM_ENQ', 'EXE_SOI_DTD'])
        ft = ft1.difference(ft2)
        df3 = self.spark.createDataFrame(
            pd.DataFrame(OrderedDict([('NUM_ENQ', ['1']), ('EXE_SOI_DTD', ['01/01/2015'])])))
        expected_ft = FlatTable('result', df3, 'result', ['NUM_ENQ', 'EXE_SOI_DTD'])
        self.assertEqual(expected_ft, ft)

    def test_union_all(self):
        df1 = self.spark.createDataFrame(pd.DataFrame({'NUM_ENQ': ['1'], 'EXE_SOI_DTD': ['01/01/2015']}))
        df2 = self.spark.createDataFrame(pd.DataFrame({'NUM_ENQ': ['2'], 'EXE_SOI_DTD': ['01/02/2016']}))
        df3 = self.spark.createDataFrame(pd.DataFrame({'NUM_ENQ': ['3'], 'EXE_SOI_DTD': ['01/03/2017']}))
        ft1 = FlatTable('FT1', df1, 'FT1', ['NUM_ENQ', 'EXE_SOI_DTD'])
        ft2 = FlatTable('FT2', df2, 'FT2', ['NUM_ENQ', 'EXE_SOI_DTD'])
        ft3 = FlatTable('FT3', df3, 'FT3', ['NUM_ENQ', 'EXE_SOI_DTD'])
        ft = FlatTable.union_all([ft1, ft2, ft3])
        df4 = self.spark.createDataFrame(
            pd.DataFrame({'NUM_ENQ': ['1', '2', '3'], 'EXE_SOI_DTD': ['01/01/2015', '01/02/2016', '01/03/2017']}))
        expected_ft = FlatTable('result', df4, 'result', ['NUM_ENQ', 'EXE_SOI_DTD'])
        self.assertEqual(expected_ft, ft)

    def test_intersection_all(self):
        df1 = self.spark.createDataFrame(
            pd.DataFrame({'NUM_ENQ': ['1', '2'], 'EXE_SOI_DTD': ['01/01/2015', '01/02/2016']}))
        df2 = self.spark.createDataFrame(
            pd.DataFrame({'NUM_ENQ': ['3', '2'], 'EXE_SOI_DTD': ['01/03/2017', '01/02/2016']}))
        df3 = self.spark.createDataFrame(
            pd.DataFrame({'NUM_ENQ': ['1', '2', '3'], 'EXE_SOI_DTD': ['01/01/2015', '01/02/2016', '01/03/2017']}))
        ft1 = FlatTable('FT1', df1, 'FT1', ['NUM_ENQ', 'EXE_SOI_DTD'])
        ft2 = FlatTable('FT2', df2, 'FT2', ['NUM_ENQ', 'EXE_SOI_DTD'])
        ft3 = FlatTable('FT3', df3, 'FT3', ['NUM_ENQ', 'EXE_SOI_DTD'])
        ft = FlatTable.intersection_all([ft1, ft2, ft3], ['NUM_ENQ', 'EXE_SOI_DTD'])
        df4 = self.spark.createDataFrame(
            pd.DataFrame(OrderedDict([('NUM_ENQ', ['2']), ('EXE_SOI_DTD', ['01/02/2016'])])))
        expected_ft = FlatTable('result', df4, 'result', ['NUM_ENQ', 'EXE_SOI_DTD'])
        self.assertEqual(expected_ft, ft)

    def test_different_all(self):
        df1 = self.spark.createDataFrame(
            pd.DataFrame({'NUM_ENQ': ['1', '2', '3'], 'EXE_SOI_DTD': ['01/01/2015', '01/02/2016', '01/03/2017']}))
        df2 = self.spark.createDataFrame(pd.DataFrame({'NUM_ENQ': ['1'], 'EXE_SOI_DTD': ['01/01/2015']}))
        df3 = self.spark.createDataFrame(pd.DataFrame({'NUM_ENQ': ['2'], 'EXE_SOI_DTD': ['01/02/2016']}))
        ft1 = FlatTable('FT1', df1, 'FT1', ['NUM_ENQ', 'EXE_SOI_DTD'])
        ft2 = FlatTable('FT2', df2, 'FT2', ['NUM_ENQ', 'EXE_SOI_DTD'])
        ft3 = FlatTable('FT3', df3, 'FT3', ['NUM_ENQ', 'EXE_SOI_DTD'])
        ft = FlatTable.difference_all([ft1, ft2, ft3], ['NUM_ENQ', 'EXE_SOI_DTD'])
        df4 = self.spark.createDataFrame(
            pd.DataFrame(OrderedDict([('NUM_ENQ', ['3']), ('EXE_SOI_DTD', ['01/03/2017'])])))
        expected_ft = FlatTable('result', df4, 'result', ['NUM_ENQ', 'EXE_SOI_DTD'])
        self.assertEqual(expected_ft, ft)
