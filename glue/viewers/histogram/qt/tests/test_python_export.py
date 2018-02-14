import os
import pytest

import numpy as np

from glue.core import Data, DataCollection
from glue.app.qt.application import GlueApplication
from glue.viewers.histogram.qt import HistogramViewer
from matplotlib.testing.compare import compare_images


def random_with_nan(nsamples, nan_index):
    x = np.random.random(nsamples)
    x[nan_index] = np.nan
    return x


class TestExportPython:

    def setup_method(self, method):

        self.data = Data(**dict((name, random_with_nan(100, nan_index=idx + 1)) for idx, name in enumerate('abcdefgh')))
        self.data_collection = DataCollection([self.data])
        ga = GlueApplication(self.data_collection)
        self.scatter = ga.new_data_viewer(HistogramViewer)
        self.scatter.add_data(self.data)
        self.scatter.state.x_att = self.data.id['a']

    def test_simple(self, tmpdir):
        self.assert_same(tmpdir)

    def test_simple_visual(self, tmpdir):
        self.scatter.state.layers[0].color = 'blue'
        self.scatter.state.layers[0].alpha = 0.5
        self.assert_same(tmpdir)

    def test_cumulative(self, tmpdir):
        self.scatter.state.cumulative = True
        self.assert_same(tmpdir)

    def test_normalize(self, tmpdir):
        self.scatter.state.normalize = True
        self.assert_same(tmpdir)

    def test_subset(self, tmpdir):
        self.data_collection.new_subset_group('mysubset', self.data.id['a'] > 0.5)
        self.assert_same(tmpdir)

    def test_empty(self, tmpdir):
        self.scatter.state.x_min = 10
        self.scatter.state.x_max = 11
        self.scatter.state.hist_x_min = 10
        self.scatter.state.hist_x_max = 11
        self.assert_same(tmpdir)

    def assert_same(self, tmpdir, tol=0.1):

        os.chdir(tmpdir.strpath)

        expected = tmpdir.join('expected.png').strpath
        script = tmpdir.join('actual.py').strpath
        actual = tmpdir.join('myplot.png').strpath

        self.scatter.axes.figure.savefig(expected)

        self.scatter.export_as_script(script)
        with open(script) as f:
            exec(f.read())

        msg = compare_images(expected, actual, tol=tol)
        if msg:

            from base64 import b64encode

            print("SCRIPT:")
            with open(script, 'r') as f:
                print(f.read())

            print("EXPECTED:")
            with open(expected, 'rb') as f:
                print(b64encode(f.read()).decode())

            print("ACTUAL:")
            with open(actual, 'rb') as f:
                print(b64encode(f.read()).decode())

            pytest.fail(msg, pytrace=False)
