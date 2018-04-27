from __future__ import absolute_import, division, print_function

from astropy.utils import NumpyRNGContext

from glue.core import Data, DataCollection
from glue.app.qt.application import GlueApplication
from glue.viewers.histogram.qt import HistogramViewer
from glue.viewers.matplotlib.qt.tests.test_python_export import BaseTestExportPython, random_with_nan


class TestExportPython(BaseTestExportPython):

    def setup_method(self, method):

        with NumpyRNGContext(12345):
            self.data = Data(**dict((name, random_with_nan(100, nan_index=idx + 1)) for idx, name in enumerate('abcdefgh')))
        self.data_collection = DataCollection([self.data])
        self.app = GlueApplication(self.data_collection)
        self.viewer = self.app.new_data_viewer(HistogramViewer)
        self.viewer.add_data(self.data)
        self.viewer.state.x_att = self.data.id['a']

    def teardown_method(self, method):
        self.viewer.close()
        self.viewer = None
        self.app.close()
        self.app = None

    def test_simple(self, tmpdir):
        self.assert_same(tmpdir)

    def test_simple_visual(self, tmpdir):
        self.viewer.state.layers[0].color = 'blue'
        self.viewer.state.layers[0].alpha = 0.5
        self.assert_same(tmpdir)

    def test_cumulative(self, tmpdir):
        self.viewer.state.cumulative = True
        self.assert_same(tmpdir)

    def test_normalize(self, tmpdir):
        self.viewer.state.normalize = True
        self.assert_same(tmpdir)

    def test_subset(self, tmpdir):
        self.data_collection.new_subset_group('mysubset', self.data.id['a'] > 0.5)
        self.assert_same(tmpdir)

    def test_empty(self, tmpdir):
        self.viewer.state.x_min = 10
        self.viewer.state.x_max = 11
        self.viewer.state.hist_x_min = 10
        self.viewer.state.hist_x_max = 11
        self.assert_same(tmpdir)
