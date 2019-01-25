from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pyplot as plt
from astropy.utils import NumpyRNGContext

from glue.core import Data, DataCollection
from glue.app.qt.application import GlueApplication
from glue.viewers.image.qt import ImageViewer
from glue.viewers.matplotlib.qt.tests.test_python_export import BaseTestExportPython


class TestExportPython(BaseTestExportPython):

    def setup_method(self, method):

        with NumpyRNGContext(12345):
            self.data = Data(cube=np.random.random((30, 50, 20)))
        self.data_collection = DataCollection([self.data])
        self.app = GlueApplication(self.data_collection)
        self.viewer = self.app.new_data_viewer(ImageViewer)
        self.viewer.add_data(self.data)
        # FIXME: On some platforms, using an integer label size
        # causes some of the labels to be non-deterministically
        # shifted by one pixel, so we pick a non-round font size
        # to avoid this.
        self.viewer.state.x_ticklabel_size = 8.21334111
        self.viewer.state.y_ticklabel_size = 8.21334111

    def teardown_method(self, method):
        self.viewer.close()
        self.viewer = None
        self.app.close()
        self.app = None

    def test_simple(self, tmpdir):
        self.assert_same(tmpdir)

    def test_simple_att(self, tmpdir):
        self.viewer.state.x_att = self.data.pixel_component_ids[1]
        self.viewer.state.y_att = self.data.pixel_component_ids[0]
        self.assert_same(tmpdir)

    def test_simple_visual(self, tmpdir):
        self.viewer.state.layers[0].cmap = plt.cm.RdBu
        self.viewer.state.layers[0].v_min = 0.2
        self.viewer.state.layers[0].v_max = 0.8
        self.viewer.state.layers[0].stretch = 'sqrt'
        self.viewer.state.layers[0].stretch = 'sqrt'
        self.viewer.state.layers[0].contrast = 0.9
        self.viewer.state.layers[0].bias = 0.6
        self.assert_same(tmpdir)

    def test_slice(self, tmpdir):
        self.viewer.state.x_att = self.data.pixel_component_ids[1]
        self.viewer.state.y_att = self.data.pixel_component_ids[0]
        self.viewer.state.slices = (2, 3, 4)
        self.assert_same(tmpdir)

    def test_aspect(self, tmpdir):
        self.viewer.state.aspect = 'auto'
        self.assert_same(tmpdir)

    def test_subset(self, tmpdir):
        self.data_collection.new_subset_group('mysubset', self.data.id['cube'] > 0.5)
        self.assert_same(tmpdir)

    def test_subset_slice(self, tmpdir):
        self.data_collection.new_subset_group('mysubset', self.data.id['cube'] > 0.5)
        self.test_slice(tmpdir)
