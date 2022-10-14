import pytest
import numpy as np
import matplotlib.pyplot as plt
from astropy.utils import NumpyRNGContext
from astropy.wcs import WCS

from glue.core import Data, DataCollection
from glue.app.qt.application import GlueApplication
from glue.viewers.image.qt import ImageViewer
from glue.viewers.matplotlib.qt.tests.test_python_export import BaseTestExportPython


class TestExportPython(BaseTestExportPython):

    def setup_method(self, method):

        with NumpyRNGContext(12345):
            self.data = Data(cube=np.random.random((30, 50, 20)))
        # Create data version with WCS coordinates
        self.data_wcs = Data(label='cube', cube=self.data['cube'], coords=WCS(naxis=3))
        self.data_collection = DataCollection([self.data, self.data_wcs])
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

    def assert_same(self, tmpdir, tol=0.1):
        BaseTestExportPython.assert_same(self, tmpdir, tol=tol)

    @pytest.mark.parametrize('wcs', [False, True])
    def test_simple(self, tmpdir, wcs):
        if wcs:
            self.viewer = self.app.new_data_viewer(ImageViewer)
            self.viewer.add_data(self.data_wcs)
        self.assert_same(tmpdir)

    @pytest.mark.parametrize('wcs', [False, True])
    def test_simple_legend(self, tmpdir, wcs):
        if wcs:
            self.viewer = self.app.new_data_viewer(ImageViewer)
            self.viewer.add_data(self.data_wcs)
        self.viewer.state.show_legend = True
        self.assert_same(tmpdir)

    @pytest.mark.parametrize('wcs', [False, True])
    def test_simple_att(self, tmpdir, wcs):
        if wcs:
            self.viewer = self.app.new_data_viewer(ImageViewer)
            self.viewer.add_data(self.data_wcs)
        self.viewer.state.x_att = self.data.pixel_component_ids[1]
        self.viewer.state.y_att = self.data.pixel_component_ids[0]
        self.assert_same(tmpdir)

    @pytest.mark.parametrize('wcs', [False, True])
    def test_simple_visual(self, tmpdir, wcs):
        if wcs:
            self.viewer = self.app.new_data_viewer(ImageViewer)
            self.viewer.add_data(self.data_wcs)
        self.viewer.state.legend.visible = True
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

    def test_subset_legend(self, tmpdir):
        self.viewer.state.legend.visible = True
        self.data_collection.new_subset_group('mysubset', self.data.id['cube'] > 0.5)
        self.assert_same(tmpdir, tol=0.15)  # transparency and such

    def test_subset_slice(self, tmpdir):
        self.data_collection.new_subset_group('mysubset', self.data.id['cube'] > 0.5)
        self.test_slice(tmpdir)

    def test_subset_transposed(self, tmpdir):
        self.viewer.state.x_att = self.data.pixel_component_ids[0]
        self.viewer.state.y_att = self.data.pixel_component_ids[1]
        self.data_collection.new_subset_group('mysubset', self.data.id['cube'] > 0.5)
        self.assert_same(tmpdir)
