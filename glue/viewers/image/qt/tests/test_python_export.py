import pytest
import numpy as np
import matplotlib.pyplot as plt
from astropy.utils import NumpyRNGContext
from astropy.wcs import WCS

from glue.core import Data, DataCollection
from glue.core.coordinates import AffineCoordinates
from glue.app.qt.application import GlueApplication
from glue.viewers.image.qt import ImageViewer
from glue.viewers.matplotlib.qt.tests.test_python_export import BaseTestExportPython


class TestExportPython(BaseTestExportPython):

    def setup_method(self, method):

        with NumpyRNGContext(12345):
            self.data = Data(cube=np.random.random((30, 50, 20)))
        # Create data versions with WCS and affine coordinates
        matrix = np.array([[2, 3, 4, -1], [1, 2, 2, 2], [1, 1, 1, -2], [0, 0, 0, 1]])
        affine = AffineCoordinates(matrix, units=['Mm', 'Mm', 'km'], labels=['xw', 'yw', 'zw'])

        self.data_wcs = Data(label='cube', cube=self.data['cube'], coords=WCS(naxis=3))
        self.data_affine = Data(label='cube', cube=self.data['cube'], coords=affine)
        self.data_collection = DataCollection([self.data, self.data_wcs, self.data_affine])
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

    def viewer_load(self, coords):
        if coords is not None:
            self.viewer.add_data(getattr(self, f'data_{coords}'))
            self.viewer.remove_data(self.data)

    @pytest.mark.parametrize('coords', [None, 'wcs', 'affine'])
    def test_simple(self, tmpdir, coords):
        self.viewer_load(coords)
        self.assert_same(tmpdir)

    @pytest.mark.parametrize('coords', [None, 'wcs', 'affine'])
    def test_simple_legend(self, tmpdir, coords):
        self.viewer_load(coords)
        self.viewer.state.show_legend = True
        self.assert_same(tmpdir)

    @pytest.mark.parametrize('coords', [None, 'wcs', 'affine'])
    def test_simple_att(self, tmpdir, coords):
        self.viewer_load(coords)
        self.viewer.state.x_att = self.viewer.state.reference_data.pixel_component_ids[1]
        self.viewer.state.y_att = self.viewer.state.reference_data.pixel_component_ids[0]
        if coords == 'affine':
            pytest.xfail('Known issue with axis label rendering')
        self.assert_same(tmpdir)

    @pytest.mark.parametrize('coords', [None, 'wcs', 'affine'])
    def test_simple_visual(self, tmpdir, coords):
        self.viewer_load(coords)
        self.viewer.state.legend.visible = True
        self.viewer.state.layers[0].cmap = plt.cm.RdBu
        self.viewer.state.layers[0].v_min = 0.2
        self.viewer.state.layers[0].v_max = 0.8
        self.viewer.state.layers[0].stretch = 'sqrt'
        self.viewer.state.layers[0].stretch = 'sqrt'
        self.viewer.state.layers[0].contrast = 0.9
        self.viewer.state.layers[0].bias = 0.6
        self.assert_same(tmpdir)

    @pytest.mark.parametrize('coords', [None, 'wcs', 'affine'])
    def test_slice(self, tmpdir, coords):
        self.viewer_load(coords)
        self.viewer.state.x_att = self.viewer.state.reference_data.pixel_component_ids[1]
        self.viewer.state.y_att = self.viewer.state.reference_data.pixel_component_ids[0]
        self.viewer.state.slices = (2, 3, 4)
        if coords == 'affine':
            pytest.xfail('Known issue with axis label rendering')
        self.assert_same(tmpdir)

    @pytest.mark.parametrize('coords', [None, 'wcs', 'affine'])
    def test_aspect(self, tmpdir, coords):
        self.viewer_load(coords)
        self.viewer.state.aspect = 'auto'
        self.assert_same(tmpdir)

    @pytest.mark.parametrize('coords', [None, 'wcs', 'affine'])
    def test_subset(self, tmpdir, coords):
        self.viewer_load(coords)
        self.data_collection.new_subset_group('mysubset', self.data.id['cube'] > 0.5)
        self.assert_same(tmpdir)

    @pytest.mark.parametrize('coords', [None, 'wcs', 'affine'])
    def test_subset_legend(self, tmpdir, coords):
        self.viewer_load(coords)
        self.data_collection.new_subset_group('mysubset',
                                              self.viewer.state.reference_data.id['cube'] > 0.5)
        self.viewer.state.legend.visible = True
        self.assert_same(tmpdir, tol=0.15)  # transparency and such

    @pytest.mark.parametrize('coords', [None, 'wcs', 'affine'])
    def test_subset_slice(self, tmpdir, coords):
        self.viewer_load(coords)
        self.data_collection.new_subset_group('mysubset', self.data.id['cube'] > 0.5)
        self.test_slice(tmpdir, coords)

    @pytest.mark.parametrize('coords', [None, 'wcs', 'affine'])
    def test_subset_transposed(self, tmpdir, coords):
        self.viewer_load(coords)
        self.data_collection.new_subset_group('mysubset', self.data.id['cube'] > 0.5)
        self.viewer.state.x_att = self.data.pixel_component_ids[0]
        self.viewer.state.y_att = self.data.pixel_component_ids[1]
        self.assert_same(tmpdir)
