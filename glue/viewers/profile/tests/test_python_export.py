from astropy.utils import NumpyRNGContext
from astropy.wcs import WCS
import numpy as np
import pytest
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose

from glue.core import Data, DataCollection
from glue.core.application_base import Application
from glue.viewers.profile.viewer import SimpleProfileViewer
from glue.viewers.matplotlib.tests.test_python_export import BaseTestExportPython, random_with_nan
from glue.viewers.profile.tests.test_state import SimpleCoordinates
from glue.viewers.profile.python_export import python_export_profile_layer


@pytest.mark.parametrize('subset', [False, True])
def test_slice_world_coordinate_export(subset, request):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['WAVE', 'LINEAR']
    wcs.wcs.cunit = ['m', '']
    wcs.wcs.crpix = [1, 1]
    wcs.wcs.crval = [1, 0]
    wcs.wcs.pc = [[1, 10], [0, 1]]
    data = Data(flux=np.arange(12).reshape(3, 4), coords=wcs)
    app = Application(DataCollection([data]))
    viewer = app.new_data_viewer(SimpleProfileViewer)
    request.addfinalizer(lambda: plt.close(viewer.figure))
    viewer.add_data(data)
    viewer.state.x_att = data.world_component_ids[1]
    viewer.state.function = 'slice'
    viewer.state.slices = (2, 0)
    if subset:
        app.data_collection.new_subset_group('selected', data.id['flux'] > 8)
    artist = viewer.layers[-1]
    imports, script = python_export_profile_layer(artist)
    namespace = dict(layer_data=artist.state.layer, ax=viewer.axes,
                     legend_handles=[], legend_labels=[])
    exec('\n'.join(imports) + '\n' + script, namespace)  # noqa: S102 - test the trusted exporter output
    assert_allclose(namespace['profile_x_values'], [21, 22, 23, 24])
    assert_allclose(namespace['profile_values'], [np.nan if subset else 8, 9, 10, 11])


class TestExportPython(BaseTestExportPython):

    def setup_method(self, method):

        self.data = Data(label='d1')
        self.data.coords = SimpleCoordinates()
        with NumpyRNGContext(12345):
            self.data['x'] = random_with_nan(48, 5).reshape((6, 4, 2))
            self.data['y'] = random_with_nan(48, 12).reshape((6, 4, 2))
        self.data_collection = DataCollection([self.data])
        self.app = Application(self.data_collection)
        self.viewer = self.app.new_data_viewer(SimpleProfileViewer)
        self.viewer.add_data(self.data)
        # Make legend location deterministic
        self.viewer.state.legend.location = 'lower left'

    def teardown_method(self, method):
        self.viewer = None
        self.app = None

    def test_simple(self, tmpdir):
        self.assert_same(tmpdir)

    def test_simple_legend(self, tmpdir):
        self.viewer.state.legend.visible = True
        self.assert_same(tmpdir)

    def test_color(self, tmpdir):
        self.viewer.state.layers[0].color = '#ac0567'
        self.assert_same(tmpdir)

    def test_linewidth(self, tmpdir):
        self.viewer.state.layers[0].linewidth = 7.25
        self.assert_same(tmpdir)

    def test_max(self, tmpdir):
        self.viewer.state.function = 'maximum'
        self.assert_same(tmpdir)

    def test_min(self, tmpdir):
        self.viewer.state.function = 'minimum'
        self.assert_same(tmpdir)

    def test_mean(self, tmpdir):
        self.viewer.state.function = 'mean'
        self.assert_same(tmpdir)

    def test_median(self, tmpdir):
        self.viewer.state.function = 'median'
        self.assert_same(tmpdir)

    def test_sum(self, tmpdir):
        self.viewer.state.function = 'sum'
        self.assert_same(tmpdir)

    def test_slice(self, tmpdir):
        self.viewer.state.function = 'slice'
        self.viewer.state.slices = (0, 2, 1)
        self.assert_same(tmpdir)

    def test_slice_subset(self, tmpdir):
        # The subset mask is deliberately partial along the profile axis
        self.viewer.state.function = 'slice'
        self.data_collection.new_subset_group('mysubset', self.data.pixel_component_ids[0] > 0.5)
        self.assert_same(tmpdir)

    def test_normalization(self, tmpdir):
        self.viewer.state.normalize = True
        self.assert_same(tmpdir)

    def test_subset(self, tmpdir):
        self.viewer.state.function = 'mean'
        self.data_collection.new_subset_group('mysubset', self.data.id['x'] > 0.25)
        self.assert_same(tmpdir)

    def test_subset_legend(self, tmpdir):
        self.viewer.state.legend.visible = True
        self.viewer.state.function = 'mean'
        self.viewer.state.layers[0].linewidth = 7.25
        self.data_collection.new_subset_group('mysubset', self.data.id['x'] > 0.25)
        self.assert_same(tmpdir)

    def test_xatt(self, tmpdir):
        self.viewer.x_att = self.data.pixel_component_ids[1]
        self.assert_same(tmpdir)

    def test_profile_att(self, tmpdir):
        self.viewer.layers[0].state.attribute = self.data.id['y']
        self.assert_same(tmpdir)


class TestExportPythonWCSAxes(TestExportPython):

    # Run all the export tests again with a WCSAxes-based viewer, where the
    # profile is plotted in pixel coordinates and the WCS formats the ticks

    def setup_method(self, method):

        self.data = Data(label='d1')
        self.data.coords = SimpleCoordinates()
        with NumpyRNGContext(12345):
            self.data['x'] = random_with_nan(48, 5).reshape((6, 4, 2))
            self.data['y'] = random_with_nan(48, 12).reshape((6, 4, 2))
        self.data_collection = DataCollection([self.data])
        self.app = Application(self.data_collection)
        self.viewer = SimpleProfileViewer(self.app.session, wcs=True)
        self.viewer.register_to_hub(self.app.session.hub)
        self.viewer.add_data(self.data)
        # Make legend location deterministic
        self.viewer.state.legend.location = 'lower left'

        assert self.viewer.state.wcsaxes_active
