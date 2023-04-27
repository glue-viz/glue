# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

import os

import pytest
import numpy as np

from astropy import units as u
from astropy.wcs import WCS

from numpy.testing import assert_equal, assert_allclose

from glue.core import Data
from glue.core.roi import XRangeROI
from glue.core.subset import SliceSubsetState
from glue.app.qt import GlueApplication
from glue.core.component_link import ComponentLink
from glue.viewers.matplotlib.qt.tests.test_data_viewer import BaseTestMatplotlibDataViewer
from glue.core.coordinates import IdentityCoordinates
from glue.viewers.profile.tests.test_state import SimpleCoordinates
from glue.core.tests.test_state import clone
from glue.core.state import GlueUnSerializer
from glue.plugins.wcs_autolinking.wcs_autolinking import WCSLink
from glue.config import settings, unit_converter

from ..data_viewer import ProfileViewer

DATA = os.path.join(os.path.dirname(__file__), 'data')


def setup_function(func):
    func.ORIGINAL_UNIT_CONVERTER = settings.UNIT_CONVERTER


def teardown_function(func):
    settings.UNIT_CONVERTER = func.ORIGINAL_UNIT_CONVERTER


def teardown_module():
    unit_converter._members.pop('test-spectral2')
    print(unit_converter._members)


class TestProfileCommon(BaseTestMatplotlibDataViewer):

    def init_data(self):
        return Data(label='d1',
                    x=np.random.random(24).reshape((3, 4, 2)))

    viewer_cls = ProfileViewer

    @pytest.mark.skip()
    def test_double_add_ignored(self):
        pass


class TestProfileViewer(object):

    def setup_method(self, method):

        self.data = Data(label='d1')
        self.data.coords = SimpleCoordinates()
        self.data['x'] = np.arange(24).reshape((3, 4, 2))

        self.data2 = Data(label='d2')
        self.data2['y'] = np.arange(24).reshape((3, 4, 2))

        self.app = GlueApplication()
        self.session = self.app.session
        self.hub = self.session.hub

        self.data_collection = self.session.data_collection
        self.data_collection.append(self.data)
        self.data_collection.append(self.data2)

        self.viewer = self.app.new_data_viewer(ProfileViewer)

    def teardown_method(self, method):
        self.viewer.close()
        self.viewer = None
        self.app.close()
        self.app = None

    def test_functions(self):
        self.viewer.add_data(self.data)
        self.viewer.state.function = 'mean'
        assert len(self.viewer.layers) == 1
        layer_artist = self.viewer.layers[0]
        assert_allclose(layer_artist.state.profile[0], [0, 2, 4])
        assert_allclose(layer_artist.state.profile[1], [3.5, 11.5, 19.5])

    def test_incompatible(self):
        self.viewer.add_data(self.data)
        data2 = Data(y=np.random.random((3, 4, 2)))
        self.data_collection.append(data2)
        self.viewer.add_data(data2)
        assert len(self.viewer.layers) == 2
        assert self.viewer.layers[0].enabled
        assert not self.viewer.layers[1].enabled

    def test_selection(self):

        self.viewer.add_data(self.data)

        self.viewer.state.x_att = self.data.pixel_component_ids[0]

        roi = XRangeROI(0.9, 2.1)

        self.viewer.apply_roi(roi)

        assert len(self.data.subsets) == 1
        assert_equal(self.data.subsets[0].to_mask()[:, 0, 0], [0, 1, 1])

        self.viewer.state.x_att = self.data.world_component_ids[0]

        roi = XRangeROI(1.9, 3.1)

        self.viewer.apply_roi(roi)

        assert len(self.data.subsets) == 1
        assert_equal(self.data.subsets[0].to_mask()[:, 0, 0], [0, 1, 0])

    def test_enabled_layers(self):

        data2 = Data(label='d1', y=np.arange(24).reshape((3, 4, 2)),
                     coords=IdentityCoordinates(n_dim=3))
        self.data_collection.append(data2)

        self.viewer.add_data(self.data)
        self.viewer.add_data(data2)

        assert self.viewer.layers[0].enabled
        assert not self.viewer.layers[1].enabled

        self.data_collection.add_link(ComponentLink([data2.world_component_ids[1]], self.data.world_component_ids[0], using=lambda x: 2 * x))

        assert self.viewer.layers[0].enabled
        assert self.viewer.layers[1].enabled

    def test_slice_subset_state(self):

        self.viewer.add_data(self.data)

        subset = self.data.new_subset()
        subset.subset_state = SliceSubsetState(self.data, [slice(1, 2), slice(None)])

        assert self.viewer.layers[0].enabled
        assert self.viewer.layers[1].enabled

    def test_clone(self):

        # Regression test for a bug that meant that deserializing a profile
        # viewer resulted in disabled layers

        self.viewer.add_data(self.data)

        subset = self.data.new_subset()
        subset.subset_state = SliceSubsetState(self.data, [slice(1, 2), slice(None)])

        app = clone(self.app)

        assert app.viewers[0][0].layers[0].enabled
        assert app.viewers[0][0].layers[1].enabled

        app.close()

    def test_incompatible_on_add(self):

        # Regression test for a bug when adding a dataset to a profile viewer
        # with a single incompatible subset.

        subset_state = SliceSubsetState(self.data, [slice(1, 2), slice(None)])
        self.data_collection.new_subset_group(subset_state=subset_state, label='s1')

        data2 = Data(x=[[2, 3], [4, 3]], label='d2')
        self.data_collection.append(data2)
        self.viewer.add_data(data2)

    def test_dependent_axes(self):

        # Make sure that if we pick a world component that has correlations with
        # others and is not lined up with the pixel grid, a warning is shown.

        self.viewer.add_data(self.data)

        self.viewer.state.x_att = self.data.pixel_component_ids[0]
        assert self.viewer.options_widget().ui.text_warning.text() == ''

        self.viewer.state.x_att = self.data.pixel_component_ids[1]
        assert self.viewer.options_widget().ui.text_warning.text() == ''

        self.viewer.state.x_att = self.data.pixel_component_ids[2]
        assert self.viewer.options_widget().ui.text_warning.text() == ''

        self.viewer.state.x_att = self.data.world_component_ids[0]
        assert self.viewer.options_widget().ui.text_warning.text() == ''

        self.viewer.state.x_att = self.data.world_component_ids[1]
        assert self.viewer.options_widget().ui.text_warning.text() != ''

        self.viewer.state.x_att = self.data.world_component_ids[2]
        assert self.viewer.options_widget().ui.text_warning.text() != ''

    def test_multiple_data(self, tmpdir):

        # Regression test for issues when multiple datasets are present
        # and the reference data is not the default one.

        self.viewer.add_data(self.data)
        self.viewer.add_data(self.data2)
        assert self.viewer.layers[0].enabled
        assert not self.viewer.layers[1].enabled

        # Make sure that when changing the reference data, which layer
        # is enabled changes.
        self.viewer.state.reference_data = self.data2
        assert not self.viewer.layers[0].enabled
        assert self.viewer.layers[1].enabled

        # Make sure that everything works fine after saving/reloading
        filename = tmpdir.join('test_multiple_data.glu').strpath
        self.session.application.save_session(filename)
        with open(filename, 'r') as f:
            session = f.read()
        state = GlueUnSerializer.loads(session)
        ga = state.object('__main__')
        viewer = ga.viewers[0][0]
        assert not viewer.layers[0].enabled
        assert viewer.layers[1].enabled
        ga.close()

    @pytest.mark.parametrize('protocol', [1])
    def test_session_back_compat(self, protocol):

        filename = os.path.join(DATA, 'profile_v{0}.glu'.format(protocol))

        with open(filename, 'r') as f:
            session = f.read()

        state = GlueUnSerializer.loads(session)

        ga = state.object('__main__')

        dc = ga.session.data_collection

        assert len(dc) == 1

        assert dc[0].label == 'array'

        viewer1 = ga.viewers[0][0]
        assert len(viewer1.state.layers) == 3
        assert viewer1.state.x_att_pixel is dc[0].pixel_component_ids[0]
        assert_allclose(viewer1.state.x_min, -0.5)
        assert_allclose(viewer1.state.x_max, 2.5)
        assert_allclose(viewer1.state.y_min, 13)
        assert_allclose(viewer1.state.y_max, 63)
        assert viewer1.state.function == 'maximum'
        assert not viewer1.state.normalize
        assert viewer1.state.layers[0].visible
        assert viewer1.state.layers[1].visible
        assert viewer1.state.layers[2].visible

        viewer2 = ga.viewers[0][1]
        assert viewer2.state.x_att_pixel is dc[0].pixel_component_ids[1]
        assert_allclose(viewer2.state.x_min, -0.5)
        assert_allclose(viewer2.state.x_max, 3.5)
        assert_allclose(viewer2.state.y_min, -0.1)
        assert_allclose(viewer2.state.y_max, 1.1)
        assert viewer2.state.function == 'maximum'
        assert viewer2.state.normalize
        assert viewer2.state.layers[0].visible
        assert not viewer2.state.layers[1].visible
        assert viewer2.state.layers[2].visible

        viewer3 = ga.viewers[0][2]
        assert viewer3.state.x_att_pixel is dc[0].pixel_component_ids[2]
        assert_allclose(viewer3.state.x_min, -0.5)
        assert_allclose(viewer3.state.x_max, 4.5)
        assert_allclose(viewer3.state.y_min, -0.4)
        assert_allclose(viewer3.state.y_max, 4.4)
        assert viewer3.state.function == 'minimum'
        assert not viewer3.state.normalize
        assert viewer3.state.layers[0].visible
        assert viewer3.state.layers[1].visible
        assert not viewer3.state.layers[2].visible

        viewer4 = ga.viewers[0][3]
        assert viewer4.state.x_att_pixel is dc[0].pixel_component_ids[2]
        assert_allclose(viewer4.state.x_min, -5.5)
        assert_allclose(viewer4.state.x_max, 9.5)
        assert_allclose(viewer4.state.y_min, 27.1)
        assert_allclose(viewer4.state.y_max, 31.9)
        assert viewer4.state.function == 'mean'
        assert not viewer4.state.normalize
        assert viewer4.state.layers[0].visible
        assert not viewer4.state.layers[1].visible
        assert not viewer4.state.layers[2].visible

        ga.close()

    def test_reset_limits(self):
        self.viewer.add_data(self.data)
        self.viewer.add_data(self.data2)
        self.viewer.state.x_min = 0.2
        self.viewer.state.x_max = 0.4
        self.viewer.state.y_min = 0.3
        self.viewer.state.y_max = 0.5
        self.viewer.state.reset_limits()
        assert self.viewer.state.x_min == 0
        assert self.viewer.state.x_max == 4
        assert self.viewer.state.y_min == 7
        assert self.viewer.state.y_max == 23

    def test_limits_unchanged(self):
        # Make sure the limits don't change if a subset is created or another
        # dataset added - they should only change if the reference data is changed
        self.viewer.add_data(self.data)
        self.viewer.state.x_min = 0.2
        self.viewer.state.x_max = 0.4
        self.viewer.state.y_min = 0.3
        self.viewer.state.y_max = 0.5
        self.viewer.add_data(self.data2)
        assert self.viewer.state.x_min == 0.2
        assert self.viewer.state.x_max == 0.4
        assert self.viewer.state.y_min == 0.3
        assert self.viewer.state.y_max == 0.5
        roi = XRangeROI(0.9, 2.1)
        self.viewer.apply_roi(roi)
        assert self.viewer.state.x_min == 0.2
        assert self.viewer.state.x_max == 0.4
        assert self.viewer.state.y_min == 0.3
        assert self.viewer.state.y_max == 0.5

    def test_layer_visibility(self):
        self.viewer.add_data(self.data)
        assert self.viewer.layers[0].mpl_artists[0].get_visible() is True
        self.viewer.state.layers[0].visible = False
        assert self.viewer.layers[0].mpl_artists[0].get_visible() is False


@unit_converter('test-spectral2')
class SpectralUnitConverter:

    def equivalent_units(self, data, cid, units):
        return map(str, u.Unit(units).find_equivalent_units(include_prefix_units=True, equivalencies=u.spectral()))

    def to_unit(self, data, cid, values, original_units, target_units):
        return (values * u.Unit(original_units)).to_value(target_units, equivalencies=u.spectral())


def test_unit_conversion():

    settings.UNIT_CONVERTER = 'test-spectral2'

    wcs1 = WCS(naxis=1)
    wcs1.wcs.ctype = ['FREQ']
    wcs1.wcs.crval = [1]
    wcs1.wcs.cdelt = [1]
    wcs1.wcs.crpix = [1]
    wcs1.wcs.cunit = ['GHz']

    d1 = Data(f1=[1, 2, 3])
    d1.get_component('f1').units = 'Jy'
    d1.coords = wcs1

    wcs2 = WCS(naxis=1)
    wcs2.wcs.ctype = ['WAVE']
    wcs2.wcs.crval = [10]
    wcs2.wcs.cdelt = [10]
    wcs2.wcs.crpix = [1]
    wcs2.wcs.cunit = ['cm']

    d2 = Data(f2=[2000, 1000, 3000])
    d2.get_component('f2').units = 'mJy'
    d2.coords = wcs2

    app = GlueApplication()
    session = app.session

    data_collection = session.data_collection
    data_collection.append(d1)
    data_collection.append(d2)

    data_collection.add_link(WCSLink(d1, d2))

    viewer = app.new_data_viewer(ProfileViewer)
    viewer.add_data(d1)
    viewer.add_data(d2)

    assert viewer.layers[0].enabled
    assert viewer.layers[1].enabled

    x, y = viewer.state.layers[0].profile
    assert_allclose(x, [1.e9, 2.e9, 3.e9])
    assert_allclose(y, [1, 2, 3])

    x, y = viewer.state.layers[1].profile
    assert_allclose(x, 299792458 / np.array([0.1, 0.2, 0.3]))
    assert_allclose(y, [2000, 1000, 3000])

    assert viewer.state.x_min == 1.e9
    assert viewer.state.x_max == 3.e9
    assert viewer.state.y_min == 1.
    assert viewer.state.y_max == 3.

    # Change the limits to make sure they are always converted
    viewer.state.x_min = 5e8
    viewer.state.x_max = 4e9
    viewer.state.y_min = 0.5
    viewer.state.y_max = 3.5

    roi = XRangeROI(1.4e9, 2.1e9)
    viewer.apply_roi(roi)

    assert len(d1.subsets) == 1
    assert_equal(d1.subsets[0].to_mask(), [0, 1, 0])

    assert len(d2.subsets) == 1
    assert_equal(d2.subsets[0].to_mask(), [0, 1, 0])

    viewer.state.x_display_unit = 'GHz'
    viewer.state.y_display_unit = 'mJy'

    x, y = viewer.state.layers[0].profile
    assert_allclose(x, [1, 2, 3])
    assert_allclose(y, [1000, 2000, 3000])

    x, y = viewer.state.layers[1].profile
    assert_allclose(x, 2.99792458 / np.array([1, 2, 3]))
    assert_allclose(y, [2000, 1000, 3000])

    assert viewer.state.x_min == 0.5
    assert viewer.state.x_max == 4.

    # Units get reset because they were originally 'native' and 'native' to a
    # specific unit always trigger resetting the limits since different datasets
    # might be converted in different ways.
    assert viewer.state.y_min == 1000.
    assert viewer.state.y_max == 3000.

    # Now set the limits explicitly again and make sure in future they are converted
    viewer.state.y_min = 500.
    viewer.state.y_max = 3500.

    roi = XRangeROI(0.5, 1.2)
    viewer.apply_roi(roi)

    assert len(d1.subsets) == 1
    assert_equal(d1.subsets[0].to_mask(), [1, 0, 0])

    assert len(d2.subsets) == 1
    assert_equal(d2.subsets[0].to_mask(), [0, 0, 1])

    viewer.state.x_display_unit = 'cm'
    viewer.state.y_display_unit = 'Jy'

    roi = XRangeROI(15, 35)
    viewer.apply_roi(roi)

    assert len(d1.subsets) == 1
    assert_equal(d1.subsets[0].to_mask(), [1, 0, 0])

    assert len(d2.subsets) == 1
    assert_equal(d2.subsets[0].to_mask(), [0, 1, 1])

    assert_allclose(viewer.state.x_min, (4 * u.GHz).to_value(u.cm, equivalencies=u.spectral()))
    assert_allclose(viewer.state.x_max, (0.5 * u.GHz).to_value(u.cm, equivalencies=u.spectral()))
    assert_allclose(viewer.state.y_min, 0.5)
    assert_allclose(viewer.state.y_max, 3.5)
