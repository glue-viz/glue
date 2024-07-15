from astropy import units as u
from astropy.wcs import WCS

import numpy as np
from numpy.testing import assert_equal, assert_allclose

from glue.tests.visual.helpers import visual_test
from glue.viewers.profile.viewer import SimpleProfileViewer
from glue.core.application_base import Application
from glue.core.data import Data
from glue.config import settings, unit_converter
from glue.plugins.wcs_autolinking.wcs_autolinking import WCSLink
from glue.core.roi import XRangeROI
from glue.core.data_derived import IndexedData


@visual_test
def test_simple_profile_viewer():

    # Make sure the simple viewer can be instantiated

    data1 = Data(x=[1, 2, 3], label='data1')
    data2 = Data(y=[1, 2, 3], label='data2')

    app = Application()
    app.data_collection.append(data1)
    app.data_collection.append(data2)

    viewer = app.new_data_viewer(SimpleProfileViewer)
    viewer.add_data(data1)
    viewer.add_data(data2)

    app.data_collection.new_subset_group(label='subset1', subset_state=data1.pixel_component_ids[0] > 0.8)

    viewer.state.layers[2].linewidth = 5

    return viewer.figure


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

    app = Application()
    session = app.session

    data_collection = session.data_collection
    data_collection.append(d1)
    data_collection.append(d2)

    data_collection.add_link(WCSLink(d1, d2))

    viewer = app.new_data_viewer(SimpleProfileViewer)
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

    # Regression test for a bug that caused unit changes to not work on y axis
    # if reference data was not first layer

    viewer.state.reference_data = d2
    viewer.state.y_display_unit = 'mJy'


def test_indexed_data():

    # Make sure that the profile viewer works properly with IndexedData objects

    data_4d = Data(label='hypercube_wcs',
                   x=np.random.random((3, 5, 4, 3)),
                   coords=WCS(naxis=4))

    data_2d = IndexedData(data_4d, (2, None, 3, None))

    application = Application()

    session = application.session

    hub = session.hub

    data_collection = session.data_collection
    data_collection.append(data_4d)
    data_collection.append(data_2d)

    viewer = application.new_data_viewer(SimpleProfileViewer)
    viewer.add_data(data_2d)

    assert viewer.state.x_att is data_2d.world_component_ids[0]
