import numpy as np
from numpy.testing import assert_allclose, assert_equal

import matplotlib.pyplot as plt

from glue.tests.visual.helpers import visual_test

from glue.viewers.scatter.viewer import SimpleScatterViewer
from glue.core.application_base import Application
from glue.core.data import Data
from glue.core.link_helpers import LinkSame, LinkSameWithUnits
from glue.core.data_derived import IndexedData
from glue.core.roi import RectangularROI


@visual_test
def test_simple_scatter_viewer():

    # Make sure the simple viewer can be instantiated

    data1 = Data(x=[1, 2, 3], label='data1')
    data2 = Data(y=[1, 2, 3], label='data2')

    app = Application()
    app.data_collection.append(data1)
    app.data_collection.append(data2)

    viewer = app.new_data_viewer(SimpleScatterViewer)
    viewer.add_data(data1)
    viewer.add_data(data2)

    app.data_collection.new_subset_group(label='subset1', subset_state=data1.id['x'] > 2)

    return viewer.figure


@visual_test
def test_scatter_density_map():

    # Test the scatter density map

    np.random.seed(12345)

    app = Application()

    x = np.random.normal(3, 1, 100)
    y = np.random.normal(2, 1.5, 100)
    c = np.hypot(x - 3, y - 2)
    s = (x - 3)

    data1 = app.add_data(a={"x": x, "y": y, "c": c, "s": s})[0]

    xx = np.random.normal(3, 1, 1000000)
    yy = np.random.normal(2, 1.5, 1000000)

    data2 = app.add_data(a={"x": xx, "y": yy})[0]

    app.data_collection.add_link(LinkSame(data1.id['x'], data2.id['x']))
    app.data_collection.add_link(LinkSame(data1.id['y'], data2.id['y']))

    viewer = app.new_data_viewer(SimpleScatterViewer)
    viewer.add_data(data1)
    viewer.add_data(data2)

    viewer.state.layers[0].cmap_mode = 'Linear'
    viewer.state.layers[0].cmap_att = data1.id['c']
    viewer.state.layers[0].cmap = plt.cm.viridis
    viewer.state.layers[0].size_mode = 'Linear'
    viewer.state.layers[0].size_att = data1.id['s']

    viewer.state.layers[1].zorder = 0.5

    app.data_collection.new_subset_group(label='subset1', subset_state=data1.id['x'] > 2)

    return viewer.figure


def test_reset_limits():

    data1 = Data(x=np.arange(1000), y=np.arange(1000) + 1000, label='data')

    app = Application()
    app.data_collection.append(data1)

    viewer = app.new_data_viewer(SimpleScatterViewer)
    viewer.add_data(data1)

    viewer.state.reset_limits()

    # Note that there is a margin included which is why the limits are not 0 to 999

    assert_allclose(viewer.state.x_min, -39.96)
    assert_allclose(viewer.state.x_max, 1038.96)

    assert_allclose(viewer.state.y_min, 1000 - 39.96)
    assert_allclose(viewer.state.y_max, 1000 + 1038.96)

    viewer.state.x_limits_percentile = 90
    viewer.state.y_limits_percentile = 80

    viewer.state.reset_limits()

    assert_allclose(viewer.state.x_min, 13.986)
    assert_allclose(viewer.state.x_max, 985.014)

    assert_allclose(viewer.state.y_min, 1000 + 67.932)
    assert_allclose(viewer.state.y_max, 1000 + 931.068)


def test_indexed_data():

    # Make sure that the scatter viewer works properly with IndexedData objects

    data_4d = Data(label='hypercube',
                   x=np.random.random((3, 5, 4, 3)),
                   y=np.random.random((3, 5, 4, 3)))

    data_2d = IndexedData(data_4d, (2, None, 3, None))

    application = Application()

    session = application.session

    hub = session.hub

    data_collection = session.data_collection
    data_collection.append(data_4d)
    data_collection.append(data_2d)

    viewer = application.new_data_viewer(SimpleScatterViewer)
    viewer.add_data(data_2d)

    assert viewer.state.x_att is data_2d.main_components[0]
    assert viewer.state.y_att is data_2d.main_components[1]


def test_unit_conversion():

    d1 = Data(a=[1, 2, 3], b=[2, 3, 4])
    d1.get_component('a').units = 'm'
    d1.get_component('b').units = 's'

    d2 = Data(c=[2000, 1000, 3000], d=[0.001, 0.002, 0.004])
    d2.get_component('c').units = 'mm'
    d2.get_component('d').units = 'ks'

    # d3 is the same as d2 but we will link it differently
    d3 = Data(e=[2000, 1000, 3000], f=[0.001, 0.002, 0.004])
    d3.get_component('e').units = 'mm'
    d3.get_component('f').units = 'ks'

    d4 = Data(g=[2, 2, 3], h=[1, 2, 1])
    d4.get_component('g').units = 'kg'
    d4.get_component('h').units = 'm/s'

    app = Application()
    session = app.session

    data_collection = session.data_collection
    data_collection.append(d1)
    data_collection.append(d2)
    data_collection.append(d3)
    data_collection.append(d4)

    data_collection.add_link(LinkSameWithUnits(d1.id['a'], d2.id['c']))
    data_collection.add_link(LinkSameWithUnits(d1.id['b'], d2.id['d']))
    data_collection.add_link(LinkSame(d1.id['a'], d3.id['e']))
    data_collection.add_link(LinkSame(d1.id['b'], d3.id['f']))
    data_collection.add_link(LinkSame(d1.id['a'], d4.id['g']))
    data_collection.add_link(LinkSame(d1.id['b'], d4.id['h']))

    viewer = app.new_data_viewer(SimpleScatterViewer)
    viewer.add_data(d1)
    viewer.add_data(d2)
    viewer.add_data(d3)
    viewer.add_data(d4)

    assert viewer.layers[0].enabled
    assert viewer.layers[1].enabled
    assert viewer.layers[2].enabled
    assert viewer.layers[3].enabled

    assert viewer.state.x_min == 0.92
    assert viewer.state.x_max == 3.08
    assert viewer.state.y_min == 1.92
    assert viewer.state.y_max == 4.08

    roi = RectangularROI(0.5, 2.5, 1.5, 4.5)
    viewer.apply_roi(roi)

    assert len(d1.subsets) == 1
    assert_equal(d1.subsets[0].to_mask(), [1, 1, 0])

    # Because of the LinkSameWithUnits, the points actually appear in the right
    # place even before we set the display units.
    assert len(d2.subsets) == 1
    assert_equal(d2.subsets[0].to_mask(), [0, 1, 0])

    # d3 is only linked with LinkSame not LinkSameWithUnits so currently the
    # points are outside the visible axes
    assert len(d3.subsets) == 1
    assert_equal(d3.subsets[0].to_mask(), [0, 0, 0])

    # As we haven't set display units yet, the values for this dataset are shown
    # on the same scale as for d1 as if the units had never been set.
    assert len(d4.subsets) == 1
    assert_equal(d4.subsets[0].to_mask(), [0, 1, 0])

    # Now try setting the units explicitly

    viewer.state.x_display_unit = 'km'
    viewer.state.y_display_unit = 'ms'

    assert_allclose(viewer.state.x_min, 0.92e-3)
    assert_allclose(viewer.state.x_max, 3.08e-3)
    assert_allclose(viewer.state.y_min, 1.92e3)
    assert_allclose(viewer.state.y_max, 4.08e3)

    roi = RectangularROI(0.5e-3, 2.5e-3, 1.5e3, 4.5e3)
    viewer.apply_roi(roi)

    # d1 and d2 will be as above, but d3 will now work correctly while d4 should
    # not be shown.

    assert_equal(d1.subsets[1].to_mask(), [1, 1, 0])
    assert_equal(d2.subsets[1].to_mask(), [0, 1, 0])
    assert_equal(d3.subsets[1].to_mask(), [0, 0, 0])
    assert_equal(d4.subsets[1].to_mask(), [0, 1, 0])


    # # Change the limits to make sure they are always converted
    # viewer.state.x_min = 5e8
    # viewer.state.x_max = 4e9
    # viewer.state.y_min = 0.5
    # viewer.state.y_max = 3.5

    # roi = XRangeROI(1.4e9, 2.1e9)
    # viewer.apply_roi(roi)

    # assert len(d1.subsets) == 1
    # assert_equal(d1.subsets[0].to_mask(), [0, 1, 0])

    # assert len(d2.subsets) == 1
    # assert_equal(d2.subsets[0].to_mask(), [0, 1, 0])

    # viewer.state.x_display_unit = 'GHz'
    # viewer.state.y_display_unit = 'mJy'

    # x, y = viewer.state.layers[0].profile
    # assert_allclose(x, [1, 2, 3])
    # assert_allclose(y, [1000, 2000, 3000])

    # x, y = viewer.state.layers[1].profile
    # assert_allclose(x, 2.99792458 / np.array([1, 2, 3]))
    # assert_allclose(y, [2000, 1000, 3000])

    # assert viewer.state.x_min == 0.5
    # assert viewer.state.x_max == 4.

    # # Units get reset because they were originally 'native' and 'native' to a
    # # specific unit always trigger resetting the limits since different datasets
    # # might be converted in different ways.
    # assert viewer.state.y_min == 1000.
    # assert viewer.state.y_max == 3000.

    # # Now set the limits explicitly again and make sure in future they are converted
    # viewer.state.y_min = 500.
    # viewer.state.y_max = 3500.

    # roi = XRangeROI(0.5, 1.2)
    # viewer.apply_roi(roi)

    # assert len(d1.subsets) == 1
    # assert_equal(d1.subsets[0].to_mask(), [1, 0, 0])

    # assert len(d2.subsets) == 1
    # assert_equal(d2.subsets[0].to_mask(), [0, 0, 1])

    # viewer.state.x_display_unit = 'cm'
    # viewer.state.y_display_unit = 'Jy'

    # roi = XRangeROI(15, 35)
    # viewer.apply_roi(roi)

    # assert len(d1.subsets) == 1
    # assert_equal(d1.subsets[0].to_mask(), [1, 0, 0])

    # assert len(d2.subsets) == 1
    # assert_equal(d2.subsets[0].to_mask(), [0, 1, 1])

    # assert_allclose(viewer.state.x_min, (4 * u.GHz).to_value(u.cm, equivalencies=u.spectral()))
    # assert_allclose(viewer.state.x_max, (0.5 * u.GHz).to_value(u.cm, equivalencies=u.spectral()))
    # assert_allclose(viewer.state.y_min, 0.5)
    # assert_allclose(viewer.state.y_max, 3.5)

    # # Regression test for a bug that caused unit changes to not work on y axis
    # # if reference data was not first layer

    # viewer.state.reference_data = d2
    # viewer.state.y_display_unit = 'mJy'


    # data_collection.add_link(LinkSame(d1.id['a'], d2.id['e']))
    # data_collection.add_link(LinkSame(d1.id['b'], d2.id['f']))
