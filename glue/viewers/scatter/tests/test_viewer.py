import numpy as np
from numpy.testing import assert_allclose

import matplotlib.pyplot as plt

from glue.tests.visual.helpers import visual_test

from glue.viewers.scatter.viewer import SimpleScatterViewer
from glue.core.application_base import Application
from glue.core.data import Data
from glue.core.link_helpers import LinkSame


@visual_test
def test_simple_viewer():

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
