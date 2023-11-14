import numpy as np
from numpy.testing import assert_allclose

from glue.tests.visual.helpers import visual_test

from glue.viewers.scatter.viewer import SimpleScatterViewer
from glue.core.application_base import Application
from glue.core.data import Data


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
