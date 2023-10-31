import numpy as np
from numpy.testing import assert_allclose

from glue.viewers.common.viewer import Viewer
from glue.viewers.scatter.state import ScatterViewerState
from glue.core.application_base import Application
from glue.core.data import Data


class TestScatterViewer(Viewer):
    _state_cls = ScatterViewerState


def test_reset_limits():

    data1 = Data(x=np.arange(1000), y=np.arange(1000) + 1000, label='data')

    app = Application()
    app.data_collection.append(data1)

    viewer = app.new_data_viewer(TestScatterViewer)
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
