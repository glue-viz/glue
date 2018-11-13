from numpy.testing import assert_allclose

import matplotlib.pyplot as plt

from glue.core.application_base import Application
from glue.viewers.common.viewer import Viewer
from glue.viewers.matplotlib.viewer import MatplotlibViewerMixin
from glue.viewers.matplotlib.state import MatplotlibDataViewerState


def assert_limits(viewer, x_min, x_max, y_min, y_max):
    # Convenience to check both state and matplotlib
    assert_allclose(viewer.state.x_min, x_min)
    assert_allclose(viewer.state.x_max, x_max)
    assert_allclose(viewer.state.y_min, y_min)
    assert_allclose(viewer.state.y_max, y_max)
    assert_allclose(viewer.axes.get_xlim(), (x_min, x_max))
    assert_allclose(viewer.axes.get_ylim(), (y_min, y_max))


def test_aspect_ratio():

    # Test of the aspect ratio infrastructure

    class CustomViewer(MatplotlibViewerMixin, Viewer):

        _state_cls = MatplotlibDataViewerState

        def __init__(self, *args, **kwargs):
            Viewer.__init__(self, *args, **kwargs)
            self.figure = plt.figure(figsize=(12, 6))
            self.axes = self.figure.add_axes([0, 0, 1, 1])
            MatplotlibViewerMixin.setup_callbacks(self)

        def show(self):
            pass

    class CustomApplication(Application):
        def add_widget(self, *args, **kwargs):
            pass

    app = CustomApplication()

    viewer = app.new_data_viewer(CustomViewer)
    viewer.state.aspect = 'equal'

    assert_limits(viewer, -0.5, 1.5, 0., 1.)

    # Test changing x limits in state, which should just change the y limits

    viewer.state.x_min = -2.5
    assert_limits(viewer, -2.5, 1.5, -0.5, 1.5)

    viewer.state.x_max = -1.5
    assert_limits(viewer, -2.5, -1.5, 0.25, 0.75)

    # Test changing y limits in state, which should just change the x limits

    viewer.state.y_max = 1.25
    assert_limits(viewer, -3.0, -1.0, 0.25, 1.25)

    viewer.state.y_min = 0.75
    assert_limits(viewer, -2.5, -1.5, 0.75, 1.25)

    # Test changing x limits in Matplotlib, which should just change the x limits

    viewer.axes.set_xlim(1., 3.)
    assert_limits(viewer, 1.0, 3.0, 0.5, 1.5)

    # Test changing x limits in Matplotlib, which should just change the x limits

    viewer.axes.set_ylim(0., 2.)
    assert_limits(viewer, 0.0, 4.0, 0.0, 2.0)

    # We include tests for resizing inside the Qt folder (since this doesn't
    # work correctly with the Agg backend)
