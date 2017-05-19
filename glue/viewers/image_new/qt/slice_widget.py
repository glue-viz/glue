from __future__ import absolute_import, division, print_function

import numpy as np

from qtpy import QtWidgets
from qtpy.QtCore import Qt

__all__ = ['MultiSliceWidgetHelper']


class MultiSliceWidgetHelper(object):

    def __init__(self, viewer_state=None, widget=None):

        self.widget = widget
        self.viewer_state = viewer_state

        self.layout = widget.layout()
        self.layout.setSpacing(4)
        self.layout.setContentsMargins(0, 3, 0, 3)

        self.viewer_state.add_callback('x_att', self.sync_sliders_from_state)
        self.viewer_state.add_callback('y_att', self.sync_sliders_from_state)
        self.viewer_state.add_callback('slices', self.sync_sliders_from_state)
        self.viewer_state.add_callback('reference_data', self.sync_sliders_from_state)

        self._sliders = []

        self.sync_sliders_from_state()

    @property
    def data(self):
        return self.viewer_state.reference_data

    def _clear(self):

        for _ in range(self.layout.count()):
            self.layout.takeAt(0)

        for s in self._sliders:
            s.close()

        self._slices = []

    def sync_state_from_sliders(self, *args):
        slices = []
        for i, slider in enumerate(self._sliders):
            slices.append(slider.value())
        self.viewer_state.slices = tuple(slices)

    def sync_sliders_from_state(self, *args):

        if self.data is None or self.viewer_state.x_att is None or self.viewer_state.y_att is None:
            return

        # TODO: figure out why there are no current circular calls (normally
        # we should need to add @avoid_circular)

        # Update number of sliders if needed
        if self.data.ndim != len(self._sliders):
            self._clear()
            for i in range(self.data.ndim):
                slider = QtWidgets.QSlider(Qt.Horizontal)
                slider.setMinimum(0)
                slider.setMaximum(self.data.shape[i] - 1)
                slider.valueChanged.connect(self.sync_state_from_sliders)
                self._sliders.append(slider)
                self.layout.addWidget(slider)

        # Disable sliders that correspond to visible axes and sync position

        for i, slider in enumerate(self._sliders):

            if i == self.viewer_state.x_att.axis or i == self.viewer_state.y_att.axis:
                slider.setEnabled(False)
            else:
                slider.setEnabled(True)
                slider.setValue(self.viewer_state.slices[i])



if __name__ == "__main__":

    from glue.core import Data
    from glue.utils.qt import get_qapp
    from glue.external.echo import CallbackProperty
    from glue.core.state_objects import State

    app = get_qapp()

    class FakeViewerState(State):
        x_att = CallbackProperty()
        y_att = CallbackProperty()
        reference_data = CallbackProperty()
        slices = CallbackProperty()

    viewer_state = FakeViewerState()

    data = Data(x=np.random.random((3, 50, 20, 5, 3)))

    viewer_state.reference_data = data
    viewer_state.x_att = data.get_pixel_component_id(0)
    viewer_state.y_att = data.get_pixel_component_id(3)
    viewer_state.slices = [0] * 5

    widget = MultiSliceWidget(viewer_state)
    widget.show()

    app.exec_()
