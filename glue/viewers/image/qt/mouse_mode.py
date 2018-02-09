from __future__ import absolute_import, division, print_function

from glue.core.edit_subset_mode import EditSubsetMode
from glue.core.subset import RoiSubsetState
from glue.core.qt.roi import QtPolygonalROI
from glue.viewers.common.qt.mouse_mode import MouseMode
from glue.viewers.image.layer_artist import ImageSubsetLayerArtist


__all__ = ['RoiClickAndDragMode']


_MPL_LEFT_CLICK = 1
_MPL_RIGHT_CLICK = 3


class RoiClickAndDragMode(MouseMode):
    """

    """

    def __init__(self, viewer, **kwargs):
        super(RoiClickAndDragMode, self).__init__(viewer, **kwargs)

        self._viewer = viewer
        self._dc = self._viewer.state.data_collection
        self._edit_subset_mode = EditSubsetMode()

        self._roi = None
        self._subset = None

    def _select_roi(self, roi, index, event):
        self._roi = QtPolygonalROI(self._axes, _roi=roi)
        self._roi.start_selection(event, scrubbing=True)
        self._edit_subset_mode.edit_subset = [self._dc.subset_groups[index]]

    def press(self, event):
        # Ignore button presses outside the data viewer canvas
        if event.xdata is None or event.ydata is None:
            return

        x, y = (int(event.xdata + 0.5), int(event.ydata + 0.5))

        roi_index = 0
        for layer in self._viewer.layers:
            if not isinstance(layer, ImageSubsetLayerArtist):
                continue

            subset_state = layer.state.layer.subset_state
            if isinstance(subset_state, RoiSubsetState):
                if subset_state.roi.contains(x, y):
                    if event.button == _MPL_LEFT_CLICK:
                        self._select_roi(subset_state.roi, roi_index, event)
                        self._subset = layer.state.layer
            roi_index += 1

    def move(self, event):
        if self._roi is None:
            return

        self._roi.update_selection(event)

    def release(self, event):
        if self._roi:
            self._roi.finalize_selection(event)
            self._viewer.apply_roi(self._roi.roi())

            self._roi = None
            self._subset = None
