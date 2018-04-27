from __future__ import absolute_import, division, print_function

from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QMenu, QAction

from glue.core.subset import RoiSubsetState
from glue.core.qt.roi import QtPolygonalROI
from glue.viewers.common.qt.mouse_mode import MouseMode
from glue.viewers.image.layer_artist import ImageSubsetLayerArtist


__all__ = ['RoiClickAndDragMode']


_MPL_LEFT_CLICK = 1
_MPL_RIGHT_CLICK = 3


class RoiClickAndDragMode(MouseMode):
    """
    A MouseMode that enables clicking and dragging of existing ROIs.
    """

    def __init__(self, viewer, **kwargs):
        super(RoiClickAndDragMode, self).__init__(viewer, **kwargs)

        self._viewer = viewer
        self._dc = viewer.state.data_collection
        self._edit_subset_mode = viewer.session.edit_subset_mode

        self._roi = None
        self._subset = None
        self._selected = False

    def _select_roi(self, roi, index, event):
        self._roi = QtPolygonalROI(self._axes, roi=roi)
        self._roi.start_selection(event, scrubbing=True)
        self._edit_subset_mode.edit_subset = [self._dc.subset_groups[index]]

    def _deselect_roi(self, event):

        self._edit_subset_mode.edit_subset = []

        if self._roi:
            self._roi = None
            self._subset = None

    def _display_roi_context_menu(self, roi_index):

        def delete_roi(event):
            self._dc.remove_subset_group(self._dc.subset_groups[roi_index])

        context_menu = QMenu()
        action = QAction("Delete ROI", context_menu)
        action.triggered.connect(delete_roi)
        context_menu.addAction(action)
        pos = self._viewer.mapToParent(QCursor().pos())
        context_menu.exec_(pos)

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
            if layer.visible and isinstance(subset_state, RoiSubsetState):
                if subset_state.roi.contains(x, y):
                    if event.button == _MPL_LEFT_CLICK:
                        self._select_roi(subset_state.roi, roi_index, event)
                        self._subset = layer.state.layer
                    elif event.button == _MPL_RIGHT_CLICK:
                        self._display_roi_context_menu(roi_index)
                    self._selected = True
                    break
            roi_index += 1
        else:
            self._selected = False
            self._deselect_roi(event)

    def move(self, event):
        if self._roi is None or not self._selected:
            return

        self._roi.update_selection(event)

    def release(self, event):
        if self._roi:
            self._viewer.apply_roi(self._roi.roi(), use_current=True)
            self._roi.finalize_selection(event)
            self._selected = False
