from __future__ import absolute_import, division, print_function

from glue.config import viewer_tool

from glue.core.data_derived import IndexedData

from glue.viewers.common.qt.toolbar_mode import ToolbarModeBase
from glue.viewers.image.pixel_selection_subset_state import PixelSubsetState

__all__ = ['PixelSelectionTool']


@viewer_tool
class PixelSelectionTool(ToolbarModeBase):
    """
    Selects pixel under mouse cursor.
    """

    icon = "glue_crosshair"
    tool_id = 'image:point_selection'
    action_text = 'Pixel'
    tool_tip = 'Select a single pixel based on mouse location'
    status_tip = 'CLICK to select a point, CLICK and DRAG to update the selection in real time'

    _pressed = False

    def __init__(self, *args, **kwargs):
        super(PixelSelectionTool, self).__init__(*args, **kwargs)
        self._move_callback = self._select_pixel
        self._press_callback = self._on_press
        self._release_callback = self._on_release
        self._derived = None

    def _on_press(self, mode):
        self._pressed = True
        self._select_pixel(mode)

    def _on_release(self, mode):
        self._pressed = False

    def _select_pixel(self, mode):
        """
        Select a pixel
        """

        if not self._pressed:
            return

        x, y = self._event_xdata, self._event_ydata

        if x is None or y is None:
            return None

        x = int(round(x))
        y = int(round(y))

        indices = [None] * self.viewer.state.reference_data.ndim
        indices[self.viewer.state.x_att.axis] = slice(x, x + 1)
        indices[self.viewer.state.y_att.axis] = slice(y, y + 1)

        if self._derived is None:
            self._derived = IndexedData(self.viewer.state.reference_data, indices)
            self.viewer.session.data_collection.append(self._derived)
        else:
            self._derived.indices = indices
