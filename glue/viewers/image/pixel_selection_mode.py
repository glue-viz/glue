from __future__ import absolute_import, division, print_function

from glue.config import viewer_tool
from glue.viewers.common.qt.toolbar_mode import ToolbarModeBase
from glue.core.command import ApplySubsetState
from glue.core.subset import SliceSubsetState
from glue.core.edit_subset_mode import ReplaceMode

__all__ = ['PixelSelectionTool']


class PixelSubsetState(SliceSubsetState):
    def copy(self):
        return PixelSubsetState(self.reference_data, self.slices)


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

    def _on_press(self, mode):
        self._pressed = True
        self.viewer.session.edit_subset_mode.mode = ReplaceMode
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

        slices = [slice(None)] * self.viewer.state.reference_data.ndim
        slices[self.viewer.state.x_att.axis] = slice(x, x + 1)
        slices[self.viewer.state.y_att.axis] = slice(y, y + 1)

        subset_state = PixelSubsetState(self.viewer.state.reference_data, slices)

        cmd = ApplySubsetState(data_collection=self.viewer._data,
                               subset_state=subset_state,
                               use_current=False)
        self.viewer._session.command_stack.do(cmd)
