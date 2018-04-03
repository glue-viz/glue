from __future__ import absolute_import, division, print_function

from glue.core import roi
from glue.config import viewer_tool
from glue.viewers.common.qt.toolbar_mode import ToolbarModeBase

__all__ = ['PixelSelectionTool']


@viewer_tool
class PixelSelectionTool(ToolbarModeBase):
    """
    Selects pixel under mouse cursor.
    """

    icon = "glue_point"
    tool_id = 'image:point_selection'
    action_text = 'Pixel'
    tool_tip = 'Select a point based on mouse location'
    status_tip = ('Mouse over to select a point. Click on the image to enable or disable selection.')

    _on_move = False

    def __init__(self, *args, **kwargs):
        super(PixelSelectionTool, self).__init__(*args, **kwargs)
        self._move_callback = self._select_pixel
        self._press_callback = self._on_press

    def _on_press(self, mode):
        self._on_move = not self._on_move

    def _select_pixel(self, mode):
        """
        Select a pixel
        """

        if not self._on_move:
            return

        x, y = self._event_xdata, self._event_ydata

        if x is None or y is None:
            return None

        x = int(round(x))
        y = int(round(y))

        p = roi.RectangularROI(x-0.5, x+0.5, y-0.5, y+0.5)

        if roi:
            self.viewer.apply_roi(p)
