from __future__ import absolute_import, division, print_function

from glue.config import viewer_tool
from glue.core.data_derived import IndexedData
from glue.viewers.matplotlib.toolbar_mode import ToolbarModeBase

__all__ = ['PixelExtractionTool']


@viewer_tool
class PixelExtractionTool(ToolbarModeBase):
    """
    Create a derived dataset corresponding to the selected pixel.
    """

    icon = "glue_pixel_extraction"
    tool_id = 'image:pixel_extraction'
    action_text = 'Pixel extraction'
    tool_tip = 'Extract data for a single pixel based on mouse location'
    status_tip = 'CLICK to select a point, then CLICK and DRAG to update the extracted dataset in real time'

    _pressed = False

    def __init__(self, *args, **kwargs):
        super(PixelExtractionTool, self).__init__(*args, **kwargs)
        self._move_callback = self._extract_pixel
        self._press_callback = self._on_press
        self._release_callback = self._on_release
        self._derived = None

        self._line_x = self.viewer.axes.axvline(0, color='orange')
        self._line_x.set_visible(False)

        self._line_y = self.viewer.axes.axhline(0, color='orange')
        self._line_y.set_visible(False)

    def _on_press(self, mode):
        self._pressed = True
        self._extract_pixel(mode)

    def _on_release(self, mode):
        self._pressed = False

    def _extract_pixel(self, mode):

        if not self._pressed:
            return

        x, y = self._event_xdata, self._event_ydata

        if x is None or y is None:
            return None

        xi = int(round(x))
        yi = int(round(y))

        indices = [None] * self.viewer.state.reference_data.ndim
        indices[self.viewer.state.x_att.axis] = xi
        indices[self.viewer.state.y_att.axis] = yi

        self._line_x.set_data([x, x], [0, 1])
        self._line_x.set_visible(True)
        self._line_y.set_data([0, 1], [y, y])
        self._line_y.set_visible(True)
        self.viewer.axes.figure.canvas.draw()

        if self._derived is None:
            self._derived = IndexedData(self.viewer.state.reference_data, indices)
            self.viewer.session.data_collection.append(self._derived)
        else:
            try:
                self._derived.indices = indices
            except TypeError:
                self.viewer.session.data_collection.remove(self._derived)
                self._derived = IndexedData(self.viewer.state.reference_data, indices)
                self.viewer.session.data_collection.append(self._derived)

# ImageViewer.tools.append('image:pixel_extraction')
