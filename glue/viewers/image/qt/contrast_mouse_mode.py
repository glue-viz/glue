# New contrast/bias mode that operates on viewers with state objects

from echo import delay_callback
from glue.config import viewer_tool
from glue.viewers.matplotlib.toolbar_mode import ToolbarModeBase


@viewer_tool
class ContrastBiasMode(ToolbarModeBase):
    """
    Uses right mouse button drags to set bias and contrast, DS9-style.    The
    horizontal position of the mouse sets the bias, the vertical position sets
    the contrast.
    """

    icon = 'glue_contrast'
    tool_id = 'image:contrast_bias'
    action_text = 'Contrast/Bias'
    tool_tip = 'Adjust the bias/contrast'
    status_tip = ('CLICK and DRAG on image from left to right to adjust '
                  'bias and up and down to adjust contrast')

    def move(self, event):
        """
        Update bias and contrast on Right Mouse button drag.
        """

        if event.button not in (1, 3):
            return

        x, y = self.viewer.axes.transAxes.inverted().transform((event.x, event.y))
        state = self.viewer.selected_layer.state

        with delay_callback(state, 'bias', 'contrast'):
            state.bias = -(x * 2 - 1.5)
            state.contrast = 10. ** (y * 2 - 1)

        super(ContrastBiasMode, self).move(event)
