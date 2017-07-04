# New contrast/bias mode that operates on viewers with state objects

from __future__ import absolute_import, division, print_function

import os

from qtpy import QtGui, QtWidgets
from glue.core.callback_property import CallbackProperty
from glue.core import roi
from glue.core.qt import roi as qt_roi
from glue.utils.qt import get_qapp
from glue.utils import nonpartial
from glue.utils.qt import load_ui, cmap2pixmap
from glue.viewers.common.qt.tool import Tool, CheckableTool
from glue.external.echo import delay_callback
from glue.config import viewer_tool
from glue.viewers.common.qt.mouse_mode import MouseMode


@viewer_tool
class ContrastBiasMode(MouseMode):
    """
    Uses right mouse button drags to set bias and contrast, DS9-style.    The
    horizontal position of the mouse sets the bias, the vertical position sets
    the contrast.
    """

    icon = 'glue_contrast'
    tool_id = 'image:contrast_bias'
    action_text = 'Contrast/Bias'
    tool_tip = 'Adjust the bias/contrast'

    def move(self, event):
        """
        Update bias and contrast on Right Mouse button drag.
        """

        if event.button not in (1, 3):
            return

        x, y = event.x, event.y
        dx, dy = self._axes.figure.canvas.get_width_height()
        x = 1.0 * x / dx
        y = 1.0 * y / dy

        state = self.viewer.selected_layer.state

        with delay_callback(state, 'bias', 'contrast'):
            state.bias = x
            state.contrast = (1 - y) * 10

        super(ContrastBiasMode, self).move(event)
