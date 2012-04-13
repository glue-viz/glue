from PyQt4.QtGui import QIcon

import cv_qt_resources
from glue import roi

class MouseMode(object):
    def __init__(self, axes, callback=None):
        self.icon = None
        self.mode_id = None
        self.action_text = None
        self.tool_tip = None
        self._axes = axes
        self._callback = callback

    def press(self, event):
        raise NotImplementedError

    def move(self, event):
        raise NotImplementedError

    def release(self, event):
        if self._callback is not None:
            self._callback(self)

class RectangleMode(MouseMode):
    def __init__(self, axes, callback=None):
        super(RectangleMode, self).__init__(axes, callback)
        self.icon = QIcon(':icons/square.png')
        self.mode_id = 'Rectangle'
        self.action_text = 'Rectangular ROI'
        self.tool_tip = 'Define a rectangular region of interest'
        self._roi_tool = roi.MplRectangularROI(self._axes)

    def roi(self):
        return self._roi_tool.roi()

    def press(self, event):
        self._roi_tool.start_selection(event)

    def move(self, event):
        self._roi_tool.update_selection(event)

    def release(self, event):
        self._roi_tool.finalize_selection(event)
        super(RectangleMode, self).release(event)
