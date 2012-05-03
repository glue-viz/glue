from PyQt4.QtGui import QIcon

from glue import roi


class MouseMode(object):
    def __init__(self, axes,
                 press_callback=None,
                 move_callback=None,
                 release_callback=None):

        self.icon = None
        self.mode_id = None
        self.action_text = None
        self.tool_tip = None
        self._axes = axes
        self._press_callback = press_callback
        self._move_callback = move_callback
        self._release_callback = release_callback
        self.shortcut = None

    def press(self, event):
        if self._press_callback is not None:
            self._press_callback(self)

    def move(self, event):
        if self._move_callback is not None:
            self._move_callback(self)

    def release(self, event):
        if self._release_callback is not None:
            self._release_callback(self)


class RectangleMode(MouseMode):
    def __init__(self, axes, **kwargs):
        super(RectangleMode, self).__init__(axes, **kwargs)
        self.icon = QIcon(':icons/glue_square.png')
        self.mode_id = 'Rectangle'
        self.action_text = 'Rectangular ROI'
        self.tool_tip = 'Define a rectangular region of interest'
        self._roi_tool = roi.MplRectangularROI(self._axes)
        self.shortcut = 'R'

    def roi(self):
        return self._roi_tool.roi()

    def press(self, event):
        self._roi_tool.start_selection(event)
        super(RectangleMode, self).press(event)

    def move(self, event):
        self._roi_tool.update_selection(event)
        super(RectangleMode, self).move(event)

    def release(self, event):
        self._roi_tool.finalize_selection(event)
        super(RectangleMode, self).release(event)


class CircleMode(MouseMode):
    def __init__(self, axes, **kwargs):
        super(CircleMode, self).__init__(axes, **kwargs)
        self.icon = QIcon(':icons/glue_circle.png')
        self.mode_id = 'Circle'
        self.action_text = 'Circular ROI'
        self.tool_tip = 'Define a circular region of interest'
        self._roi_tool = roi.MplCircularROI(self._axes)
        self.shortcut = 'C'

    def roi(self):
        return self._roi_tool.roi()

    def press(self, event):
        self._roi_tool.start_selection(event)
        super(CircleMode, self).press(event)

    def move(self, event):
        self._roi_tool.update_selection(event)
        super(CircleMode, self).move(event)

    def release(self, event):
        self._roi_tool.finalize_selection(event)
        super(CircleMode, self).release(event)


class PolyMode(MouseMode):
    def __init__(self, axes, **kwargs):
        super(PolyMode, self).__init__(axes, **kwargs)
        self.icon = QIcon(':icons/glue_lasso.png')
        self.mode_id = 'Lasso'
        self.action_text = 'Polygonal ROI'
        self.tool_tip = 'Lasso a region of interest'
        self._roi_tool = roi.MplPolygonalROI(self._axes)
        self.shortcut = 'L'

    def roi(self):
        return self._roi_tool.roi()

    def press(self, event):
        self._roi_tool.start_selection(event)
        super(PolyMode, self).press(event)

    def move(self, event):
        self._roi_tool.update_selection(event)
        super(PolyMode, self).move(event)

    def release(self, event):
        self._roi_tool.finalize_selection(event)
        super(PolyMode, self).release(event)
