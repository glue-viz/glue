"""MouseModes define various mouse gestures.

The GlueToolbar maintains a list of MouseModes from the visualization
it is assigned to, and sees to it that only one MouseMode is active at
a time.

Each MouseMode appears as an Icon in the GlueToolbar. Classes can
assign methods to the press_callback, move_callback, and
release_callback methods of each Mouse Mode, to implement custom
functionality

The basic usage pattern is thus:
 * visualization object instantiates the MouseModes it wants
 * each of these is passed to the add_mode method of the GlueToolbar
 * visualization object optionally attaches methods to the 3 _callback
   methods in a MouseMode, for additional behavior

"""
from PyQt4.QtGui import QIcon

import numpy as np

from glue import roi


class MouseMode(object):
    """ The base class for all MouseModes.

    MouseModes have the following attributes:

    * Icon : QIcon object
    * action_text : The action title (used in some menus)
    * tool_tip : string giving the tool itp
    * shortcut : Keyboard shortcut to toggle the mode
    * _press_callback : Callback method that will be called
      whenever a MouseMode processes a mouse press event
    * _move_callback : Same as above, for move events
    * _release_callback : Same as above, for release events

    The _callback hooks are called with the MouseMode as it's only
    argument
    """
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
    """ Defines a Rectangular ROI, accessible via the roi() method"""
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
    """ Defines a Circular ROI, accessible via the roi() method"""
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
    """ Defines a Polygonal ROI, accessible via the roi() method"""
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

class ContrastMode(MouseMode):
    """Uses right mouse button drags to set bias and contrast, ala DS9

    The horizontal position of the mouse sets the bias, the vertical
    position sets the contrast. The get_scaling method converts
    this information into scaling information for a particular data set
    """
    def __init__(self, *args, **kwargs):
        super(ContrastMode, self).__init__(*args, **kwargs)
        self.icon = QIcon(':icons/glue_contrast.png')
        self.mode_id = 'Contrast'
        self.action_text = 'Contrast'
        self.tool_tip = 'Adjust the bias/contrast'
        self.shortcut = 'B'

        self.bias = 0.5
        self.contrast = 0.5

    def get_scaling(self, data):
        """ Return the intensity values to set as the darkest and
        lightest color, given the bias and contrast.

        Parameters
        ----------
        data : ndarray. Raw intensities to scale

        Returns
        -------
        tuple of lo,hi : the intensity values to set as darkest/brightest
        """
        lo = np.nanmin(data)
        hi = np.nanmax(data)
        ra = hi - lo
        bias = lo + ra * self.bias
        vmin = bias - ra * self.contrast
        vmax = bias + ra * self.contrast
        return vmin, vmax

    def move(self, event):
        """ MoveEvent. Update bias and contrast on Right Mouse button drag """
        if event.button != 3: # RMB drag only
            return
        x, y = event.x, event.y
        dx, dy = self._axes.figure.canvas.get_width_height()
        x = 1.0 * x / dx
        y = 1.0 * y / dy
        theta = np.pi - max(min(y, 1), 0) * np.pi

        self.bias = x
        self.contrast = np.tan(theta)

        super(ContrastMode, self).move(event)
