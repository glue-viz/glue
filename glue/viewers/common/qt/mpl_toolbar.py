from __future__ import absolute_import, division, print_function

from qtpy import QtCore
from qtpy import PYQT5

from glue.icons.qt import get_icon
from glue.utils import nonpartial
from glue.viewers.common.qt.tool import CheckableTool, Tool
from glue.viewers.common.qt.mouse_mode import MouseMode
from glue.viewers.common.qt.toolbar import BasicToolbar

if PYQT5:
    from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
else:
    from matplotlib.backends.backend_qt4 import NavigationToolbar2QT

__all__ = ['HomeTool', 'SaveTool', 'BackTool', 'ForwardTool', 'PanTool',
           'ZoomTool', 'MatplotlibViewerToolbar']


class HomeTool(Tool):

    def __init__(self, viewer, toolbar=None):
        super(HomeTool, self).__init__(viewer=viewer)
        self.tool_id = 'mpl:home'
        self.icon = get_icon('glue_home')
        self.action_text = 'Home'
        self.tool_tip = 'Reset original zoom'
        self.shortcut = 'H'
        self.checkable = False
        self.toolbar = toolbar

    def activate(self):
        self.toolbar.home()


class SaveTool(Tool):

    def __init__(self, viewer, toolbar=None):
        super(SaveTool, self).__init__(viewer=viewer)
        self.tool_id = 'mpl:save'
        self.icon = get_icon('glue_filesave')
        self.action_text = 'Save'
        self.tool_tip = 'Save the figure'
        self.shortcut = 'Ctrl+Shift+S'
        self.toolbar = toolbar

    def activate(self):
        self.toolbar.save_figure()


class BackTool(Tool):

    def __init__(self, viewer, toolbar=None):
        super(BackTool, self).__init__(viewer=viewer)
        self.tool_id = 'mpl:back'
        self.icon = get_icon('glue_back')
        self.action_text = 'Back'
        self.tool_tip = 'Back to previous view'
        self.toolbar = toolbar

    def activate(self):
        self.toolbar.back()


class ForwardTool(Tool):

    def __init__(self, viewer, toolbar=None):
        super(ForwardTool, self).__init__(viewer=viewer)
        self.tool_id = 'mpl:forward'
        self.icon = get_icon('glue_forward')
        self.action_text = 'Forward'
        self.tool_tip = 'Forward to next view'
        self.toolbar = toolbar

    def activate(self):
        self.toolbar.forward()


class PanTool(CheckableTool):

    def __init__(self, viewer, toolbar=None):
        super(PanTool, self).__init__(viewer=viewer)
        self.tool_id = 'mpl:pan'
        self.icon = get_icon('glue_move')
        self.action_text = 'Pan'
        self.tool_tip = 'Pan axes with left mouse, zoom with right'
        self.shortcut = 'M'
        self.toolbar = toolbar

    def activate(self):
        self.toolbar.pan()

    def deactivate(self):
        self.toolbar.pan()


class ZoomTool(CheckableTool):

    def __init__(self, viewer, toolbar=None):
        super(ZoomTool, self).__init__(viewer=viewer)
        self.tool_id = 'mpl:zoom'
        self.icon = get_icon('glue_zoom_to_rect')
        self.action_text = 'Zoom'
        self.tool_tip = 'Zoom to rectangle'
        self.shortcut = 'Z'
        self.toolbar = toolbar

    def activate(self):
        self.toolbar.zoom()

    def deactivate(self):
        self.toolbar.zoom()


class MatplotlibViewerToolbar(BasicToolbar):

    pan_begin = QtCore.Signal()
    pan_end = QtCore.Signal()

    def __init__(self, parent):

        self.canvas = parent.central_widget.canvas

        # Set up virtual Matplotlib navigation toolbar (don't show it)
        self._mpl_nav = NavigationToolbar2QT(self.canvas, parent)
        self._mpl_nav.hide()

        BasicToolbar.__init__(self, parent)

    def setup_default_modes(self):

        # Set up default Matplotlib Tools - this gets called by the __init__
        # call to the parent class above.

        home_mode = HomeTool(self.parent(), toolbar=self._mpl_nav)
        self.add_tool(home_mode)

        save_mode = SaveTool(self.parent(), toolbar=self._mpl_nav)
        self.add_tool(save_mode)

        back_mode = BackTool(self.parent(), toolbar=self._mpl_nav)
        self.add_tool(back_mode)

        forward_mode = ForwardTool(self.parent(), toolbar=self._mpl_nav)
        self.add_tool(forward_mode)

        pan_mode = PanTool(self.parent(), toolbar=self._mpl_nav)
        self.add_tool(pan_mode)

        zoom_mode = ZoomTool(self.parent(), toolbar=self._mpl_nav)
        self.add_tool(zoom_mode)

        self._connections = []

    def activate_tool(self, mode):
        if isinstance(mode, MouseMode):
            self._connections.append(self.canvas.mpl_connect('button_press_event', mode.press))
            self._connections.append(self.canvas.mpl_connect('motion_notify_event', mode.move))
            self._connections.append(self.canvas.mpl_connect('button_release_event', mode.release))
            self._connections.append(self.canvas.mpl_connect('key_press_event', mode.key))
        super(MatplotlibViewerToolbar, self).activate_tool(mode)

    def deactivate_tool(self, mode):
        for connection in self._connections:
            self.canvas.mpl_disconnect(connection)
        self._connections = []
        super(MatplotlibViewerToolbar, self).deactivate_tool(mode)
