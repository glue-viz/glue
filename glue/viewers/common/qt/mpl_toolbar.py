from __future__ import absolute_import, division, print_function

from qtpy import QtCore
from qtpy import PYQT5

from glue.icons.qt import get_icon
from glue.utils import nonpartial
from glue.viewers.common.qt.mode import CheckableMode, NonCheckableMode
from glue.viewers.common.qt.mouse_mode import MouseMode
from glue.viewers.common.qt.toolbar import BasicToolbar

if PYQT5:
    from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
else:
    from matplotlib.backends.backend_qt4 import NavigationToolbar2QT


class HomeMode(NonCheckableMode):

    def __init__(self, toolbar=None):
        super(HomeMode, self).__init__()
        self.mode_id = 'HOME'
        self.icon = get_icon('glue_home')
        self.action_text = 'Home'
        self.tool_tip = 'Reset original zoom'
        self.shortcut = 'H'
        self.checkable = False
        self.toolbar = toolbar

    def activate(self):
        self.toolbar.home()


class SaveMode(NonCheckableMode):

    def __init__(self, toolbar=None):
        super(SaveMode, self).__init__()
        self.mode_id = 'SAVE'
        self.icon = get_icon('glue_filesave')
        self.action_text = 'Save'
        self.tool_tip = 'Save the figure'
        self.shortcut = 'Ctrl+Shift+S'
        self.toolbar = toolbar

    def activate(self):
        self.toolbar.save_figure()


class BackMode(NonCheckableMode):

    def __init__(self, toolbar=None):
        super(BackMode, self).__init__()
        self.mode_id = 'BACK'
        self.icon = get_icon('glue_back')
        self.action_text = 'Back'
        self.tool_tip = 'Back to previous view'
        self.toolbar = toolbar

    def activate(self):
        self.toolbar.back()


class ForwardMode(NonCheckableMode):

    def __init__(self, toolbar=None):
        super(ForwardMode, self).__init__()
        self.mode_id = 'FORWARD'
        self.icon = get_icon('glue_forward')
        self.action_text = 'Forward'
        self.tool_tip = 'Forward to next view'
        self.toolbar = toolbar

    def activate(self):
        self.toolbar.forward()


class PanMode(CheckableMode):

    def __init__(self, toolbar=None):
        super(PanMode, self).__init__()
        self.mode_id = 'PAN'
        self.icon = get_icon('glue_move')
        self.action_text = 'Pan'
        self.tool_tip = 'Pan axes with left mouse, zoom with right'
        self.shortcut = 'M'
        self.toolbar = toolbar

    def activate(self):
        self.toolbar.pan()

    def deactivate(self):
        self.toolbar.pan()


class ZoomMode(CheckableMode):

    def __init__(self, toolbar=None):
        super(ZoomMode, self).__init__()
        self.mode_id = 'ZOOM'
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

    def __init__(self, canvas, parent, name=None):

        self.canvas = canvas

        # Set up virtual Matplotlib navigation toolbar (don't show it)
        self._mpl_nav = NavigationToolbar2QT(canvas, parent)
        self._mpl_nav.hide()

        BasicToolbar.__init__(self, parent)

    def setup_default_modes(self):

        # Set up default Matplotlib Modes - this gets called by the __init__
        # call to the parent class above.

        home_mode = HomeMode(toolbar=self._mpl_nav)
        self.add_mode(home_mode)

        save_mode = SaveMode(toolbar=self._mpl_nav)
        self.add_mode(save_mode)

        back_mode = BackMode(toolbar=self._mpl_nav)
        self.add_mode(back_mode)

        forward_mode = ForwardMode(toolbar=self._mpl_nav)
        self.add_mode(forward_mode)

        pan_mode = PanMode(toolbar=self._mpl_nav)
        self.add_mode(pan_mode)

        zoom_mode = ZoomMode(toolbar=self._mpl_nav)
        self.add_mode(zoom_mode)

        self._connections = []

    def activate_mode(self, mode):
        if isinstance(mode, MouseMode):
            self._connections.append(self.canvas.mpl_connect('button_press_event', mode.press))
            self._connections.append(self.canvas.mpl_connect('motion_notify_event', mode.move))
            self._connections.append(self.canvas.mpl_connect('button_release_event', mode.release))
            self._connections.append(self.canvas.mpl_connect('key_press_event', mode.key))
        super(MatplotlibViewerToolbar, self).activate_mode(mode)

    def deactivate_mode(self, mode):
        for connection in self._connections:
            self.canvas.mpl_disconnect(connection)
        self._connections = []
        super(MatplotlibViewerToolbar, self).deactivate_mode(mode)
