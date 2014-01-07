from __future__ import absolute_import, division, print_function

import os
import matplotlib
from matplotlib.backends.backend_qt4 import NavigationToolbar2QT
from ..external.qt import QtCore, QtGui
from ..external.qt.QtGui import QMenu
from ..external.qt.QtCore import Qt, Signal
from ..core.callback_property import add_callback
from .qtutil import get_icon, nonpartial


class GlueToolbar(NavigationToolbar2QT):
    pan_begin = Signal()
    pan_end = Signal()
    mode_activated = Signal()
    mode_deactivated = Signal()

    def __init__(self, canvas, frame, name=None):
        """ Create a new toolbar object

        Parameters
        ----------
        data_collection : DataCollection instance
         The data collection that this toolbar is meant to edit.
         The toolbar looks to this collection for the available subsets
         to manipulate.
        canvas : Maptloblib canvas instance
         The drawing canvas to interact with
        frame : QWidget
         The QT frame that the canvas is embedded within.
        """
        self.buttons = {}
        self.__active = None
        self.basedir = None
        NavigationToolbar2QT.__init__(self, canvas, frame)
        if name is not None:
            self.setWindowTitle(name)
        self.setIconSize(QtCore.QSize(25, 25))
        self.layout().setSpacing(1)
        self.setFocusPolicy(Qt.StrongFocus)
        self._idKey = None

        # pyside is prone to segfaults if slots hold the only
        # reference to a signal, so we hold an extra reference
        # see https://bugreports.qt-project.org/browse/PYSIDE-88
        self.__signals = []

    def _init_toolbar(self):
        self.basedir = os.path.join(matplotlib.rcParams['datapath'], 'images')
        parent = QtGui.QToolBar.parent(self)

        a = QtGui.QAction(get_icon('glue_home'),
                          'Home', parent)
        a.triggered.connect(nonpartial(self.home))
        a.setToolTip('Reset original zoom')
        a.setShortcut('H')
        a.setShortcutContext(Qt.WidgetShortcut)
        parent.addAction(a)
        self.buttons['HOME'] = a
        self.addAction(a)

        a = QtGui.QAction(get_icon('glue_filesave'),
                          'Save', parent)
        a.triggered.connect(nonpartial(self.save_figure))
        a.setToolTip('Save the figure')
        a.setShortcut('Ctrl+Shift+S')
        parent.addAction(a)
        self.buttons['SAVE'] = a
        self.addAction(a)

        a = QtGui.QAction(get_icon('glue_back'),
                          'Back', parent)
        a.triggered.connect(nonpartial(self.back))
        parent.addAction(a)
        self.addAction(a)
        self.buttons['BACK'] = a
        a.setToolTip('Back to previous view')

        a = QtGui.QAction(get_icon('glue_forward'),
                          'Forward', parent)
        a.triggered.connect(nonpartial(self.forward))
        a.setToolTip('Forward to next view')
        parent.addAction(a)
        self.buttons['FORWARD'] = a
        self.addAction(a)

        a = QtGui.QAction(get_icon('glue_move'),
                          'Pan', parent)
        a.triggered.connect(nonpartial(self.pan))
        a.setToolTip('Pan axes with left mouse, zoom with right')
        a.setCheckable(True)
        a.setShortcut('M')
        a.setShortcutContext(Qt.WidgetShortcut)
        parent.addAction(a)
        self.addAction(a)
        self.buttons['PAN'] = a

        a = QtGui.QAction(get_icon('glue_zoom_to_rect'),
                          'Zoom', parent)
        a.triggered.connect(nonpartial(self.zoom))
        a.setToolTip('Zoom to rectangle')
        a.setShortcut('Z')
        a.setShortcutContext(Qt.WidgetShortcut)
        a.setCheckable(True)
        parent.addAction(a)
        self.addAction(a)
        self.buttons['ZOOM'] = a

        #self.adj_window = None

    @property
    def _active(self):
        return self.__active

    @_active.setter
    def _active(self, value):
        if self.__active == value:
            return

        self.__active = value
        if value not in [None, '']:
            self.mode_activated.emit()
        else:
            self.mode_deactivated.emit()

    def home(self, *args):
        super(GlueToolbar, self).home(*args)
        self.canvas.homeButton.emit()

    def zoom(self, *args):
        self._deactivate_custom_modes()
        super(GlueToolbar, self).zoom(*args)
        self._update_buttons_checked()

    def pan(self, *args):
        self._deactivate_custom_modes()
        super(GlueToolbar, self).pan(*args)
        self._update_buttons_checked()

    def _deactivate_custom_modes(self):
        if self._idPress is not None:
            self._idPress = self.canvas.mpl_disconnect(self._idPress)
        if self._idRelease is not None:
            self._idRelease = self.canvas.mpl_disconnect(self._idRelease)
        if self._idDrag is not None:
            self._idDrag = self.canvas.mpl_disconnect(
                self._idDrag)
            self._idDrag = self.canvas.mpl_connect('motion_notify_event',
                                                   self.mouse_move)
        if self._idKey is not None:
            self._idKey = self.canvas.mpl_disconnect(self._idKey)

        self.mode = ''

    def add_mode(self, mode):
        parent = QtGui.QToolBar.parent(self)

        def toggle():
            self._custom_mode(mode)

        def enable():
            # turn on if not
            if self._active != mode.mode_id:
                self._custom_mode(mode)

        action = QtGui.QAction(mode.icon, mode.action_text, parent)
        action.triggered.connect(nonpartial(toggle))
        parent.addAction(action)

        self.__signals.extend([toggle, enable])

        if mode.shortcut is not None:
            action.setShortcut(mode.shortcut)
            action.setShortcutContext(Qt.WidgetShortcut)

        action.setToolTip(mode.tool_tip)
        action.setCheckable(True)
        self.buttons[mode.mode_id] = action

        menu_actions = mode.menu_actions()
        if len(menu_actions) > 0:
            menu = QMenu(self)
            for ma in mode.menu_actions():
                ma.setParent(self)
                menu.addAction(ma)
            action.setMenu(menu)
            menu.triggered.connect(nonpartial(enable))

        self.addAction(action)

        # bind action status to mode.enabled
        def toggle(state):
            action.setVisible(state)
            action.setEnabled(state)
        add_callback(mode, 'enabled', toggle)

        return action

    def set_mode(self, mode):
        if self._active != mode.mode_id:
            self._custom_mode(mode)

    def _custom_mode(self, mode):
        if self._active == mode.mode_id:
            self._active = None
        else:
            self._active = mode.mode_id

        self._deactivate_custom_modes()

        if self._active:
            self._idPress = self.canvas.mpl_connect(
                'button_press_event', mode.press)
            self._idDrag = self.canvas.mpl_connect(
                'motion_notify_event', mode.move)
            self._idRelease = self.canvas.mpl_connect(
                'button_release_event', mode.release)
            self._idKey = self.canvas.mpl_connect(
                'key_press_event', mode.key)
            self.mode = mode.action_text
            self.canvas.widgetlock(self)
            mode.activate()

            # allows grabbing of key events before clicking on plot
            # XXX qt specific syntax here
            try:
                self.canvas.setFocus()
            except AttributeError:
                pass
        else:
            self.canvas.widgetlock.release(self)

        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(None)

        self.set_message(self.mode)
        self._update_buttons_checked()

    def press_pan(self, event):
        self.pan_begin.emit()
        super(GlueToolbar, self).press_pan(event)

    def release_pan(self, event):
        self.pan_end.emit()
        super(GlueToolbar, self).release_pan(event)

    def _update_buttons_checked(self):
        for mode in self.buttons:
            self.buttons[mode].setChecked(self._active == mode)

    def set_message(self, s):
        self.emit(QtCore.SIGNAL("message"), s)
        parent = QtGui.QToolBar.parent(self)
        if parent is None:
            return
        sb = parent.statusBar()
        if sb is None:
            return
        sb.showMessage(s.replace(', ', '\n'))
