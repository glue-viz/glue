import os

import matplotlib
from matplotlib.backends.backend_qt4 import NavigationToolbar2QT
from ..external.qt import QtCore, QtGui
from ..external.qt.QtGui import QIcon, QMenu
from ..external.qt.QtCore import Qt

from . import glue_qt_resources  # pylint: disable=W0611


class GlueToolbar(NavigationToolbar2QT):

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
        self.basedir = None
        NavigationToolbar2QT.__init__(self, canvas, frame)
        if name is not None:
            self.setWindowTitle(name)
        self.setIconSize(QtCore.QSize(25, 25))
        self.layout().setSpacing(1)
        self.setFocusPolicy(Qt.StrongFocus)
        self._idKey = None

    def _init_toolbar(self):
        self.basedir = os.path.join(matplotlib.rcParams['datapath'], 'images')
        parent = self.parent()

        a = QtGui.QAction(QIcon(':icons/glue_home.png'),
                          'Home', parent)
        a.triggered.connect(self.home)
        a.setToolTip('Reset original zoom')
        a.setShortcut('H')
        a.setShortcutContext(Qt.WidgetShortcut)
        parent.addAction(a)
        self.buttons['HOME'] = a
        self.addAction(a)

        a = QtGui.QAction(QIcon(':icons/glue_filesave.png'),
                          'Save', parent)
        a.triggered.connect(self.save_figure)
        a.setToolTip('Save the figure')
        a.setShortcut('Ctrl+Shift+S')
        parent.addAction(a)
        self.buttons['SAVE'] = a
        self.addAction(a)

        a = QtGui.QAction(QIcon(':icons/glue_back.png'),
                          'Back', parent)
        a.triggered.connect(self.back)
        parent.addAction(a)
        self.addAction(a)
        self.buttons['BACK'] = a
        a.setToolTip('Back to previous view')

        a = QtGui.QAction(QIcon(':icons/glue_forward.png'),
                          'Forward', parent)
        a.triggered.connect(self.forward)
        a.setToolTip('Forward to next view')
        parent.addAction(a)
        self.buttons['FORWARD'] = a
        self.addAction(a)

        a = QtGui.QAction(QIcon(':icons/glue_move.png'),
                          'Pan', parent)
        a.triggered.connect(self.pan)
        a.setToolTip('Pan axes with left mouse, zoom with right')
        a.setCheckable(True)
        a.setShortcut('M')
        a.setShortcutContext(Qt.WidgetShortcut)
        parent.addAction(a)
        self.addAction(a)
        self.buttons['PAN'] = a

        a = QtGui.QAction(QIcon(':icons/glue_zoom_to_rect.png'),
                          'Zoom', parent)
        a.triggered.connect(self.zoom)
        a.setToolTip('Zoom to rectangle')
        a.setShortcut('Z')
        a.setShortcutContext(Qt.WidgetShortcut)
        a.setCheckable(True)
        parent.addAction(a)
        self.addAction(a)
        self.buttons['ZOOM'] = a

        #self.adj_window = None

    def zoom(self, *args):
        self._deactivate_custom_modes()
        super(GlueToolbar, self).zoom(self, *args)
        self._update_buttons_checked()

    def pan(self, *args):
        self._deactivate_custom_modes()
        super(GlueToolbar, self).pan(self, *args)
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
        parent = self.parent()

        def toggle():
            self._custom_mode(mode)

        def enable():
            #turn on if not
            if self._active != mode.mode_id:
                self._custom_mode(mode)

        action = QtGui.QAction(mode.icon, mode.action_text, parent)
        action.triggered.connect(toggle)
        parent.addAction(action)

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
            menu.triggered.connect(enable)

        self.addAction(action)

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
        else:
            self.canvas.widgetlock.release(self)

        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(None)

        self.set_message(self.mode)
        self._update_buttons_checked()

    def _update_buttons_checked(self):
        for mode in self.buttons:
            self.buttons[mode].setChecked(self._active == mode)

    def set_message(self, s):
        self.emit(QtCore.SIGNAL("message"), s)
        parent = self.parent()
        if parent is None:
            return
        sb = parent.statusBar()
        if sb is None:
            return
        sb.showMessage(s.replace(', ', '\n'))
