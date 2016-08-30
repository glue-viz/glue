from __future__ import absolute_import, division, print_function

import os
import warnings

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

from glue.external import six
from glue.core.callback_property import add_callback
from glue.utils import nonpartial
from glue.viewers.common.qt.tool import CheckableTool, Tool
from glue.config import viewer_tool
from glue.icons.qt import get_icon

__all__ = ['BasicToolbar']


class BasicToolbar(QtWidgets.QToolBar):

    mode_activated = QtCore.Signal()
    mode_deactivated = QtCore.Signal()

    def __init__(self, parent):
        """
        Create a new toolbar object
        """

        super(BasicToolbar, self).__init__(parent=parent)

        self.buttons = {}
        self.tools = {}
        self.setIconSize(QtCore.QSize(25, 25))
        self.layout().setSpacing(1)
        self.setFocusPolicy(Qt.StrongFocus)
        self._mode = None

        self.setup_default_modes()

    def setup_default_modes(self):
        pass

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_mode):

        old_mode = self._mode

        # If the mode is as before, we don't need to do anything
        if old_mode is new_mode:
            return

        # Otheriwse, if the mode changes, then we need to disable the previous
        # mode...
        if old_mode is not None:
            self.deactivate_mode(old_mode)
            if isinstance(old_mode, CheckableTool):
                button = self.buttons[old_mode.tool_id]
                if button.isChecked():
                    button.blockSignals(True)
                    button.setChecked(False)
                    button.blockSignals(False)

        # ... and enable the new one
        if new_mode is not None:
            self.activate_mode(new_mode)
            if isinstance(new_mode, CheckableTool):
                button = self.buttons[new_mode.tool_id]
                if button.isChecked():
                    button.blockSignals(True)
                    button.setChecked(True)
                    button.blockSignals(False)

        if isinstance(new_mode, CheckableTool):
            self._mode = new_mode
            self.mode_activated.emit()
        else:
            self._mode = None
            self.mode_deactivated.emit()

    def activate_mode(self, mode):
        mode.activate()

    def deactivate_mode(self, mode):
        if isinstance(mode, CheckableTool):
            mode.deactivate()

    def add_tool(self, mode):

        parent = QtWidgets.QToolBar.parent(self)

        if isinstance(mode.icon, six.string_types):
            if os.path.exists(mode.icon):
                icon = QtGui.QIcon(mode.icon)
            else:
                icon = get_icon(mode.icon)
        else:
            icon = mode.icon

        action = QtWidgets.QAction(icon, mode.action_text, parent)

        def toggle(checked):
            if checked:
                self.mode = mode
            else:
                self.mode = None

        def trigger(checked):
            self.mode = mode

        parent.addAction(action)

        if isinstance(mode, CheckableTool):
            action.toggled.connect(toggle)
        else:
            action.triggered.connect(trigger)

        shortcut = None

        if mode.shortcut is not None:

            # Make sure that the keyboard shortcut is unique
            for m in self.tools.values():
                if mode.shortcut == m.shortcut:
                    warnings.warn("Tools '{0}' and '{1}' have the same shortcut "
                                  "('{2}'). Ignoring shortcut for "
                                  "'{1}'".format(m.tool_id, mode.tool_id, mode.shortcut))
                    break
            else:
                shortcut = mode.shortcut
                action.setShortcut(mode.shortcut)
                action.setShortcutContext(Qt.WidgetShortcut)


        if shortcut is None:
            action.setToolTip(mode.tool_tip)
        else:
            action.setToolTip(mode.tool_tip + " [shortcut: {0}]".format(shortcut))

        action.setCheckable(isinstance(mode, CheckableTool))
        self.buttons[mode.tool_id] = action

        menu_actions = mode.menu_actions()
        if len(menu_actions) > 0:
            menu = QtWidgets.QMenu(self)
            for ma in mode.menu_actions():
                ma.setParent(self)
                menu.addAction(ma)
            action.setMenu(menu)
            menu.triggered.connect(trigger)

        self.addAction(action)

        self.tools[mode.tool_id] = mode

        return action
