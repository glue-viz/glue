from __future__ import absolute_import, division, print_function

import os
import warnings

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

from glue.external import six
from glue.core.callback_property import add_callback
from glue.viewers.common.qt.tool import CheckableTool
from glue.icons.qt import get_icon

__all__ = ['BasicToolbar']


class BasicToolbar(QtWidgets.QToolBar):

    tool_activated = QtCore.Signal()
    tool_deactivated = QtCore.Signal()

    def __init__(self, parent):
        """
        Create a new toolbar object
        """

        super(BasicToolbar, self).__init__(parent=parent)

        self.actions = {}
        self.tools = {}
        self.setIconSize(QtCore.QSize(25, 25))
        self.layout().setSpacing(1)
        self.setFocusPolicy(Qt.StrongFocus)
        self._active_tool = None
        self.setup_default_modes()

    def setup_default_modes(self):
        pass

    @property
    def active_tool(self):
        return self._active_tool

    @active_tool.setter
    def active_tool(self, new_tool):

        old_tool = self._active_tool

        # If the tool is as before, we don't need to do anything
        if old_tool is new_tool:
            return

        # Otheriwse, if the tool changes, then we need to disable the previous
        # tool...
        if old_tool is not None:
            self.deactivate_tool(old_tool)
            if isinstance(old_tool, CheckableTool):
                button = self.actions[old_tool.tool_id]
                if button.isChecked():
                    button.blockSignals(True)
                    button.setChecked(False)
                    button.blockSignals(False)

        # ... and enable the new one
        if new_tool is not None:
            self.activate_tool(new_tool)
            if isinstance(new_tool, CheckableTool):
                button = self.actions[new_tool.tool_id]
                if button.isChecked():
                    button.blockSignals(True)
                    button.setChecked(True)
                    button.blockSignals(False)

        if isinstance(new_tool, CheckableTool):
            self._active_tool = new_tool
            self.parent().set_status(new_tool.status_tip)
            self.tool_activated.emit()
        else:
            self._active_tool = None
            self.parent().set_status('')
            self.tool_deactivated.emit()

    def activate_tool(self, tool):
        tool.activate()

    def deactivate_tool(self, tool):
        if isinstance(tool, CheckableTool):
            tool.deactivate()

    def add_tool(self, tool):

        parent = QtWidgets.QToolBar.parent(self)

        if isinstance(tool.icon, six.string_types):
            if os.path.exists(tool.icon):
                icon = QtGui.QIcon(tool.icon)
            else:
                icon = get_icon(tool.icon)
        else:
            icon = tool.icon

        action = QtWidgets.QAction(icon, tool.action_text, parent)

        def toggle(checked):
            if checked:
                self.active_tool = tool
            else:
                self.active_tool = None

        def trigger(checked):
            self.active_tool = tool

        parent.addAction(action)

        if isinstance(tool, CheckableTool):
            action.toggled.connect(toggle)
        else:
            action.triggered.connect(trigger)

        shortcut = None

        if tool.shortcut is not None:

            # Make sure that the keyboard shortcut is unique
            for m in self.tools.values():
                if tool.shortcut == m.shortcut:
                    warnings.warn("Tools '{0}' and '{1}' have the same shortcut "
                                  "('{2}'). Ignoring shortcut for "
                                  "'{1}'".format(m.tool_id, tool.tool_id, tool.shortcut))
                    break
            else:
                shortcut = tool.shortcut
                action.setShortcut(tool.shortcut)
                action.setShortcutContext(Qt.WidgetShortcut)

        if shortcut is None:
            action.setToolTip(tool.tool_tip)
        else:
            action.setToolTip(tool.tool_tip + " [shortcut: {0}]".format(shortcut))

        action.setCheckable(isinstance(tool, CheckableTool))
        self.actions[tool.tool_id] = action

        menu_actions = tool.menu_actions()
        if len(menu_actions) > 0:
            menu = QtWidgets.QMenu(self)
            for ma in tool.menu_actions():
                ma.setParent(self)
                menu.addAction(ma)
            action.setMenu(menu)
            menu.triggered.connect(trigger)

        self.addAction(action)

        # Bind tool visibility to tool.enabled
        def toggle(state):
            action.setVisible(state)
            action.setEnabled(state)
        add_callback(tool, 'enabled', toggle)

        self.tools[tool.tool_id] = tool

        return action
