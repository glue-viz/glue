from __future__ import absolute_import, division, print_function

import os
import warnings

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

from glue.external import six
from glue.core.callback_property import add_callback
from glue.viewers.common.qt.tool import CheckableTool, DropdownTool
from glue.icons.qt import get_icon

__all__ = ['BasicToolbar']


class BasicToolbar(QtWidgets.QToolBar):

    tool_activated = QtCore.Signal()
    tool_deactivated = QtCore.Signal()

    def __init__(self, parent, default_mouse_mode_cls=None):
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
        self._default_mouse_mode_cls = default_mouse_mode_cls
        self._default_mouse_mode = None
        self.setup_default_modes()

    def setup_default_modes(self):
        if self._default_mouse_mode_cls is not None:
            self._default_mouse_mode = self._default_mouse_mode_cls(self.parent())
            self._default_mouse_mode.activate()

    @property
    def active_tool(self):
        return self._active_tool

    @active_tool.setter
    def active_tool(self, new_tool):

        if isinstance(new_tool, six.string_types):
            if new_tool in self.tools:
                new_tool = self.tools[new_tool]
            else:
                raise ValueError("Unrecognized tool '{0}', should be one of {1}"
                                 .format(new_tool, ", ".join(sorted(self.tools))))

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

        # We need to then set that no tool is set so that if the next tool
        # opens a viewer that needs to check whether a tool is active, we
        # know that it isn't.
        self._active_tool = None

        # ... and enable the new one
        if new_tool is not None:
            self.activate_tool(new_tool)
            if isinstance(new_tool, CheckableTool):
                button = self.actions[new_tool.tool_id]
                if not button.isChecked():
                    button.blockSignals(True)
                    button.setChecked(True)
                    button.blockSignals(False)

        if isinstance(new_tool, CheckableTool):
            self._active_tool = new_tool
            self.parent().set_status(new_tool.status_tip)
            self.tool_activated.emit()
        else:
            self.parent().set_status('')
            self.tool_deactivated.emit()

    def activate_tool(self, tool):
        if isinstance(tool, CheckableTool) and self._default_mouse_mode is not None:
            self._default_mouse_mode.deactivate()
        tool.activate()

    def deactivate_tool(self, tool):
        if isinstance(tool, CheckableTool):
            tool.deactivate()
        if self._default_mouse_mode is not None:
            self._default_mouse_mode.activate()

    def _make_action(self, tool, menu=None):

        parent = QtWidgets.QToolBar.parent(self)

        if isinstance(tool.icon, six.string_types):
            if os.path.exists(tool.icon):
                icon = QtGui.QIcon(tool.icon)
            else:
                icon = get_icon(tool.icon)
        else:
            icon = tool.icon

        if isinstance(tool, DropdownTool):
            # We use a QToolButton here explicitly so that we can make sure
            # that the whole button opens the pop-up.
            button = QtWidgets.QToolButton()
            if tool.action_text:
                button.setText(tool.action_text)
            if icon:
                button.setIcon(icon)
            button.setPopupMode(button.InstantPopup)
            button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            action = self.addWidget(button)
            if menu:
                button.setMenu(menu)
            button.clicked.connect(button.showMenu)
        else:
            if icon:
                action = QtWidgets.QAction(icon, tool.action_text, parent)
            else:
                action = QtWidgets.QAction(tool.action_text, parent)

        def toggle(checked):
            if checked:
                self.active_tool = tool
            else:
                self.active_tool = None

        def trigger():
            self.active_tool = tool

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

        return action

    def add_tool(self, tool):

        if isinstance(tool, DropdownTool) and len(tool.subtools) > 0:
            menu = QtWidgets.QMenu(self)
            for t in tool.subtools:
                action = self._make_action(t)
                menu.addAction(action)
        elif len(tool.menu_actions()) > 0:
            menu = QtWidgets.QMenu(self)
            for ma in tool.menu_actions():
                ma.setParent(self)
                menu.addAction(ma)
        else:
            menu = None

        action = self._make_action(tool, menu=menu)

        self.addAction(action)

        self.actions[tool.tool_id] = action

        # Bind tool visibility to tool.enabled
        def toggle(state):
            action.setVisible(state)
            action.setEnabled(state)

        add_callback(tool, 'enabled', toggle)

        self.tools[tool.tool_id] = tool

        return action

    def cleanup(self):
        # We need to make sure we set _default_mouse_mode to None otherwise
        # we keep a reference to the viewer (parent) inside the mouse mode,
        # creating a circular reference.
        self._default_mouse_mode = None
        self.active_tool = None
