from __future__ import absolute_import, division, print_function

from qtpy import QtWidgets
from glue.core.edit_subset_mode import (EditSubsetMode, OrMode, AndNotMode,
                                        AndMode, XorMode, ReplaceMode)
from glue.app.qt.actions import action
from glue.utils import nonpartial


def set_mode(mode):
    edit_mode = EditSubsetMode()
    edit_mode.mode = mode


class EditSubsetModeToolBar(QtWidgets.QToolBar):

    def __init__(self, title="Subset Update Mode", parent=None):
        super(EditSubsetModeToolBar, self).__init__(title, parent)
        self._group = QtWidgets.QActionGroup(self)
        self._modes = {}
        self._add_actions()
        self._modes[EditSubsetMode().mode].trigger()
        self._backup_mode = None

    def _make_mode(self, name, tip, icon, mode):
        a = action(name, self, tip, icon)
        a.setCheckable(True)
        a.triggered.connect(nonpartial(set_mode, mode))
        self._group.addAction(a)
        self.addAction(a)
        self._modes[mode] = a
        label = name.split()[0].lower().replace('&', '')
        self._modes[label] = mode

    def _add_actions(self):
        self._make_mode("&Replace Mode", "Replace selection",
                        'glue_replace', ReplaceMode)
        self._make_mode("&Or Mode", "Add to selection",
                        'glue_or', OrMode)
        self._make_mode("&And Mode", "Set selection as intersection",
                        'glue_and', AndMode)
        self._make_mode("&Xor Mode", "Set selection as exclusive intersection",
                        'glue_xor', XorMode)
        self._make_mode("&Not Mode", "Remove from selection",
                        'glue_andnot', AndNotMode)

    def set_mode(self, mode):
        """Temporarily set the edit mode to mode
        :param mode: Name of the mode (Or, Not, And, Xor, Replace)
        :type mode: str
        """
        try:
            mode = self._modes[mode]  # label to mode class
        except KeyError:
            raise KeyError("Unrecognized mode: %s" % mode)

        self._backup_mode = self._backup_mode or EditSubsetMode().mode
        self._modes[mode].trigger()  # mode class to action

    def unset_mode(self):
        """Restore the mode to the state before set_mode was called"""
        mode = self._backup_mode
        self._backup_mode = None
        if mode:
            self._modes[mode].trigger()
