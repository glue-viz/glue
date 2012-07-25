from PyQt4 import QtCore, QtGui
from functools import partial

from ...core.edit_subset_mode import (EditSubsetMode, OrMode, AndNotMode,
                                    AndMode, XorMode, SpawnMode, ReplaceMode)
from ..actions import act


def set_mode(mode):
    edit_mode = EditSubsetMode()
    edit_mode.mode = mode


class EditSubsetModeToolBar(QtGui.QToolBar):

    def __init__(self, title="Subset Update Mode", parent=None):
        super(EditSubsetModeToolBar, self).__init__(title, parent)
        self._group = QtGui.QActionGroup(self)
        self._modes = {}
        self._add_actions()
        self._modes[EditSubsetMode().mode].trigger()

    def _make_mode(self, name, tip, icon, mode):
        a = act(name, self, tip, icon)
        a.setCheckable(True)
        a.triggered.connect(partial(set_mode, mode))
        self._group.addAction(a)
        self.addAction(a)
        self._modes[mode] = a

    def _add_actions(self):
        self._make_mode("Replace Mode", "Replace selection",
                        'glue_replace.png', ReplaceMode)
        self._make_mode("Or Mode", "Add to selection",
                        'glue_or.png', OrMode)
        self._make_mode("Not Mode", "Remove from selection",
                        'glue_andnot.png', AndNotMode)
        self._make_mode("And Mode", "Set selection as intersection",
                        'glue_and.png', AndMode)
        self._make_mode("Xor Mode", "Set selection as exclusive intersection",
                        'glue_xor.png', XorMode)
        self._make_mode("Spawn Mode", "Spawn new selection",
                        'glue_spawn.png', SpawnMode)
