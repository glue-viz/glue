from __future__ import absolute_import, division, print_function

from qtpy import QtCore, QtWidgets
from glue.core.edit_subset_mode import (NewMode, OrMode,
                                        AndNotMode, AndMode, XorMode,
                                        ReplaceMode)
from glue.app.qt.actions import action
from glue.utils import nonpartial
from glue.core.message import EditSubsetMessage
from glue.core.hub import HubListener
from glue.external.six import string_types


class EditSubsetModeToolBar(QtWidgets.QToolBar, HubListener):

    def __init__(self, title="Subset Update Mode", parent=None):
        super(EditSubsetModeToolBar, self).__init__(title, parent)

        spacer = QtWidgets.QWidget()
        spacer.setMinimumSize(20, 10)
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Fixed,
                             QtWidgets.QSizePolicy.Preferred)

        self.addWidget(spacer)

        self.addWidget(QtWidgets.QLabel("Selection Mode:"))
        self.setIconSize(QtCore.QSize(20, 20))
        self._group = QtWidgets.QActionGroup(self)
        self._modes = {}
        self._add_actions()
        self._edit_subset_mode = self.parent()._session.edit_subset_mode
        self._modes[self._edit_subset_mode.mode].trigger()
        self._backup_mode = None

        spacer = QtWidgets.QWidget()
        spacer.setMinimumSize(20, 10)
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Fixed,
                             QtWidgets.QSizePolicy.Preferred)

        self.addWidget(spacer)

        self.parent()._hub.subscribe(self, EditSubsetMessage, handler=self._update_mode)

    def _make_mode(self, name, tip, icon, mode):

        def set_mode(mode):
            self._edit_subset_mode.mode = mode

        a = action(name, self, tip, icon)
        a.setCheckable(True)
        a.triggered.connect(nonpartial(set_mode, mode))
        self._group.addAction(a)
        self.addAction(a)
        self._modes[mode] = a
        label = name.split()[0].lower().replace('&', '')
        self._modes[label] = mode

    def _add_actions(self):
        self._make_mode("&New Mode", "Create new selection",
                        'glue_spawn', NewMode)
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

    def _update_mode(self, message):
        self.set_mode(message.mode)

    def set_mode(self, mode):
        """Temporarily set the edit mode to mode
        :param mode: Name of the mode (Or, Not, And, Xor, Replace)
        :type mode: str
        """
        if isinstance(mode, string_types):
            try:
                mode = self._modes[mode]  # label to mode class
            except KeyError:
                raise KeyError("Unrecognized mode: %s" % mode)

        self._backup_mode = self._backup_mode or self._edit_subset_mode.mode
        self._modes[mode].trigger()  # mode class to action

    def unset_mode(self):
        """Restore the mode to the state before set_mode was called"""
        mode = self._backup_mode
        self._backup_mode = None
        if mode:
            self._modes[mode].trigger()
