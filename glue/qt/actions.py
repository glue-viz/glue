from __future__ import absolute_import, division, print_function

from glue.external.qt.QtGui import QAction
from glue.qt.qtutil import get_icon


def act(name, parent, tip='', icon=None, shortcut=None):
    """ Factory for making a new action """
    a = QAction(name, parent)
    a.setToolTip(tip)
    if icon:
        a.setIcon(get_icon(icon))
    if shortcut:
        a.setShortcut(shortcut)
    return a
"""
tab_new = act('New Tab',
              shortcut=QKeySequence.AddTab,
              tip='Add a new tab')

tab_tile = act("Tile",
               tip="Tile windows in the current tab")

tab_cascade = act("Cascade",
                  tip = "Cascade windows in the current tab")

window_new = act('New Window',
                 shortcut=QKeySequence.New,
                 tip='Add a new visualization window to the current tab')

subset_or = act("Union Combine",
                icon='glue_or',
                tip = 'Define a new subset as a union of selection')

subste_and = act("Intersection Combine",
                 icon="glue_and",
                 tip = 'Define a new subset as intersection of selection')

subset_xor = act("XOR Combine",
                 icon='glue_xor',
                 tip= 'Define a new subset as non-intersection of selection')

subset_not = act("Invert",
                 icon="glue_not",
                 tip="Invert current subset")

subset_copy = act("Copy subset",
                  tip="Copy the definition for the selected subset",
                  shortcut=QKeySequence.Copy)

subset_paste = act("Paste subset",
                   tip = "Replace the selected subset with clipboard",
                   shortcut=QKeySequence.Paste)

subset_new = act("New subset",
                 tip="Create a new subset for the selected data",
                 shortcut=QKeySequence.New)

subset_clear = act("Clear subset",
                   tip="Clear current selection")

subset_duplicate = act("Duplicate subset",
                       tip="Duplicate the current subset",
                       shortcut="Ctrl+D")

layer_delete = act("Delete layer",
                   shortcut=QKeySequence.Delete,
                   tip="Remove the highlighted layer")


"""
