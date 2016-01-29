from __future__ import absolute_import, division, print_function

import os

from glue.external.qt import QtGui 
from glue.external.qt.QtCore import Qt
from glue.utils.qt import get_text

__all__ = ['update_combobox', 'GlueTabBar', 'load_ui', 'CUSTOM_QWIDGETS']


def update_combobox(combo, labeldata):
    """
    Redefine the items in a QComboBox

    Parameters
    ----------
    widget : QComboBox
       The widget to update
    labeldata : sequence of N (label, data) tuples
       The combobox will contain N items with the appropriate
       labels, and data set as the userData

    Returns
    -------
    combo : QComboBox
        The updated input

    Notes
    -----

    If the current userData in the combo box matches
    any of labeldata, that selection will be retained.
    Otherwise, the first item will be selected.

    Signals are disabled while the combo box is updated

    The QComboBox is modified inplace
    """

    combo.blockSignals(True)
    idx = combo.currentIndex()
    if idx >= 0:
        current = combo.itemData(idx)
    else:
        current = None

    combo.clear()
    index = 0
    for i, (label, data) in enumerate(labeldata):
        combo.addItem(label, userData=data)
        if data is current:
            index = i
    combo.blockSignals(False)
    combo.setCurrentIndex(index)

    # We need to force emit this, otherwise if the index happens to be the
    # same as before, even if the data is different, callbacks won't be
    # called.
    if idx == index or idx == -1:
        combo.currentIndexChanged.emit(index)


class GlueTabBar(QtGui.QTabBar):

    def __init__(self, *args, **kwargs):
        super(GlueTabBar, self).__init__(*args, **kwargs)

    def rename_tab(self, index=None):
        """ Prompt user to rename a tab
        :param index: integer. Index of tab to edit. Defaults to current index
        """
        index = index or self.currentIndex()
        label = get_text("New Tab Label")
        if not label:
            return
        self.setTabText(index, label)

    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        index = self.tabAt(event.pos())
        if index >= 0:
            self.rename_tab(index)


CUSTOM_QWIDGETS = []

def load_ui(path, parent=None, directory=None):
    """
    Load a .ui file

    Parameters
    ----------
    path : str
        Name of ui file to load

    parent : QObject
        Object to use as the parent of this widget

    Returns
    -------
    w : QtGui.QWidget
        The new widget
    """

    if directory is not None:
        full_path = os.path.join(directory, path)
    else:
        full_path = os.path.abspath(path)

    if not os.path.exists(full_path) and 'site-packages.zip' in full_path:
        # Workaround for Mac app
        full_path = os.path.join(full_path.replace('site-packages.zip', 'glue'))

    from glue.external.qt import load_ui 
    return load_ui(full_path, parent, custom_widgets=CUSTOM_QWIDGETS)