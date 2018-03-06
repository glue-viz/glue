from __future__ import absolute_import, division, print_function

import os
import sys
from contextlib import contextmanager

from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt
from qtpy.uic import loadUi
from glue.utils.qt import get_text

__all__ = ['update_combobox', 'GlueTabBar', 'load_ui', 'process_dialog',
           'combo_as_string', 'qurl_to_path']


def update_combobox(combo, labeldata, default_index=0):
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
    index = None
    for i, (label, data) in enumerate(labeldata):
        combo.addItem(label, userData=data)
        if data is current or data == current:
            index = i

    if default_index < 0:
        default_index = combo.count() + default_index

    if index is None:
        index = min(default_index, combo.count() - 1)
    combo.setCurrentIndex(index)

    combo.blockSignals(False)

    # We need to force emit this, otherwise if the index happens to be the
    # same as before, even if the data is different, callbacks won't be
    # called. So we block the signals until just before now then always call
    # callback manually.
    combo.currentIndexChanged.emit(index)


class GlueTabBar(QtWidgets.QTabBar):

    def __init__(self, *args, **kwargs):
        super(GlueTabBar, self).__init__(*args, **kwargs)

    def choose_rename_tab(self, index=None):
        """
        Prompt user to rename a tab

        Parameters
        ----------
        index : int
            Index of tab to edit. Defaults to current index
        """
        index = index or self.currentIndex()
        label = get_text("New Tab Label")
        if not label:
            return
        self.rename_tab(index, label)

    def rename_tab(self, index, label):
        """
        Updates the name used for given tab

        Parameters
        ----------
        index : int
            Index of tab to edit. Defaults to current index
        label : str
            New label to use for this tab
        """
        self.setTabText(index, label)

    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        index = self.tabAt(event.pos())
        if index >= 0:
            self.choose_rename_tab(index)


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
    w : QtWidgets.QWidget
        The new widget
    """

    if directory is not None:
        full_path = os.path.join(directory, path)
    else:
        full_path = os.path.abspath(path)

    if not os.path.exists(full_path) and 'site-packages.zip' in full_path:
        # Workaround for Mac app
        full_path = os.path.join(full_path.replace('site-packages.zip', 'glue'))

    return loadUi(full_path, parent)


@contextmanager
def process_dialog(delay=0, accept=False, reject=False, function=None):
    """
    Context manager to automatically capture the active dialog and carry out
    certain actions.

    Note that only one of ``accept``, ``reject``, or ``function`` should be
    specified.

    Parameters
    ----------
    delay : int, optional
        The delay in ms before acting on the dialog (since it may not yet exist
        when the context manager is called).
    accept : bool, optional
        If `True`, accept the dialog after the specified delay.
    reject : bool, optional
        If `False`, reject the dialog after the specified delay
    function : func, optional
        For more complex user actions, specify a function that takes the dialog
        as the first and only argument.
    """

    def _accept(dialog):
        dialog.accept()

    def _reject(dialog):
        dialog.reject()

    n_args = sum((accept, reject, function is not None))

    if n_args > 1:
        raise ValueError("Only one of ``accept``, ``reject``, or "
                         "``function`` should be specified")
    elif n_args == 0:
        raise ValueError("One of ``accept``, ``reject``, or "
                         "``function`` should be specified")

    if accept:
        function = _accept
    elif reject:
        function = _reject

    def wrapper():
        from glue.utils.qt import get_qapp
        app = get_qapp()
        # Make sure that any window/dialog that needs to be shown is shown
        app.processEvents()
        dialog = app.activeWindow()
        function(dialog)

    timer = QtCore.QTimer()
    timer.setInterval(delay)
    timer.setSingleShot(True)
    timer.timeout.connect(wrapper)
    timer.start()

    yield


def combo_as_string(combo):
    """
    Return the text labels of a combo box as a string to make it easier to
    check the content of a combo box in tests.
    """
    items = [combo.itemText(i) for i in range(combo.count())]
    return ":".join(items)


def qurl_to_path(url):
    """
    Convert a local QUrl to a normal path
    """

    # Get path to file
    path = url.path()

    # Workaround for a Qt bug that causes paths to start with a /
    # on Windows: https://bugreports.qt.io/browse/QTBUG-46417
    if sys.platform.startswith('win'):
        if path.startswith('/') and path[2] == ':':
            path = path[1:]

    return path
