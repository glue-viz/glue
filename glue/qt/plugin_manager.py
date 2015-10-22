from __future__ import absolute_import, division, print_function

from ..external.qt import QtGui, QtCore
from .._plugin_helpers import PluginConfig

from .qtutil import load_ui

__all__ = ["QtPluginManager"]


class QtPluginManager(object):

    def __init__(self):

        self.ui = load_ui('plugin_manager.ui', None)

        self.ui.cancel.clicked.connect(self.reject)
        self.ui.confirm.clicked.connect(self.finalize)

        self._checkboxes = {}

        self.update_list()

    def clear(self):
        self._checkboxes.clear()
        self.ui.tree.clear()

    def update_list(self):

        self.clear()

        config = PluginConfig.load()

        for plugin in sorted(config.plugins):
            check = QtGui.QTreeWidgetItem(self.ui.tree.invisibleRootItem(),
                                          ["", plugin])
            check.setFlags(check.flags() | QtCore.Qt.ItemIsUserCheckable)
            if config.plugins[plugin]:
                check.setCheckState(0, QtCore.Qt.Checked)
            else:
                check.setCheckState(0, QtCore.Qt.Unchecked)
            self._checkboxes[plugin] = check

        self.ui.tree.resizeColumnToContents(0)
        self.ui.tree.resizeColumnToContents(1)

    def reject(self):
        self.ui.reject()

    def finalize(self):

        config = PluginConfig.load()

        for name in self._checkboxes:
            config.plugins[name] = self._checkboxes[name].checkState(0) > 0

        try:
            config.save()
        except Exception:
            import traceback
            detail = str(traceback.format_exc())
            from glue.utils.qt import QMessageBoxPatched as QMessageBox
            message = QMessageBox(QMessageBox.Critical,
                                  "Error",
                                  "Could not save plugin configuration")
            message.setDetailedText(detail)
            message.exec_()
            return

        self.ui.accept()
        

