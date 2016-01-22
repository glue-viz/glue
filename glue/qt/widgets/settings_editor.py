from glue.external.qt import QtGui
from glue.external.qt.QtCore import Qt


class SettingsEditor(object):

    def __init__(self, app):
        w = QtGui.QTableWidget(parent=None)
        w.setColumnCount(2)
        w.setRowCount(len(list(app.settings)))
        w.setHorizontalHeaderLabels(["Setting", "Value"])
        for row, (key, value) in enumerate(app.settings):
            k = QtGui.QTableWidgetItem(key)
            v = QtGui.QTableWidgetItem(str(value))
            k.setFlags(k.flags() ^ (Qt.ItemIsEditable | Qt.ItemIsSelectable))
            w.setItem(row, 0, k)
            w.setItem(row, 1, v)
        w.sortItems(0)
        w.cellChanged.connect(self.update_setting)
        w.setWindowModality(Qt.ApplicationModal)
        w.resize(350, 340)
        w.setColumnWidth(0, 160)
        w.setColumnWidth(1, 160)
        w.setWindowTitle("Glue Settings")
        self._widget = w
        self.app = app

    def update_setting(self, row, column):
        key = self._widget.item(row, 0).text()
        value = self._widget.item(row, 1).text()
        try:
            self.app.set_setting(key, value)
        except ValueError:
            pass

        new_txt = str(self.app.get_setting(key))
        self._widget.item(row, 1).setText(new_txt)

    @property
    def widget(self):
        return self._widget
