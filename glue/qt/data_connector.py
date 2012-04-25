from PyQt4.QtGui import QDialog, QListWidgetItem
from PyQt4.QtCore import Qt

import glue
from glue.util import glue_components_1to1
from ui_data_connector import Ui_DataConnector
class DataConnector(QDialog):

    def __init__(self, data, parent=None):
        super(DataConnector, self).__init__(parent)
        self.ui = Ui_DataConnector()
        self.ui.setupUi(self)

        self._data = data
        self._init_widgets()
        self._links = set()
        self._connect()

    def _init_widgets(self):
        for d in self._data:
            self.ui.left_combo.addItem(d.label, userData=d)
            self.ui.right_combo.addItem(d.label, userData=d)
        self._populate_layer_lists()

    def _populate_layer_lists(self):
        lid = self.ui.left_combo.currentIndex()
        rid = self.ui.right_combo.currentIndex()
        if lid == -1 or rid == -1:
            return
        ldata = self.ui.left_combo.itemData(lid).toPyObject()
        rdata = self.ui.right_combo.itemData(rid).toPyObject()
        self.ui.left_list.clear()
        self.ui.right_list.clear()

        for component in ldata.component_ids():
            item = QListWidgetItem(component.label)
            item.setData( Qt.UserRole, (ldata, component))
            self.ui.left_list.addItem(item)

        for component in rdata.component_ids():
            item = QListWidgetItem(component.label)
            item.setData( Qt.UserRole, (rdata, component))
            self.ui.right_list.addItem(item)

    def _connect(self):
        self.ui.left_combo.currentIndexChanged.connect(
            self._populate_layer_lists)
        self.ui.right_combo.currentIndexChanged.connect(
            self._populate_layer_lists)
        self.ui.glue_button.pressed.connect(self.link_selected)
        self.ui.un_glue_button.pressed.connect(self.remove_selected)
        self.ui.ok.pressed.connect(self.accept)
        self.ui.cancel.pressed.connect(self.reject)

    def link_selected(self):
        left_item = self.ui.left_list.currentItem()
        right_item = self.ui.right_list.currentItem()
        if left_item is None or right_item is None:
            return

        left_data = left_item.data(Qt.UserRole).toPyObject()
        right_data = right_item.data(Qt.UserRole).toPyObject()
        link = (left_data, right_data)
        if link in self._links:
            return
        self._links.add(link)
        item = QListWidgetItem("%s - %s <-> %s - %s" % (left_data[0].label,
                                                        left_data[1].label,
                                                        right_data[0].label,
                                                        right_data[1].label))
        item.setData(Qt.UserRole, link)
        self.ui.link_list.addItem(item)
        pass

    def remove_selected(self):
        link_item = self.ui.link_list.takeItem(self.ui.link_list.currentRow())
        if link_item is None:
            return
        data = link_item.data(Qt.UserRole).toPyObject()
        self._links.remove(data)

    @staticmethod
    def set_connections(data):
        widget = DataConnector(data)
        if widget.exec_() == QDialog.Accepted:
            widget._apply_links_to_data()

    def _apply_links_to_data(self):
        for link in self._links:
            data0, comp0 = link[0]
            data1, comp1 = link[1]
            glue_components_1to1(data0, comp0, data1, comp1)

def main():
    import sys
    from PyQt4.QtGui import QApplication, QMainWindow

    app = QApplication(sys.argv)
    data = glue.example_data.pipe()[:2]
    dc = glue.DataCollection(list(data))
    DataConnector.set_connections(dc)

if __name__ == "__main__":
    main()

