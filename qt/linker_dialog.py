from PyQt4.QtCore import SIGNAL
from PyQt4.QtCore import Qt
from PyQt4.QtGui import *
from ui_subsetlinkerdialog import Ui_SubsetLinkerDialog

import cloudviz as cv

class LinkerDialog(QDialog):
    """ A modal dialog widget to interactively link subsets together."""

    def __init__(self, data, parent=None):
        QDialog.__init__(self, parent)
        self.ui = Ui_SubsetLinkerDialog()
        self.ui.setupUi(self)
        self.setModal(True)
        self.data = data
        self.tree_map = {}
        self.add_connections()
        self.populate_layer_tree()

    def add_connections(self):
        self.ui.okButton.clicked.connect(self.accept)
        self.ui.cancelButton.clicked.connect(self.reject)
        self.ui.layerTree.itemChanged.connect(self.toggle_selected)

    def populate_layer_tree(self):
        for d in self.data:
            label = d.label
            branch = QTreeWidgetItem(self.ui.layerTree, [label])
            self.tree_map[branch] = d
            for s in d.subsets:
                label = s.label
                leaf = QTreeWidgetItem(branch, [label])
                self.tree_map[leaf] = s
                leaf.setCheckState(0, Qt.Unchecked)
            self.ui.layerTree.expandItem(branch)

    def checked(self):
        subsets = []
        iter = QTreeWidgetItemIterator(self.ui.layerTree, QTreeWidgetItemIterator.Checked)
        while iter.value():
            subsets.append(self.tree_map[iter.value()])
            iter += 1
        return subsets

    def toggle_selected(self):
        self.ui.okButton.setDisabled(len(self.checked()) < 2)

    def register_link(self, result):
        subsets = result.subsets
        hubs = []
        for s in subsets:
            if hasattr(s, 'data') and hasattr(s.data, 'hub'):
                hubs.append(s.data.hub)
        for h in hubs:
            result.register_to_hub(h)

    def getLink(self, link_class = None):

        if link_class is None:
            link_class = cv.subset_link.SubsetLink

        result = self.exec_()
        if result == QDialog.Accepted:
            subsets = self.checked()
            result = link_class(subsets)
            self.register_link(result)
            return result
        else:
            return None


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    d, d2, s, s2 = cv.example_data.pipe()

    h = cv.Hub(d, d2, s, s2)
    gui = LinkerDialog([d, d2])
    print gui.getLink()
