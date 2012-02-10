from PyQt4.QtCore import SIGNAL
from PyQt4.QtCore import Qt
from PyQt4.QtGui import *

import cloudviz as cv

class QtLinkerDialog(QMainWindow):
    def __init__(self, data, link_class=None, parent=None):
        QMainWindow.__init__(self, parent)

        if link_class is None:
            link_class = cv.subset_link.SubsetLink
        self.link_class = link_class
        self.link_object = None

        self.setWindowTitle("Subset Linker")
        self.data = data
        self.create_main_frame()

    def create_layer_tree(self):
        self.tree = {}
        tree = QTreeWidget()
        self.tree = {'root':tree}
        tree.setHeaderLabels(["Layer"])
        ok = QPushButton("OK")
        ok.setDisabled(True)
        self.ok = ok
        self.connect(ok, SIGNAL('clicked(bool)'), self.create_link)
        self.connect(tree, SIGNAL('itemChanged(QTreeWidgetItem *, int)'),
                     self.toggle_selected)
        cancel = QPushButton("Cancel")
        row = QHBoxLayout()
        row.addWidget(ok)
        row.addWidget(cancel)
        row.setContentsMargins(0,0,0,0)
        return tree, row

    def populate_layer_tree(self):
        tree = self.tree['root']
        for d in self.data:
            print d
            label = d.label
            branch = QTreeWidgetItem(self.tree['root'], [label])
            self.tree[branch] = d
            for s in d.subsets:
                print s
                label = s.label
                leaf = QTreeWidgetItem(branch, [label])
                self.tree[leaf] = s
                leaf.setCheckState(0, Qt.Unchecked)
            tree.expandItem(branch)

    def checked(self):
        subsets = []
        iter = QTreeWidgetItemIterator(self.tree['root'], QTreeWidgetItemIterator.Checked)
        while iter.value():
            subsets.append(self.tree[iter.value()])
            print subsets[-1]
            iter += 1
        return subsets

    def toggle_selected(self):
        print 'toggle'
        self.ok.setDisabled(len(self.checked()) < 2)

    def create_link(self):
        subsets = self.checked()
        result = self.link_class(subsets)
        self.link_object = result

    def create_main_frame(self):
        self.main_frame = QWidget()
        self.layout = QVBoxLayout()

        self.main_frame.setLayout(self.layout)
        self.setCentralWidget(self.main_frame)

        tree, row = self.create_layer_tree()
        self.populate_layer_tree()
        self.layout.addLayout(row)
        self.layout.addWidget(tree)


if __name__ == "__main__":
    import sys

    d, d2, s, s2 = cv.example_data.pipe()
    app = QApplication(sys.argv)

    h = cv.Hub(d, d2, s, s2)
    gui = QtLinkerDialog([d, d2])
    gui.show()
    sys.exit(app.exec_())
