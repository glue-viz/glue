from PyQt4.QtGui import QDialog, QListWidgetItem

from ui_link_editor import Ui_LinkEditor
import glue

class LinkEditor(QDialog):

    def __init__(self, collection, functions = None, parent=None):
        super(LinkEditor, self).__init__(parent)
        self._collection = collection
        self._functions = functions or glue.env['link_functions']

        self._ui = Ui_LinkEditor()
        self._init_widgets()
        self._connect()

    def _init_widgets(self):
        self._ui.setupUi(self)
        self._ui.signature_editor.setup(self._functions)
        self._ui.component_selector.setup(self._collection)
        for link in self._collection.links:
            self._add_link(link)

    def _connect(self):
        self._ui.signature_editor.add_button.clicked.connect(
            self._add_new_link)
        self._ui.remove_link.clicked.connect(self._remove_link)

    def _add_link(self, link):
        current = self._ui.current_links
        item = QListWidgetItem(str(link))
        current.addItem(item)
        current.data[item] = link

    def _add_new_link(self):
        links = self._ui.signature_editor.links()
        for link in links:
            self._add_link(link)

    def links(self):
        current = self._ui.current_links
        return current.data.values()

    def _remove_link(self):
        current = self._ui.current_links
        item = current.currentItem()
        row = current.currentRow()
        if item is None:
            return
        current.data.pop(item)
        deleted = current.takeItem(row)
        assert deleted == item # sanity check

    @classmethod
    def update_links(cls, collection):
        widget = cls(collection)
        isok = widget.exec_()
        if isok:
            links = widget.links()
            collection.links = links


def main(): # pragma: no cover
    from PyQt4.QtGui import QApplication

    import numpy as np

    app = QApplication([''])

    d = glue.Data(label = 'd1')
    d2 = glue.Data(label = 'd2')
    c1 = glue.Component(np.array([1, 2, 3]))
    c2 = glue.Component(np.array([1, 2, 3]))
    c3 = glue.Component(np.array([1, 2, 3]))
    d.add_component(c1, 'a')
    d.add_component(c2, 'b')
    d2.add_component(c3, 'c')
    dc = glue.DataCollection()
    dc.append(d)
    dc.append(d2)

    def f(a, b, c):
        pass

    def g(h, i):
        pass

    def h(j, k=None):
        pass

    w = LinkEditor(dc, [f, g, h])
    w.show()
    app.exec_()

if __name__ == "__main__":
    main()