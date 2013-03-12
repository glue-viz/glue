from ..external.qt.QtGui import QDialog, QListWidgetItem

from .. import core

from .ui.link_editor import Ui_LinkEditor


class LinkEditor(QDialog):

    def __init__(self, collection, functions=None, parent=None):
        super(LinkEditor, self).__init__(parent)
        self._collection = collection

        self._ui = Ui_LinkEditor()
        self._init_widgets()
        self._connect()
        if len(collection) > 1:
            self._ui.right_components.set_data_row(1)
        self._size = None

    def _init_widgets(self):
        self._ui.setupUi(self)
        self._ui.left_components.setup(self._collection)
        self._ui.right_components.setup(self._collection)
        self._ui.signature_editor.hide()
        for link in self._collection.links:
            self._add_link(link)

    def _connect(self):
        self._ui.add_link.clicked.connect(self._add_new_link)
        self._ui.remove_link.clicked.connect(self._remove_link)
        self._ui.toggle_editor.clicked.connect(self._toggle_advanced)
        self._ui.signature_editor._ui.addButton.clicked.connect(
            self._add_new_link)

    @property
    def advanced(self):
        return self._ui.signature_editor.isVisible()

    @advanced.setter
    def advanced(self, state):
        """Set whether the widget is in advanced state"""
        size = self.size()
        self._ui.signature_editor.setVisible(state)
        self._ui.toggle_editor.setText("Basic" if state else "Advanced")
        self.resize(size)

    def _toggle_advanced(self):
        """Show or hide the signature editor widget"""
        self.advanced = not self.advanced

    def _selected_components(self):
        result = []
        id1 = self._ui.left_components.component
        id2 = self._ui.right_components.component
        if id1:
            result.append(id1)
        if id2:
            result.append(id2)
        return result

    def _simple_links(self):
        """Return identity links which connect the highlighted items
        in each component selector.

        Returns:
          A list of :class:`~glue.core.ComponentLink` objects
          If items are not selected in the component selectors,
          an empty list is returned
        """
        comps = self._selected_components()
        if len(comps) != 2:
            return []
        assert isinstance(comps[0], core.data.ComponentID), comps[0]
        assert isinstance(comps[1], core.data.ComponentID), comps[1]
        link1 = core.component_link.ComponentLink([comps[0]], comps[1])
        return [link1]

    def _add_link(self, link):
        current = self._ui.current_links
        item = QListWidgetItem(str(link))
        current.addItem(item)
        item.setHidden(link.hide_from_editor)
        current.data[item] = link

    def _add_new_link(self):
        if not self.advanced:
            links = self._simple_links()
        else:
            links = self._ui.signature_editor.links()
            self._ui.signature_editor.clear_inputs()

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
        assert deleted == item  # sanity check

    @classmethod
    def update_links(cls, collection):
        widget = cls(collection)
        isok = widget.exec_()
        if isok:
            links = widget.links()
            collection.set_links(links)
