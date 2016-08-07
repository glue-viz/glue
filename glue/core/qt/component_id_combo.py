from __future__ import absolute_import, division, print_function

from qtpy import QtWidgets
from glue.core.hub import HubListener
from glue.core.message import ComponentsChangedMessage


class ComponentIDCombo(QtWidgets.QComboBox, HubListener):

    """ A widget to select among componentIDs in a dataset """

    def __init__(self, data=None, parent=None, visible_only=True):
        QtWidgets.QComboBox.__init__(self, parent)
        self._data = data
        self._visible_only = visible_only

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if value is None:
            return
        self._data = value
        if value.hub is not None:
            self.register_to_hub(value.hub)
        self.refresh_components()

    @property
    def component(self):
        return self.itemData(self.currentIndex())

    @component.setter
    def component(self, value):
        for i in range(self.count()):
            if self.itemData(i) is value:
                self.setCurrentIndex(i)
                return
        else:
            raise ValueError("Unable to select %s" % value)

    def refresh_components(self):
        if self.data is None:
            return

        self.blockSignals(True)
        old_data = self.itemData(self.currentIndex())

        self.clear()
        if self._visible_only:
            fields = self.data.visible_components
        else:
            fields = self.data.components

        index = 0
        for i, f in enumerate(fields):
            self.addItem(f.label, userData=f)
            if f == old_data:
                index = i

        self.blockSignals(False)
        self.setCurrentIndex(index)

    def register_to_hub(self, hub):
        hub.subscribe(self,
                      ComponentsChangedMessage,
                      handler=lambda x: self.refresh_components,
                      filter=lambda x: x.data is self._data)