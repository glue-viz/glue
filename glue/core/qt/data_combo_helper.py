from __future__ import absolute_import, division, print_function

from glue.core import Data, DataCollection
from glue.external.qt import QtGui
from glue.external.qt.QtCore import Qt
from glue.core.hub import HubListener
from glue.core.message import ComponentsChangedMessage, ComponentIDRenamedMessage
from glue.core.message import (DataCollectionAddMessage,
                              DataCollectionDeleteMessage,
                              DataUpdateMessage)
from glue.utils import nonpartial
from glue.utils.qt import update_combobox
from glue.utils.qt.widget_properties import CurrentComboDataProperty


class ComponentIDComboHelper(HubListener):
    """
    The purpose of this class is to set up a combo showing componentIDs for one
    or more datasets, and to update these componentIDs if needed, for example
    if new componnts are added to a dataset, or if componentIDs are renamed.
    """

    def __init__(self, component_id_combo):
        self._component_id_combo = component_id_combo
        self._data = []
        self.hub = None

    def clear(self):
        self._data.clear()
        self.refresh()

    def append(self, data):
        
        if self.hub is None:
            if data.hub is None:
                raise ValueError("Hub is not set on Data object")
            else:
                self.hub = data.hub
        elif data.hub is not self.hub:
            raise ValueError("Data Hub is different from current hub")

        self._data.append(data)

        self.refresh()

    def remove(self, data):
        self._data.remove(data)
        self.refresh()

    @property
    def hub(self):
        return self._hub

    @hub.setter
    def hub(self, value):
        self._hub = value
        if value is not None:
            self.register_to_hub(value)

    def refresh(self):

        label_data = []

        for data in self._data:

            if len(self._data) > 1:
                if data.label is None:
                    label_data.append((data.label, None))
                else:
                    label_data.append(("Untitled Data", None))

            component_ids = data.visible_components
            label_data.extend([(cid.label, cid) for cid in component_ids])

        update_combobox(self._component_id_combo, label_data)

        # Disable header rows
        model = self._component_id_combo.model()
        for index in range(self._component_id_combo.count()):
            if self._component_id_combo.itemData(index) is None:
                item = model.item(index)
                palette = self._component_id_combo.palette()
                item.setFlags(item.flags() & ~(Qt.ItemIsSelectable | Qt.ItemIsEnabled))
                item.setData(palette.color(QtGui.QPalette.Disabled, QtGui.QPalette.Text))

    def register_to_hub(self, hub):
        hub.subscribe(self,
                      ComponentsChangedMessage,
                      handler=nonpartial(self.refresh),
                      filter=lambda x: x.data in self._data)
        hub.subscribe(self,
                      ComponentIDRenamedMessage,
                      handler=nonpartial(self.refresh),
                      filter=lambda x: x.component_id in self.component_ids)

    @property
    def component_ids(self):
        component_ids = set()
        for data in self._data:
            component_ids.update(data.visible_components)
        return component_ids


class BaseDataComboHelper(HubListener):
    """

    """

    _data = CurrentComboDataProperty('_data_combo')

    def __init__(self, data_combo):
        super(BaseDataComboHelper, self).__init__()
        self._data_combo = data_combo
        self._component_id_helpers = []
        self._data_combo.currentIndexChanged.connect(self.refresh_component_ids)

    def refresh(self):
        label_data = [(data.label, data) for data in self._datasets]
        update_combobox(self._data_combo, label_data)
        self.refresh_component_ids()

    def refresh_component_ids(self):
        for helper in self._component_id_helpers:
            helper.clear()
            if self._data is not None:
                helper.append(self._data)
            helper.refresh()

    def register_to_hub(self, hub):
        dc_filt = lambda msg: msg.sender is self.data_collection
        hub.subscribe(self, DataUpdateMessage,
                      handler=nonpartial(self.refresh),
                      filter=lambda x: x.data in self._datasets)

    def add_component_id_combo(self, combo):
        helper = ComponentIDComboHelper(combo)
        self._component_id_helpers.append(helper)
        if self._data is not None:
            helper.append(self._data)


class ManualDataComboHelper(BaseDataComboHelper):
    """

    """

    def __init__(self, data_combo):
        super(ManualDataComboHelper, self).__init__(data_combo)
        self._datasets = []

    def append(self, data):
        self._datasets.append(data)
        self.refresh()

    def remove(self, data):
        self._datasets.remove(data)
        self.refresh()


class DataCollectionComboHelper(BaseDataComboHelper):
    """
    The purpose of this class is to set up a combo box showing data objects in
    sync with a DataCollection, and optionally a combo box showing ComponentIDs
    for one or more datasets, and keep those in sync (in case new ComponentIDs
    are added or existing ones are renamed).

    Parameters
    ----------
    data_combo : ``QComboBox`` instance
        The data combo to set up
    component_id_combo : ``QComboBox`` instance
        The component ID combo to set up
    """

    def __init__(self, data_combo, data_collection, component_id_combo=None):
        super(DataCollectionComboHelper, self).__init__(data_combo)

        if data_collection.hub is None:
            raise ValueError("Hub on data collection is not set")

        self._datasets = data_collection
        self.register_to_hub(data_collection.hub)
        self.refresh()

    def register_to_hub(self, hub):
        super(DataCollectionComboHelper, self).register_to_hub(hub)
        hub.subscribe(self,DataCollectionAddMessage,
                      handler=nonpartial(self.refresh),
                      filter=lambda msg: msg.sender is self._datasets)
        hub.subscribe(self, DataCollectionDeleteMessage,
                      handler=nonpartial(self.refresh),
                      filter=lambda msg: msg.sender is self._datasets)


if __name__ == "__main__":

    from glue.external.qt import get_qapp

    app = get_qapp()

    window = QtGui.QWidget()

    layout = QtGui.QVBoxLayout()

    window.setLayout(layout)

    data_combo = QtGui.QComboBox()
    layout.addWidget(data_combo)

    cid1_combo = QtGui.QComboBox()
    layout.addWidget(cid1_combo)

    cid2_combo = QtGui.QComboBox()
    layout.addWidget(cid2_combo)

    d1 = Data(x=[1,2,3], y=[2,3,4], label='banana')
    d2 = Data(a=[0,1,1], b=[2,1,1], label='apple')
    dc = DataCollection([d1, d2])

    helper = DataCollectionComboHelper(data_combo, dc)

    helper.add_component_id_combo(cid1_combo)
    helper.add_component_id_combo(cid2_combo)

    window.show()
    window.raise_()
    # app.exec_()
