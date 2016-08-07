from __future__ import absolute_import, division, print_function

from glue.core import Data, DataCollection
from qtpy import QtGui, QtWidgets
from qtpy.QtCore import Qt
from glue.core.hub import HubListener
from glue.core.message import (ComponentsChangedMessage,
                               DataCollectionAddMessage,
                               DataCollectionDeleteMessage,
                               DataUpdateMessage)
from glue.utils import nonpartial
from glue.utils.qt import update_combobox
from glue.utils.qt.widget_properties import CurrentComboDataProperty

__all__ = ['ComponentIDComboHelper', 'ManualDataComboHelper',
           'DataCollectionComboHelper']

class ComponentIDComboHelper(HubListener):
    """
    The purpose of this class is to set up a combo showing componentIDs for
    one or more datasets, and to update these componentIDs if needed, for
    example if new components are added to a dataset, or if componentIDs are
    renamed.

    Parameters
    ----------
    component_id_combo : Qt combo widget
        The Qt widget for the component ID combo box
    data_collection : :class:`~glue.core.DataCollection`
        The data collection to which the datasets belong - this is needed
        because if a dataset is removed from the data collection, we want to
        remove it here.
    visible : bool, optional
        Only show visible components
    numeric : bool, optional
        Show numeric components
    categorical : bool, optional
        Show categorical components
    """

    def __init__(self, component_id_combo, data_collection, visible=True,
                 numeric=True, categorical=True):

        super(ComponentIDComboHelper, self).__init__()

        if data_collection.hub is None:
            raise ValueError("Hub on data collection is not set")

        self._visible = visible
        self._numeric = numeric
        self._categorical = categorical
        self._component_id_combo = component_id_combo
        self._data = []
        self._data_collection = data_collection
        self.hub = data_collection.hub

    def clear(self):
        self._data.clear()
        self.refresh()

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        self._visible = value
        self.refresh()

    @property
    def numeric(self):
        return self._numeric

    @numeric.setter
    def numeric(self, value):
        self._numeric = value
        self.refresh()

    @property
    def categorical(self):
        return self._categorical

    @categorical.setter
    def categorical(self, value):
        self._categorical = value
        self.refresh()

    def append_data(self, data):

        if self.hub is None:
            if data.hub is None:
                raise ValueError("Hub is not set on Data object")
            else:
                self.hub = data.hub
        elif data.hub is not self.hub:
            raise ValueError("Data Hub is different from current hub")

        self._data.append(data)

        self.refresh()

    def remove_data(self, data):
        self._data.remove(data)
        self.refresh()

    def set_multiple_data(self, datasets):
        """
        Add multiple datasets to the combo in one go (and clear any previous datasets).

        Parameters
        ----------
        datasets : list
            The list of :class:`~glue.core.data.Data` objects to add
        """
        try:
            self._data.clear()
        except AttributeError:  # PY2
            self._data[:] = []
        self._data.extend(datasets)
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
                if data.label is None or data.label == '':
                    label_data.append(("Untitled Data", None))
                else:
                    label_data.append((data.label, None))

            if self.visible:
                all_component_ids = data.visible_components
            else:
                all_component_ids = data.components

            component_ids = []
            for cid in all_component_ids:
                comp = data.get_component(cid)
                if (comp.numeric and self.numeric) or (comp.categorical and self.categorical):
                    component_ids.append(cid)

            label_data.extend([(cid.label, (cid, data)) for cid in component_ids])

        update_combobox(self._component_id_combo, label_data)

        # Disable header rows
        model = self._component_id_combo.model()
        for index in range(self._component_id_combo.count()):
            if self._component_id_combo.itemData(index) is None:
                item = model.item(index)
                palette = self._component_id_combo.palette()
                item.setFlags(item.flags() & ~(Qt.ItemIsSelectable | Qt.ItemIsEnabled))
                item.setData(palette.color(QtGui.QPalette.Disabled, QtGui.QPalette.Text))

        index = self._component_id_combo.currentIndex()
        if self._component_id_combo.itemData(index) is None:
            for index in range(index + 1, self._component_id_combo.count()):
                if self._component_id_combo.itemData(index) is not None:
                    self._component_id_combo.setCurrentIndex(index)
                    break


    def register_to_hub(self, hub):
        hub.subscribe(self, ComponentsChangedMessage,
                      handler=nonpartial(self.refresh),
                      filter=lambda msg: msg.data in self._data)
        hub.subscribe(self, DataCollectionDeleteMessage,
                      handler=lambda msg: self.remove_data(msg.data),
                      filter=lambda msg: msg.sender is self._data_collection)

    def unregister(self, hub):
        hub.unsubscribe_all(self)


class BaseDataComboHelper(HubListener):
    """
    This is a base class for helpers for combo boxes that need to show a list
    of data objects.

    Parameters
    ----------
    data_combo : Qt combo widget
        The Qt widget for the data combo box
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
                helper.append_data(self._data)
            helper.refresh()

    def add_component_id_combo(self, combo):
        helper = ComponentIDComboHelper(combo)
        self._component_id_helpers.append_data(helper)
        if self._data is not None:
            helper.append_data(self._data)

    @property
    def hub(self):
        return self._hub

    @hub.setter
    def hub(self, value):
        self._hub = value
        if value is not None:
            self.register_to_hub(value)

    def register_to_hub(self, hub):
        pass


class ManualDataComboHelper(BaseDataComboHelper):
    """
    This is a helper for combo boxes that need to show a list of data objects
    that is manually curated.

    Datasets are added and removed using the
    :meth:`~ManualDataComboHelper.append_data` and
    :meth:`~ManualDataComboHelper.remove_data` methods.

    Parameters
    ----------
    data_combo : Qt combo widget
        The Qt widget for the data combo box
    data_collection : :class:`~glue.core.DataCollection`
        The data collection to which the datasets belong - this is needed
        because if a dataset is removed from the data collection, we want to
        remove it here.
    """

    def __init__(self, data_combo, data_collection):
        super(ManualDataComboHelper, self).__init__(data_combo)

        if data_collection.hub is None:
            raise ValueError("Hub on data collection is not set")

        self._data_collection = data_collection
        self._datasets = []
        self.hub = data_collection.hub

    def append_data(self, data):
        self._datasets.append(data)
        self.refresh()

    def remove_data(self, data):
        self._datasets.remove(data)
        self.refresh()

    def register_to_hub(self, hub):

        super(ManualDataComboHelper, self).register_to_hub(hub)

        hub.subscribe(self, DataUpdateMessage,
                      handler=nonpartial(self.refresh),
                      filter=lambda msg: msg.sender in self._datasets)
        hub.subscribe(self, DataCollectionDeleteMessage,
                      handler=lambda msg: self.remove_data(msg.data),
                      filter=lambda msg: msg.sender is self._data_collection)


class DataCollectionComboHelper(BaseDataComboHelper):
    """
    This is a helper for combo boxes that need to show a list of data objects
    that is always in sync with a :class:`~glue.core.DataCollection`.

    Parameters
    ----------
    data_combo : Qt combo widget
        The Qt widget for the data combo box
    data_collection : :class:`~glue.core.DataCollection`
        The data collection with which to stay in sync
    """

    def __init__(self, data_combo, data_collection):
        super(DataCollectionComboHelper, self).__init__(data_combo)

        if data_collection.hub is None:
            raise ValueError("Hub on data collection is not set")

        self._datasets = data_collection
        self.register_to_hub(data_collection.hub)
        self.refresh()

    def register_to_hub(self, hub):
        super(DataCollectionComboHelper, self).register_to_hub(hub)
        hub.subscribe(self, DataUpdateMessage,
                      handler=nonpartial(self.refresh),
                      filter=lambda msg: msg.sender in self._datasets)
        hub.subscribe(self,DataCollectionAddMessage,
                      handler=nonpartial(self.refresh),
                      filter=lambda msg: msg.sender is self._datasets)
        hub.subscribe(self, DataCollectionDeleteMessage,
                      handler=nonpartial(self.refresh),
                      filter=lambda msg: msg.sender is self._datasets)


if __name__ == "__main__":

    from glue.utils.qt import get_qapp

    app = get_qapp()

    window = QtWidgets.QWidget()

    layout = QtWidgets.QVBoxLayout()

    window.setLayout(layout)

    data_combo = QtWidgets.QComboBox()
    layout.addWidget(data_combo)

    cid1_combo = QtWidgets.QComboBox()
    layout.addWidget(cid1_combo)

    cid2_combo = QtWidgets.QComboBox()
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
