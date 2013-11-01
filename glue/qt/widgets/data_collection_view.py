from ...external.qt.QtGui import (QTreeWidgetItem, QPainter,
                                  QPixmap, QTreeWidgetItemIterator, QIcon,
                                  QItemSelectionModel, QTreeWidget
                                  )
from ...external.qt.QtCore import Qt

from .. import qtutil
from ... import core
from ..qtutil import icon_path


class DataCollectionView(qtutil.GlueTreeWidget, core.hub.HubListener):
    """ Passive view into a data collection.

    Uses hub messages to remain synced

    usage:

       dcv = DataCollectionView()
       dcv.setup(data_collection, hub)
       dcv.show()
    """
    # pylint: disable=R0904
    def __init__(self, parent=None):
        qtutil.GlueTreeWidget.__init__(self, parent)
        core.hub.HubListener.__init__(self)
        self.setStyleSheet('QTreeWidget{ background-color:rgb(250, 250, 250)}')
        self._data_collection = None
        self._hub = None
        self._layer_dict = {}
        self.checkable = False
        self.setColumnCount(2)

    def setup(self, data_collection, hub):
        """ Sync to data collection and hub
        :param data_collection: data collection to view
        :param hub: hub to use to remain synced

        Side effect:
           The data_collection will be registered to the hub
        """
        self._data_collection = data_collection
        self._hub = hub
        self.register_to_hub(hub)

        for data in data_collection:
            self._add_data(data, check_sync=False)
        self._assert_view_synced()

    @property
    def data_collection(self):
        return self._data_collection

    def register_to_hub(self, hub):
        """ Connect to hub, to automatically maintain
        synchronization with the underlying datacollection object

        hub: Hub instance to register to
        """

        data_filt = lambda x: x.sender.data in self.data_collection
        dc_filt = lambda x: x.sender is self.data_collection
        self.data_collection.register_to_hub(hub)

        hub.subscribe(self,
                      core.message.SubsetCreateMessage,
                      handler=lambda x: self._add_subset(x.sender),
                      filter=data_filt)
        hub.subscribe(self,
                      core.message.SubsetUpdateMessage,
                      handler=lambda x: self._sync_layer(x.sender),
                      filter=data_filt)
        hub.subscribe(self,
                      core.message.SubsetDeleteMessage,
                      handler=lambda x: self._remove_layer(x.sender),
                      filter=data_filt)
        hub.subscribe(self,
                      core.message.DataCollectionAddMessage,
                      handler=lambda x: self._add_data(x.data),
                      filter=dc_filt)
        hub.subscribe(self,
                      core.message.DataUpdateMessage,
                      handler=lambda x: self._sync_layer(x.sender),
                      filter=data_filt)
        hub.subscribe(self,
                      core.message.DataCollectionDeleteMessage,
                      handler=lambda x: self._remove_layer(x.data),
                      filter=dc_filt)

    def _assert_in_collection(self, layer):
        assert layer.data in self.data_collection

    def __getitem__(self, key):
        return self._layer_dict[key]

    def __setitem__(self, key, value):
        self.set_data(key, value)  # for drag and drop
        self._layer_dict[key] = value

    def __contains__(self, key):
        return key in self._layer_dict

    def _pop(self, key):
        self._layer_dict.pop(key)

    def __len__(self):
        return len(self._layer_dict)

    def _add_data(self, data, check_sync=True):
        """ Add a new data object to the view

        :param data: new data object.

        :param check_sync: If true, will assert the view is synced at
        end of method
        """

        self._assert_in_collection(data)

        if data in self:
            return

        label = data.label
        assert isinstance(self, QTreeWidget)

        branch = QTreeWidgetItem([label, '', '', ''])
        self.addTopLevelItem(branch)

        if self.checkable:
            branch.setCheckState(0, Qt.Checked)
        self.expandItem(branch)

        self[data] = branch
        self[branch] = data
        self.set_data(branch, data)

        for subset in data.subsets:
            self._add_subset(subset, check_sync=False)

        self.setCurrentItem(branch)
        self._sync_layer(data)
        if check_sync:
            self._assert_view_synced()

    def _add_subset(self, subset, check_sync=True):
        """ Add a new subset to the view.

        :param subset: new subset object.
        :param check_sync: If true, will assert the view is synced
        at end of method
        """
        self._assert_in_collection(subset)
        if subset in self:
            return

        if subset.data not in self:
            self._add_data(subset.data)  # will add subset too
            return

        label = subset.label
        parent = self[subset.data]
        branch = QTreeWidgetItem(parent, [label, '', '', ''])

        if self.checkable:
            branch.setCheckState(0, Qt.Checked)

        self[subset] = branch
        self[branch] = subset
        self.set_data(subset, subset)

        self._sync_layer(subset)
        if check_sync:
            self.setCurrentItem(branch)
            self._assert_view_synced()

        #possible that user generated with ctrl key held down,
        #which adds subset to selection. Override this
        self.setCurrentItem(branch, 0, QItemSelectionModel.Clear)
        self.setCurrentItem(branch, 0, QItemSelectionModel.Select)

    def _sync_layer(self, layer):
        """ Sync columns of display tree, to
        reflect the current style settings of the given layer

        :param layer: data or subset object
        """
        if layer not in self:
            return

        widget_item = self[layer]
        icon = qtutil.layer_icon(layer)
        widget_item.setIcon(1, icon)
        label = layer.label

        widget_item.setText(0, label)
        ncol = self.columnCount()
        for i in range(ncol):
            self.resizeColumnToContents(i)

    def _remove_layer(self, layer, check_sync=True):
        """ Remove a data or subset from the layer tree.

        :param layer: subset or data object to remove
        :type layer:
           :class:`~glue.core.subset.Subset` or
           :class:`~glue.core.data.Data`
        """
        if layer not in self:
            return

        widget_item = self[layer]
        parent = widget_item.parent()
        if parent:
            parent.removeChild(widget_item)
        else:
            index = self.indexOfTopLevelItem(widget_item)
            if index >= 0:
                self.takeTopLevelItem(index)

        self._pop(layer)
        self._pop(widget_item)

        if layer.data is layer:
            for subset in layer.subsets:
                self._remove_layer(subset, check_sync=False)

        if check_sync:
            self._assert_view_synced()

    def _assert_view_synced(self):
        layers_in_widget = set()
        layers_in_collection = set()
        for d in self.data_collection:
            layers_in_collection.add(d)
            for s in d.subsets:
                layers_in_collection.add(s)

        for item in self.items():
            assert item in self, 'orphan TreeWidgetItem: %s' % item
            layer = self[item]
            layers_in_widget.add(layer)

        assert layers_in_widget == layers_in_collection

    def items(self):
        """ Yields an iterator over QTreeWidgetItems"""
        iterator = QTreeWidgetItemIterator(self)
        while iterator.value() is not None:
            yield iterator.value()
            iterator += 1

    def unregister(self, hub):
        super(DataCollectionView, self).unregister(hub)
        self.data_collection.unregister(hub)

    def paintEvent(self, event):
        super(DataCollectionView, self).paintEvent(event)
        if self.topLevelItemCount() != 0:
            return
        painter = QPainter(self.viewport())
        font = painter.font()
        font.setPointSize(14)
        painter.setFont(font)
        painter.setRenderHint(painter.Antialiasing)
        rect = event.rect()
        rect.setBottom(rect.bottom() - 30)
        painter.drawText(rect, Qt.AlignLeft | Qt.AlignBottom,
                         "  Load Data")
        x = 10
        y = event.rect().bottom() - 30
        painter.drawPixmap(x, y, QPixmap(icon_path('glue_down_arrow')))


def _color_icon(color, size=20, alpha=1.0):
    pixm = QPixmap(size, size)
    color = qtutil.mpl_to_qt4_color(color, alpha=alpha)
    pixm.fill(color)
    icon = QIcon(pixm)
    return icon
