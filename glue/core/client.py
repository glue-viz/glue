from __future__ import absolute_import, division, print_function

from glue.core.message import (DataUpdateMessage,
                               SubsetUpdateMessage,
                               SubsetCreateMessage,
                               SubsetDeleteMessage,
                               DataCollectionDeleteMessage,
                               NumericalDataChangedMessage)
from glue.core.data_collection import DataCollection
from glue.core.subset import Subset
from glue.core.data import Data
from glue.core.hub import HubListener


__all__ = ['Client', 'BasicClient']


class Client(HubListener):

    """
    Base class for interaction / visualization modules

    Attributes
    ----------
    data: DataCollection instance
        The data associated with this client.

    """

    def __init__(self, data):
        """
        Create a new client object.

        Parameters
        ----------
        data: Data, DataCollection, or list of data
            The primary data associated with this client.

        Raises
        ------
        TypeError: If the data input is the wrong type
        """
        super(Client, self).__init__()
        self._data = data
        if not isinstance(data, DataCollection):
            raise TypeError("Input data must be a DataCollection: %s"
                            % type(data))

    @property
    def data(self):
        """ Returns the data collection """
        return self._data

    def register_to_hub(self, hub):
        """The main method to establish a link with a hub,
        and set up event handlers. For common message types

        Client subclasses at a minimum should override these methods
        to provide functionality:
        _add_subset
        _update_subset
        _remove_subset
        _remove_data

        Clients can also override register_to_hub to add additional
        event handlers.

        Attributes
        ----------
        hub: The hub to subscribe to

        """
        has_data = lambda x: x.sender.data in self._data
        has_data_collection = lambda x: x.sender is self._data

        hub.subscribe(self,
                      SubsetCreateMessage,
                      handler=self._add_subset,
                      filter=has_data)
        hub.subscribe(self,
                      SubsetUpdateMessage,
                      handler=self._update_subset,
                      filter=has_data)
        hub.subscribe(self,
                      SubsetDeleteMessage,
                      handler=self._remove_subset,
                      filter=has_data)
        hub.subscribe(self,
                      DataUpdateMessage,
                      handler=self._update_data,
                      filter=has_data)
        hub.subscribe(self,
                      NumericalDataChangedMessage,
                      handler=self._numerical_data_changed,
                      filter=has_data)
        hub.subscribe(self,
                      DataCollectionDeleteMessage,
                      handler=self._remove_data,
                      filter=has_data_collection)

    def _add_subset(self, message):
        raise NotImplementedError

    def _remove_data(self, message):
        raise NotImplementedError

    def _remove_subset(self, message):
        raise NotImplementedError

    def _update_data(self, message):
        """ Default handler for DataMessage """
        raise NotImplementedError

    def _update_subset(self, message):
        """ Default handler for SubsetUpdateMessage """
        raise NotImplementedError

    def apply_roi(self, roi):
        raise NotImplementedError

    def _numerical_data_changed(self, message):
        raise NotImplementedError


class BasicClient(Client):

    def _add_subset(self, message):
        subset = message.subset
        self.add_layer(subset)

    def _update_subset(self, message):
        subset = message.subset
        self.update_layer(subset)

    def _remove_subset(self, message):
        subset = message.subset
        self.remove_layer(subset)

    def _remove_data(self, message):
        self.remove_layer(message.data)

    def _update_data(self, message):
        self.update_layer(message.data)

    def add_layer(self, layer):
        if self.layer_present(layer):
            return
        if layer.data not in self.data:
            raise TypeError("Data not in collection")

        if isinstance(layer, Data):
            self._do_add_data(layer)
            for subset in layer.subsets:
                self.add_layer(subset)
        else:
            if not self.layer_present(layer.data):
                self.add_layer(layer.data)
            else:
                self._do_add_subset(layer)

        self.update_layer(layer)

    def update_layer(self, layer):
        if not self.layer_present(layer):
            return
        if isinstance(layer, Subset):
            self._do_update_subset(layer)
        else:
            self._do_update_data(layer)

    def remove_layer(self, layer):
        if not self.layer_present(layer):
            return
        if isinstance(layer, Data):
            self._do_remove_data(layer)
            for subset in layer.subsets:
                self._do_remove_subset(subset)
        else:
            self._do_remove_subset(layer)

    def _do_add_data(self, data):
        raise NotImplementedError

    def _do_add_subset(self, subset):
        raise NotImplementedError

    def _do_update_subset(self, subset):
        raise NotImplementedError

    def _do_update_data(self, data):
        raise NotImplementedError

    def _do_remove_subset(self, subset):
        raise NotImplementedError

    def _do_remove_data(self, data):
        raise NotImplementedError

    def layer_present(self, layer):
        raise NotImplementedError
