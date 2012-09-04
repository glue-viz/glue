from .hub import Hub, HubListener
from .data import Data
from .link_manager import LinkManager
from .live_link import LiveLinkManager
from .registry import Registry
from .message import (DataCollectionAddMessage,
                      DataCollectionDeleteMessage,
                      DataAddComponentMessage)

__all__ = ['DataCollection']


class DataCollection(HubListener):
    """DataCollections manage sets of data. They have the following
    responsibilities:

       * Providing a way to retrieve and store data
       * Broadcasting messages when data are added or removed
       * Keeping each managed data set's list of DerivedComponents up-to-date
    """

    def __init__(self, data=None):
        """
        :param data: glue.Data object, or list of such objects (optional)
                      These objects will be auto-appended to the collection
        """
        super(DataCollection, self).__init__()
        self.hub = None
        self._link_manager = LinkManager()
        self.live_link_manager = LiveLinkManager()

        self._data = []
        if isinstance(data, Data):
            self.append(data)
        elif isinstance(data, list):
            for d in data:
                self.append(d)

    @property
    def data(self):
        """ The data objects in the collection (Read Only) """
        return self._data

    def append(self, data):
        """ Add a new dataset to this collection.

        Appending emits a DataCollectionAddMessage.
        It also updates the list of DerivedComponents that each
        data set can work with.

        :param data: :class:`~glue.core.data.Data` object to add
        """
        if data in self:
            return
        self._data.append(data)
        if self.hub:
            data.hub = self.hub
            for s in data.subsets:
                s.register()
            msg = DataCollectionAddMessage(self, data)
            self.hub.broadcast(msg)
        self._sync_link_manager()

    def remove(self, data):
        """ Remove a data set from the collection

        Emits a DataCollectionDeleteMessage

        :param data: the object to remove
        :type data: :class:`~glue.core.data.Data`
        """
        if data not in self._data:
            return
        self._data.remove(data)
        Registry().unregister(data, Data)
        if self.hub:
            msg = DataCollectionDeleteMessage(self, data)
            self.hub.broadcast(msg)

    def _sync_link_manager(self):
        """ update the LinkManager, so all the DerivedComponents
        for each data set are up-to-date
        """

        # add any links in the data
        for d in self._data:
            for derived in d.derived_components:
                self._link_manager.add_link(d.get_component(derived).link)
            for link in d.coordinate_links:
                self._link_manager.add_link(link)

        for d in self._data:
            self._link_manager.update_data_components(d)

    @property
    def links(self):
        return self._link_manager.links

    @links.setter
    def links(self, links):
        self._link_manager.clear()
        for link in links:
            self._link_manager.add_link(link)

        for d in self._data:
            self._link_manager.update_data_components(d)

    def register_to_hub(self, hub):
        """ Register managed data objects to a hub
        :param hub: The hub to register with
        :type hub: :class:`~glue.core.hub.Hub`
        """
        if not isinstance(hub, Hub):
            raise TypeError("Input is not a Hub object: %s" % type(hub))
        self.hub = hub
        self.live_link_manager.hub = hub

        #re-assign all data, subset hub instances to this hub
        for d in self._data:
            d.register_to_hub(hub)
            for s in d.subsets:
                s.register()

        hub.subscribe(self, DataAddComponentMessage,
                      lambda msg: self._sync_link_manager(),
                      filter=lambda x: x.sender in self._data)

    def __contains__(self, obj):
        return obj in self._data

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)
