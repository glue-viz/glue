import glue


class DataCollection(glue.HubListener):
    """DataCollections manage sets of data. They have the following
    responsibilities:

       * Providing a way to retrieve and store data
       * Broadcasting messages when data are added or removed
       * Keeping each managed data set's list of DerivedComponents up-to-date
    """

    def __init__(self, data=None):
        """ Create a new DataCollection

        Parameters
        ----------
        data : glue.Data object, or list of such objects (optional)
               These objects will be auto-appended to the collection
        """
        self.hub = None
        self._link_manager = glue.LinkManager()

        self._data = []
        if isinstance(data, glue.data.Data):
            self.append(data)
        elif isinstance(data, list):
            for d in data:
                self.append(d)

    def append(self, data):
        """ Add a new dataset to this collection.

        Appending emits a DataCollectionAddMessage.
        It also updates the list of DerivedComponents that each
        data set can work with.

        Parameters
        ----------
        data : `glue.Data` object to add
        """
        if data in self:
            return
        self._data.append(data)
        if self.hub:
            data.hub = self.hub
            for s in data.subsets:
                s.register()
            msg = glue.message.DataCollectionAddMessage(self, data)
            self.hub.broadcast(msg)
        self._sync_link_manager()

    def remove(self, data):
        """ Remove a data set from the collection

        Emits a DataCollectionDeleteMessage

        Parameters
        ----------
        data : the glue.Data object to remove

        """
        if data not in self._data:
            return
        self._data.remove(data)
        if self.hub:
            msg = glue.message.DataCollectionDeleteMessage(self, data)
            self.hub.broadcast(msg)

    def _sync_link_manager(self):
        # update the LinkManager, so all the DerivedComponents
        # for each data set are up-to-date

        # add any links in the data
        for d in self._data:
            for derived in d.derived_components:
                self._link_manager.add_link(d.get_component(derived).link)
            for link in d.coordinate_links:
                self._link_manager.add_link(link)


        for d in self._data:
            self._link_manager.update_data_components(d)

    def register_to_hub(self, hub):
        """ Register managed data objects to a hub"""
        if not isinstance(hub, glue.Hub):
            raise TypeError("Input is not a Hub object: %s" % type(hub))
        self.hub = hub

        #re-assign all data, subset hub instances to this hub
        for d in self._data:
            d.register_to_hub(hub)
            for s in d.subsets:
                s.register()

        hub.subscribe(self, glue.message.DataAddComponentMessage,
                      lambda msg: self._sync_link_manager(),
                      filter=lambda x:x.sender in self._data)

    def __contains__(self, obj):
        return obj in self._data

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)
