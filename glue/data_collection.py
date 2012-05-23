import glue


class DataCollection(object):
    """ DataCollections manage sets of data for Clients.
    """

    def __init__(self, data=None):
        self.hub = None
        self._link_manager = glue.LinkManager()

        self._data = []
        if isinstance(data, glue.data.Data):
            self.append(data)
            self._data = [data]
        elif isinstance(data, list):
            for d in data:
                self.append(d)

    def append(self, data):
        self._data.append(data)
        self._sync_layer_manager(data)
        if self.hub:
            data.hub = self.hub
            for s in data.subsets:
                s.register()
            msg = glue.message.DataCollectionAddMessage(self, data)
            self.hub.broadcast(msg)

    def remove(self, data):
        if data not in self._data:
            return
        self._data.remove(data)
        if self.hub:
            msg = glue.message.DataCollectionDeleteMessage(self, data)
            self.hub.broadcast(msg)

    def _sync_layer_manager(self, data):
        pass

    def get(self, index):
        if index < len(self._data):
            return self._data[index]
        raise IndexError("index is greater than number of data sets")

    def all_data(self):
        return self._data

    def register_to_hub(self, hub):
        if not isinstance(hub, glue.Hub):
            raise TypeError("Input is not a Hub object: %s" % type(hub))
        self.hub = hub

        #re-assign all data, subset hub instances to this hub
        for d in self._data:
            d.register_to_hub(hub)
            for s in d.subsets:
                s.register()

    def __contains__(self, obj):
        return obj in self._data

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)
