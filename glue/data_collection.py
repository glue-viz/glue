import glue


class DataCollection(object):
    """ DataCollections manage sets of data for Clients.

    In addition to storing data, DataCollections have an "active"
    object (either a data or subset object) that clients can use as
    the "editable" components of a user interaction.  DataCollections
    will broadcast messages when the active object is reassigned.
    """

    def __init__(self, data=None):
        self._active = None
        self.hub = None

        self._data = []
        if isinstance(data, glue.data.Data):
            self._data = [data]
            self._active = data

        elif isinstance(data, list):
            self._data = data
            self._active = data[0]

        self._links = []

    def add_link(self, link):
        self._links.append(link)
        if self.hub:
            link.register_to_hub(self.hub)

    def remove_link(self, link):
        if link not in self._links:
            return
        self._links.remove(link)
        self.hub.remove(link)

    def append(self, data):
        self._data.append(data)
        if self.hub:
            data.hub = self.hub
            for s in data.subsets:
                s.register()
            msg = glue.message.DataCollectionAddMessage(self, data)
            self.hub.broadcast(msg)
        if len(self._data) == 1:
            self._active = data

    def remove(self, data):
        if data not in self._data:
            return
        self._data.remove(data)
        if self.hub:
            msg = glue.message.DataCollectionDeleteMessage(self, data)
            self.hub.broadcast(msg)
        if data == self.active:
            self._active = None

    def get(self, index):
        if index < len(self._data):
            return self._data[index]
        raise IndexError("index is greater than number of data sets")

    def all_data(self):
        return self._data

    @property
    def active_data(self):
        return self._active.data if self._active else None

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, new):
        if isinstance(new, glue.Subset):
            if new not in [s for d in self._data for s in d.subsets]:
                raise TypeError("Object not in data collection: %s" % new)
        if isinstance(new, glue.Data):
            if new not in self._data:
                raise TypeError("Object not in data collection: %s" % new)

        changed = self.active != new

        old_data = self.active.data if self._active else None
        data = new.data
        self._active = new

        if changed and self.hub is not None:
            msg = glue.message.DataCollectionActiveChange(self)
            self.hub.broadcast(msg)

        if data != old_data and self.hub is not None:
            msg = glue.message.DataCollectionActiveDataChange(self)
            self.hub.broadcast(msg)

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
