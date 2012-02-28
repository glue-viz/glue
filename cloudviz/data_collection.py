import cloudviz as cv

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
        if isinstance(data, cv.data.Data):
            self._data = [data]
            self._active = data

        elif isinstance(data, list):
            self._data = data
            self._active = data[0]


    def append(self, data):
        self._data.append(data)
        if self.hub:
            msg = cv.message.DataCollectionAddMessage(self, data)
            self.hub.broadcast(msg)
        if len(self._data) == 1: self._active = data

    def remove(self, data):
        self._data.remove(data)
        if self.hub:
            msg = cv.message.DataCollectionDeleteMessage(self, data)
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
        if isinstance(new, cv.Subset):
            if new not in [s for d in self._data for s in d.subsets]:
                raise TypeError("Object not in data collection: %s" % new)
        if isinstance(new, cv.Data):
            if new not in self._data:
                raise TypeError("Object not in data collection: %s" % new)

        changed = self.active != new

        old_data = self.active.data if self._active else None
        data = new.data
        self._active = new
        if changed and self.hub is not None:
            msg = cv.message.DataCollectionActiveChange(self)
            self.hub.broadcast(msg)
        if data != old_data and self.hub is not None:
            msg = cv.message.DataCollectionActiveDataChange(self)
            self.hub.broadcast(msg)


    def register_to_hub(self, hub):
        if not isinstance(hub, cv.Hub):
            raise TypeError("Input is not a Hub object: %s" % type(hub))
        self.hub = hub

    def __contains__(self, obj):
        return obj in self._data

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)
