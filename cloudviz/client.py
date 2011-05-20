class Client(object):
    """
    Base class for interaction / visualization modules

    Attributes:
    """

    _hub = None
    _data = None
    _event_listner = None
    __frozen = False

    def __init__(self, hub, data):
        """Create a new client object.
        
        Attributes:
        hub: The hub that this client should belong to
        data: The primary data associated with this client. 

        Raises:
        TypeError: If the hub and data inputs are not Hub and Data objects.
        """
        if (not isinstance(hub, Hub)):
            raise TypeError("Input hub is not a Hub object: %s" % type(hub))
        if (not isinstance(data, Data)):
            raise TypeError("Input data is not a Data object: %s" % type(data))
                           
        self._hub = hub
        self._data = data
        self.__frozen = True

    def update_subset(self, subset, attr = None, new = False, delete = False):
        if (subset.data != self._data):
            raise TypeError("Input subset describes the wrong data set")
        if (new): self.add_subset(subset)
        if (delete): self.delete_subset(subset)
        if (attr): self.refresh_subset(subset, attr)

    def add_subset(self, subset):
        pass

    def delete_subset(self, subset):
        pass

    def refresh_subset(self, subset, attr):
        pass

    def __setattr__(self, name, value):
        if (self.__frozen and (name in ["hub", "data"])):
            raise AttributeError("Cannot modify client's hub or data assignment after creation");
        if name == "event_listener" and (not isinstance(value, EventListener)):
            raise TypeError("Input is not an event listener object: %s" % type(value))
        object.__setattr__(self, name, value)
