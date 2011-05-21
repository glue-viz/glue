from data import Data

class Client(object):
    """
    Base class for interaction / visualization modules

    Attributes:
    data: The data associated with this client
    """

    def __init__(self, hub, data):
        """Create a new client object.
        
        Attributes:
        hub: The hub that this client should belong to
        data: The primary data associated with this client. 

        Raises:
        TypeError: If the hub and data inputs are not Hub and Data objects.
        """
        
        from hub import Hub  # avoid a circular import at header. Kind of hacky
        if (not isinstance(hub, Hub)):
            raise TypeError("Input hub is not a Hub object: %s" % type(hub))
        if (not isinstance(data, Data)):
            raise TypeError("Input data is not a Data object: %s" % type(data))
                           
        self._hub = hub
        self.data = data
        self.__frozen = True
        self._event_listner = None
        
    def update_subset(self, subset, attr = None, new = False, delete = False):
        if (new): self._add_subset(subset)
        if (delete): self._delete_subset(subset)
        if (attr): self._refresh_subset(subset, attr)

    def _add_subset(self, subset):
        pass

    def _delete_subset(self, subset):
        pass

    def _refresh_subset(self, subset, attr):
        pass

    def __setattr__(self, name, value):
        if (self.__frozen and (name in ["_Client__hub", "_Client_data"])):
            raise AttributeError("Cannot modify client's hub or data assignment after creation");
#        if name == "event_listener" and (not isinstance(value, EventListener)):
#            raise TypeError("Input is not an event listener object: %s" % type(value))
        object.__setattr__(self, name, value)
