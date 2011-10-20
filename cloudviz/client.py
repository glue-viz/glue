import cloudviz
import cloudviz.message as msg


class Client(cloudviz.HubListener):
    """
    Base class for interaction / visualization modules

    Subclasses should override the _add_subset, _remove_subset,
    _update_subset, and  _update_all methods.

    Attributes
    ----------
    data: Data instance
        The primary data associated with this client.

    """

    def __init__(self, data):
        """
        Create a new client object.

        Parameters
        ----------
        data: Data instance
            The primary data associated with this client.

        Raises
        ------
        TypeError: If the data input is not a Data instance.
        """

        if not isinstance(data, cloudviz.Data):
            raise TypeError("Input data is not a Data object: %s" % type(data))

        self._data = [data]
        self._event_listener = None

    @property
    def data(self):
        return self._data[0]

    def add_data(self, data):
        if data in self._data: return
        self._data.append(data)

    def get_data(self, index=None):
        if index is not None:
            return self._data[index]
        else:
            return self._data

    def register_to_hub(self, hub):
        """
        The main method to establish a link with a hub,
        and set up event handlers.

        This method subscribes to 4 basic message types:
        SubsetCreateMessage, SubsetUpdateMessage, SubsetRemoveMessage,
        and DataMessage. It defines filter methods so that only
        messages related to self.data are relayed. These 4 messages
        are relayed to the _add_subset, _update_subset,
        _remove_subset, and _update_all methods, respectively

        Client subclasses at a minimum should override these methods
        to provide functionality. They can also override
        register_to_hub to add additional event handlers.

        Attributes
        ----------
        hub: The hub to subscribe to
        """

        hub.subscribe(self,
                      msg.SubsetCreateMessage,
                      handler=self._add_subset,
                      filter=lambda x: \
                          x.sender.data in self._data)
        
        hub.subscribe(self,
                      msg.SubsetUpdateMessage,
                      handler=self._update_subset,
                      filter=lambda x: \
                          x.sender.data in self._data)

        hub.subscribe(self,
                      msg.SubsetDeleteMessage,
                      handler=self._remove_subset,
                      filter=lambda x: \
                          x.sender.data in self._data)

        hub.subscribe(self,
                      msg.DataMessage,
                      handler=self._update_all,
                      filter=lambda x: x.sender in self.data)
        
    def _add_subset(self, message):
        raise NotImplementedError("_add_subset not implemented")

    def _remove_subset(self, message):
        raise NotImplementedError("_remove_subset not implemented")

    def _update_all(self, message):
        raise NotImplementedError("_update_all not implemented")

    def _update_subset(self, message):
        raise NotImplementedError("_update_subset not implemented")

    def select(self):
        """
        General purpose function for selecting a subset
        """

        # Initialize a new empty subset
        subset = self.data.new_subset()

        # Here would be some code for (e.g. GUI) selection, which would
        # define some parameters for the selection, e.g a polygon. The client
        # then calls the following each time the GUI selection changes, and
        # until the user validates the selection:
        subset.modify()

        # Once the section is done, just leave the function

    def __setattr__(self, name, value):

        # Check if data has already been set
        if name == "data" and hasattr(self, "data"):
            raise AttributeError("Cannot modify client's data"
                                 " assignment after creation")

        # Check type of event listener
        if name == "event_listener":
            if not isinstance(value, cloudviz.EventListener):
                raise TypeError("Input is not an event listener object: %s" %
                                type(value))

        object.__setattr__(self, name, value)
