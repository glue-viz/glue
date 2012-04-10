import cloudviz
import cloudviz.message as msg


class Client(cloudviz.HubListener):
    """
    Base class for interaction / visualization modules

    Attributes
    ----------
    data: DataCollection instance
        The data associated with this client.

    """

    def __init__(self, data=None):
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
        if isinstance(data, cloudviz.Data):
            self._data = cloudviz.DataCollection(data)
        elif isinstance(data, list):
            self._data = cloudviz.DataCollection(data)
        elif isinstance(data, cloudviz.DataCollection):
            self._data = data
        else:
            raise TypeError("Input data must be a Data object, "
                            "list of Data, or DatCollection: %s"
                            % type(data))

    @property
    def data(self):
        """ Returns the data collection """
        return self._data

    def register_to_hub(self, hub):
        """The main method to establish a link with a hub,
        and set up event handlers.

        This method subscribes to 2 basic message types:
        SubsetUpdateMessage and DataUpdateMessage. It defines filter
        methods so that only messages related to the client's data
        sets' are relayed. These 4 messages are relayed to the
        _update_subset, and _update_data methods, respectively

        Client subclasses at a minimum should override these methods
        to provide functionality. They can also override
        register_to_hub to add additional event handlers.

        Attributes
        ----------
        hub: The hub to subscribe to

        """
        dfilter = lambda x:x.sender.data in self._data
        dcfilter = lambda x:x.sender is self._data

        hub.subscribe(self,
                      msg.SubsetCreateMessage,
                      handler=self._add_subset,
                      filter=dfilter)
        hub.subscribe(self,
                      msg.SubsetUpdateMessage,
                      handler=self._update_subset,
                      filter=dfilter)
        hub.subscribe(self,
                      msg.SubsetDeleteMessage,
                      handler=self._remove_subset,
                      filter=dfilter)
        hub.subscribe(self,
                      msg.DataUpdateMessage,
                      handler=self._update_data,
                      filter=dfilter)
        hub.subscribe(self,
                      msg.DataCollectionDeleteMessage,
                      handler=self._remove_data,
                      filter=dcfilter)

    def _remove_data(self, message):
        raise NotImplementedError("_remove_data not implemented")

    def _remove_subset(self, message):
        raise NotImplementedError("_remove_data not implemented")

    def _update_data(self, message):
        """ Default handler for DataMessage """
        raise NotImplementedError("_update_data not implemented")

    def _update_subset(self, message):
        """ Default handler for SubsetUpdateMessage """
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

        object.__setattr__(self, name, value)
