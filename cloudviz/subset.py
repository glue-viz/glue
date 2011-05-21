class Subset(object):
    """Base class to handle subsets of data.

    These objects both describe subsets of a dataset, and relay any
    state changes to the hub that their parent data are assigned to.

    This base class only directly impements the logic that relays 
    state changes back to the hub. Subclasses implement the actual
    description and manipulation of data subsets

    Attributes:
    data: The dataset that this subset describes
    """

    def __init__(self, data):
        """ Create a new subclass object.

        This method should always be called by subclasses. It attaches
        data to the subset, and starts listening for state changes to 
        send to the hub

        Attributes:
        data: A data set that this subset will describe

        """

        self.data = data
        hub = self.data.hub
        if(hub):
            hub.broadcast_subset_update(self, new=True)
        self.__broadcasting = True

        self.data = None
        
        # whether state changes should be sent to the hub
        self.__broadcasting = False

    def do_broadcast(value):
        """ Set whether state changes to the subset are relayed to a hub.

        It can be useful to turn off broadcasting, when modifying the 
        subset in ways that don't impact any of the clients.

        Attributes:
        value: Whether the subset should broadcast state changes (True/False)

        """
        object.__setattr__(self, '_Subset__broadcasting', value)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if (not self.__broadcasting): return
        hub = self.data.hub
        if(hub):
            hub.broadcast_subset_update(self, attr=name)


class TreeSubset(Subset):
    pass


class PixelSubset(Subset):
    pass
