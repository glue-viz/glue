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
        self.data.add_subset(self)  # this will broadcast the message
        self._broadcasting = True

    def do_broadcast(self, value):
        """
        Set whether state changes to the subset are relayed to a hub.

        It can be useful to turn off broadcasting, when modifying the
        subset in ways that don't impact any of the clients.

        Attributes:
        value: Whether the subset should broadcast state changes (True/False)

        """
        object.__setattr__(self, '_broadcasting', value)

    def modify(self, *args, **kwargs):

        # Modify the selection based on arguments

        # Broadcast changes
        if self.data.hub is not None:
            self.data.hub.broadcast(self, action='update')

    def __setattr__(self, attribute, value):
        object.__setattr__(self, attribute, value)
        if not hasattr(self, '_broadcasting') \
           or not self._broadcasting or attribute == '_broadcasting':
            return
        elif self.data is not None and self.data.hub is not None:
            self.data.hub.broadcast(self, attribute=attribute, action='update')

class TreeSubset(Subset):
    pass


class PixelSubset(Subset):
    pass
