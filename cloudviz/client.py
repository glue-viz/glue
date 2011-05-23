import cloudviz


class Client(object):
    """
    Base class for interaction / visualization modules

    Attributes
    ----------
    data: Data instance
        The primary data associated with this client.

    """

    def __init__(self, hub, data):
        """
        Create a new client object.

        Parameters
        ----------
        hub: Hub instance
            The hub that this client should belong to.
        data: Data instance
            The primary data associated with this client.

        Raises
        ------
        TypeError: If the hub and data inputs are not Hub and Data objects.
        """

        if not isinstance(hub, cloudviz.Hub):
            raise TypeError("Input hub is not a Hub object: %s" % type(hub))

        if not isinstance(data, cloudviz.Data):
            raise TypeError("Input data is not a Data object: %s" % type(data))

        self._hub = hub
        self.data = data
        self._frozen = True
        self._event_listener = None

    def update_subset(self, subset, attr=None, new=False, delete=False):

        if new is not None:
            self._add_subset(subset)

        if delete is not None:
            self._delete_subset(subset)

        if attr is not None:
            self._refresh_subset(subset, attr)

    def _add_subset(self, subset):
        pass

    def _delete_subset(self, subset):
        pass

    def _refresh_subset(self, subset, attr):
        pass

    def __setattr__(self, name, value):

        # Check if hub or data have already been set
        if name in ["_hub", "data"]:
            if hasattr(self, '_frozen') and self._frozen:
                raise AttributeError("Cannot modify client's hub or data"
                                     "assignment after creation")

        # Check type of event listener
        if name == "event_listener":
            if not isinstance(value, cloudviz.EventListener):
                raise TypeError("Input is not an event listener object: %s" %
                                type(value))

        object.__setattr__(self, name, value)
