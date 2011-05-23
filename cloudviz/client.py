import cloudviz


class Client(object):
    """
    Base class for interaction / visualization modules

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

        self.data = data
        self._event_listener = None

    def update_subset(self, subset, attr=None, new=False, delete=False):

        if new:
            self._add_subset(subset)

        if delete:
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
