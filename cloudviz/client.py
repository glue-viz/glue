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

    def update(self, subset=None, attribute=None, action='update'):
        '''
        Update the view in the client. If no arguments are specified, the
        whole view of the data is updated.

        Parameters
        ----------
        subset: Subset instance, optional
            The subset being added/updated/removed
        attribute: str, optional
            The specific data or subset attribute to update
        action: str
            Can be one of add/remove/update. If no subset is specified,
            this should be set to 'update'.
        '''

        if subset is None and action != 'update':
            raise Exception("Cannot specify action=%s if subset=None" % action)

        if subset not in self.data.subsets:
            raise Exception("subset is not part of the "
                            "dataset being shown by this client")

        if action == 'add':
            self._add_subset(subset)
        elif action == 'remove':
            self._remove_subset(subset)
        elif action == 'update':
            if subset is None:
                self._update_all(attribute=attribute)
            else:
                self._update_subset(subset, attribute=attribute)
        else:
            raise Exception("Unknown action: %s (should be one of "
                            "add/remove/update)" % action)

    def _add_subset(self, subset):
        pass

    def _remove_subset(self, subset):
        pass

    def _update_all(attribute=None):
        pass

    def _update_subset(self, subset, attribute=None):
        pass

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
