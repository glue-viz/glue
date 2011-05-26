import cloudviz


class Hub(object):
    """
    The hub manages the communication between visualization clients.

    Ths hub holds references to 0 or more client objects, and 0 or 1
    translator object. When subsets within clients are modified (e.g,,
    via a client), these changes are automatically broadcast to all
    the clients within the hub that share that data. In addition, the
    translator can attempt to translate a subset across datasets if
    requested.

    Attributes
    ----------
    translator: Translator instance, optional
    """

    # do not allow more than MAX_CLIENTS clients
    _MAX_CLIENTS = 50

    def __init__(self):
        """
        Create an empty hub.
        """

        # Collection of viz/interaction clients
        self._clients = []

        # Translator object will translate subsets across data sets
        self.translator = None

    def __setattr__(self, name, value):
        if name == "translator" and hasattr(self, 'translator') and \
           not isinstance(value, cloudviz.Translator):
            raise AttributeError("input is not a Translator object: %s" %
                                 type(value))
        object.__setattr__(self, name, value)

    def add_client(self, client):
        """
        Register a new client with the hub

        This will also attach the client's data attribute to the
        hub. Trying to attach the same data set to multiple,
        different hubs will cause an error.

        Parameters
        ----------
        client: Client instance
            The new client to add.

        Raises
        ------
        TypeError: If the input is not a Client object.
        Exception: If too many clients are added.
        """

        if len(self._clients) == self._MAX_CLIENTS:
            raise AttributeError("Exceeded maximum number of clients: %i" %
                            self._MAX_CLIENTS)

        if not isinstance(client, cloudviz.Client):
            raise Exception("Input is not a Client object: %s" %
                            type(client))

        # Add client to list of clients associated with current hub
        self._clients.append(client)

        # Give the Data instance a pointer to the Hub
        client.data._hub = self

        client.update()

    def remove_client(self, client):
        """
        Remove a client from the hub.

        This stops any communication between the hub and client.

        Parameters
        ----------
        client: Client instance
            The client to remove

        Raises
        ------
        Exception: if client does not exist in hub
        """
        if client in self._clients:
            self._clients.remove(client)
        else:
            raise Exception("Hub does not contain client")

    def broadcast(self, item, attribute=None, action='update'):
        """
        Communicate to relevant clients that things have changed. This can
        be either a whole dataset, a subset, or a specific attribute of a
        dataset or subset.

        Parameters
        ----------
        item: Data or Subset instance
            The data or the subset that has changed
        attribute: str, optional
            The specific data or subset attribute to update
        action: str
            Can be one of add/remove/update. If no subset is specified,
            this should be set to 'update'.
        """

        if isinstance(item, cloudviz.Data):
            data = item
            subset = None
        elif isinstance(item, cloudviz.Subset):
            data = item.data
            subset = item
        else:
            raise Exception("item should be an instance of Data or Subset")

        for client in self._clients:
            if client.data is data:
                client.update(subset=subset,
                              attribute=attribute,
                              action=action)

    def translate_subset(self, subset, *args, **kwargs):
        """
        Translate a subset to all clients, even if they use different data.

        The translator object attempts to translate the subset for each unique
        dataset used by the clients. If successful, the translated subset is
        added to the appropriate data set.

        If the translator cannot translate the subset to a given dataset,
        it is quietly skipped. If there is no translator in the hub, the
        method quietly exits.

        Parameters
        ----------
        subset: Subset instance
            The subset to translate

        Any additional arguments and keyword arguments are passed to the
        translator.
        """
        if self.translator is None:
            return

        data = [c.data for c in self._clients]
        data = list(set(data))  # remove duplicates
        for d in data:
            new_subset = self.translator.translate(subset, d, *args, **kwargs)
            if new_subset is not None:
                d.add_subset(new_subset)
