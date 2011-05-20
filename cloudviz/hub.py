class Hub(object):
    """The hub manages the communication between visualization clients.

    Ths hub holds references to 0 or more client objects, and 0 or 1
    translator object. When subsets within clients are modified (e.g,,
    via a client), these changes are automatically broadcast to all
    the clients within the hub that share that data. In addition, the
    translator can attempt to translate a subset across datasets if
    requested.

    Attributes:
    translator: The optional translator object

    """

    # collection of viz/interaction clients
    _clients = []

    # Translator object will translate subsets across data sets
    translator = None

    # do not allow more than MAX_CLIENTS clients
    _MAX_CLIENTS = 50

    def __init__(self):
        """Create an empty hub."""
        pass

    def __setattr__(self, name, value):
        if name == "translator" and not isinstance(value, Translator):
            raise AttributeError("input is not a Translator object: %s" %
                                  type(value))
        object.__setattr__(self, name, value)

    def add_client(self, client):
        """ Register a new client with the hub

        This will also attach the client's data attribute to the 
        hub. Trying to attach the same data set to multiple,
        different hubs will cause an error.

        Attributes:
        client: The new client to add. Must be a Client object

        Raises:
        TypeError: If the input is not a Client object
        Exception: If too many clients are added.

        """
        
        if (len(self._clients) == self._MAX_CLIENTS):
            raise AttributeError("Exceeded maximum number of clients: %i" %
                            self._MAX_CLIENTS)
        if (not isinstance(client, Client)):
            raise Exception("Input is not a Client object: %s" %
                            type(client))
        
        self._clients.append(client)
        client.data.hub = self

    def remove_client(self, client):
        """Remove a client from the hub.

        This stops any communication between the hub and client.

        Attributes:
        client: The client to remove
        """
        try:
            self._clients.remove(client)
        except ValueError:
            pass

    def broadcast_subset_update(self, subset,
                                attr=None,
                                new=False,
                                delete=False):
        """Communicate to relevant clients that a subset has changed.

        For each client using the same dataset as the input subset,
        the hub will issue a call to update_subset.

        Attributes:

        subset: The subset object that has been modified
        attr: The name of the attribute that has been changed, if any
        new: Set to True if the subset is being created delete: Set to
        True if the subset is being deleted

        """

        d = subset.data
        cs = (c for c in self._clients if c.data == d)
        for c in cs:
            c.update_subset(subset, attr=attr, new=new, delete=delete)

    def translate_subset(self, subset, *args, **kwargs):
        """Translate a subset to all clients, even if they use different data.

        The translator object attempts to translate the subset for each unique
        dataset used by the clients. If successful, the translated subset is
        added to the appropriate data set.

        If the translator cannot translate the subset to a given dataset,
        it is quietly skipped. If there is no translator in the hub, the
        method quietly exits.

        Attributes:
        subset: The subset to translate
        *args: Extra arguments to pass to the translator
        **kwargs: Extra keywords to pass to the translator

        """
        if (not self.translator):
            return

        data = [c.data for c in self._clients]
        data = list(set(data))  # remove duplicates
        for d in data:
            new_subset = self.translator.translate(subset, d, *args, **kwargs)
            if (new_subset):
                d.add_subset(new_subset)
