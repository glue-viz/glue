from .hub import HubListener
from .message import SubsetUpdateMessage
from .message import LiveLinkAddMessage
from .message import LiveLinkDeleteMessage


class LiveLinkManager(object):
    """ A collection to create, store, and remove LiveLinks

    Broadcasts LiveLinkAddMessage and LiveLinkDeleteMessage
    """

    def __init__(self, hub=None):
        self.hub = hub
        self._links = []

    @property
    def links(self):
        return self._links

    def add_link_between(self, *subsets):
        """ Create a LiveLInk for the input :clsss:`~glue.core.Subset` objects,
        and add to manager
        """
        if self.hub is None:
            raise TypeError("Hub attribute is null -- cannot create links")

        result = LiveLink(subsets)
        result.register_to_hub(self.hub)
        self._links.append(result)
        msg = LiveLinkAddMessage(self, result)
        self.hub.broadcast(msg)
        return result

    def remove_links_from(self, subset):
        """ Find remove, and unregister all LiveLinks involving subset """
        if self.hub is None:
            raise TypeError("hub attribute is None. Cannot delete links")

        for link in list(self._links):
            if subset in link.subsets:
                self._links.remove(link)
                link.unregister(self.hub)
                msg = LiveLinkDeleteMessage(self, link)
                self.hub.broadcast(msg)

    def has_link(self, subset):
        for link in self._links:
            if subset in link.subsets:
                return True
        return False


class LiveLink(HubListener):
    """ An object to keep subsets in sync """

    def __init__(self, subsets):
        """ Create a new link instance

        :param subsets:
        A list of class:`~glue.core.subset.Subset` instances to link
        """
        super(LiveLink, self).__init__()
        self._subsets = subsets
        self._listen = True
        self.sync(self._subsets[0])

    @property
    def subsets(self):
        return self._subsets

    def register_to_hub(self, hub):
        """
        Register the link object to the hub, to receive messages when
        any subset is updated.

        :param hub:  The hub to register to
        :type hub: :class:`~glue.core.hub.Hub` instance
        """
        def subset_in_link(message):
            return message.sender in self._subsets

        hub.subscribe(self,
                      SubsetUpdateMessage,
                      filter=subset_in_link)

    def sync(self, reference, attribute=None):
        """ Sync all tracked subsets to a reference

        :param reference: The subset to sync to
        :param attribute: The subset attribute that was updated
        """
        state, style = reference.subset_state, reference.style

        for subset in self._subsets:
            if subset is reference:
                continue
            if attribute != 'style':
                subset.subset_state = state.copy()
            if attribute != 'subset_state':
                subset.style = style.copy()

    def notify(self, message):
        """Sync subset states when a SubsetUpdateMessage is called

        This method is called by the hub whenever a relevant subset
        is modified. It updates all other relevant subsets
        to match the update.

        :param message: The message sent from the hub
        :type message: :class:`~glue.core.message.Message` instance

        """
        if not self._listen:
            return
        assert message.sender in self._subsets, "Hub filter error"

        self._listen = False
        self.sync(message.sender, message.attribute)
        self._listen = True
