from .hub import HubListener
from .message import SubsetUpdateMessage

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

    def sync(self, reference):
        """ Sync all tracked subsets to a reference

        :param reference: The subset to sync to
        """
        state, style = reference.subset_state, reference.style

        for subset in self._subsets:
            if subset is reference:
                continue
            subset.subset_state = state
            subset.style = style


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
        self.sync(message.sender)
        self._listen = True
