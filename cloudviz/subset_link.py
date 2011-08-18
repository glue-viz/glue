from cloudviz.hub import HubListener
import cloudviz.message as msg


class SubsetLink(HubListener):
    """ The base class for representing subsets linked across data.

    Subsets are attached to specific datasets. In many applications,
    however, logical connections exist between datasets. For example,
    two datasets may both describe objects in the same area of the
    sky. This class facilitates syncing subset descriptions from
    different data, in situations where such syncing logically
    well-defined.

    Here is the typical usage pattern: two datasets (d1 and d2) each
    have one subset (s1 and s2, respectively). A SubsetLink object is
    created with references to s1 and s2. It then subscribes to a hub,
    to receive messages anytime either s1 or s2 are updated. When the
    link receives one of these messages (say, when s1 is updated), it
    implements the logic to echo this modificaion to the other subset
    (i.e., s2 is modified to reflect s1's new description).
    """
    

    def __init__(self, s1, s2):
        """ Create a new link instance
        
        Parameters
        ==========
        s1: class:`cloudviz.subset.Subset` instance
            The first subset to link
        s2: class:`cloudviz.subset.Subset` instance
            The second subset to link
        """
        self._s1 = s1
        self._s2 = s2
        self._listen = True

    def register_to_hub(self, hub):
        """
        Register the link object to the hub, to receive messages when
        either subset is updated.

        Parameters
        ==========
        hub: class:`cloudviz.hub.Hub` instance
             The hub to register to
        """
        hub.subscribe(self, 
                      msg.SubsetUpdateMessage,
                      filter=lambda x: \
                          x.sender in [self._s1, self._s2])
                
    def notify(self, message):
        """ Message handling event when one of the subsets is updated.
        
        This class calls the convert method

        Parameters:
        ===========
        message: class:`cloudviz.message.Message` instance
           The message sent from the hub
        """
        if not self._listen:
            return
        self._listen = False
        if message.sender is self._s1:
            self.convert(self._s1, self._s2)
        elif message.sender is self._s2:
            self.convert(self._s2, self._s1)
        else:
            raise TypeError("Linker can't handle message sent from %s " %
                            message.sender)
        self._listen = True

    def convert(self, source, target):
        """ Updates the description of the target """
        raise NotImplementedError()
