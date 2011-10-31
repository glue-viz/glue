from cloudviz.hub import HubListener
import cloudviz.message as msg
from cloudbiz.subset import RoiSubset

class SubsetLink(HubListener):
    """ The base class for representing subsets linked across data.

    Subsets are attached to specific datasets. In many applications,
    however, logical connections exist between datasets. For example,
    two datasets may both describe objects in the same area of the
    sky. This class facilitates syncing subset descriptions from
    different data, in situations where such syncing is logically
    well-defined.

    A SubsetLink object is created with references to one subset in
    two or more datasets. It subscribes to the hub to receive messages
    when any of these subsets are modified. At this point, it alters
    the subsets in the other datasets to logically duplicate the
    original change. In this way, the subsets across different data
    sets are linked.
    """
    

    def __init__(self, subsets):
        """ Create a new link instance
        
        Parameters
        ==========
        subsets : A list of class:`cloudviz.subset.Subset` instances
           The subsets to link together
        """
        self._subsets = subsets
        self._listen = True

    def register_to_hub(self, hub):
        """
        Register the link object to the hub, to receive messages when
        any subset is updated.

        Parameters
        ==========
        hub: class:`cloudviz.hub.Hub` instance
             The hub to register to
        """
        hub.subscribe(self, 
                      msg.SubsetUpdateMessage,
                      filter=lambda x: \
                          x.sender in self._subsets)
                
    def notify(self, message):
        """ Message handling event when one of the subsets is updated.
        
        This class calls the convert method, which actually modifies
        the appropriate subsets. The extra notify method is needed to
        temporarly disable message processing from the hub. If this
        step isn't done, then SubsetLinks would create infinite message
        loops.

        Parameters:
        ===========
        message: class:`cloudviz.message.Message` instance
           The message sent from the hub
        """
        if not self._listen:
            return
        self._listen = False
        if message.sender in self._subsets:
            self.convert(message)
        else:
            raise TypeError("Linker can't handle message sent from %s " %
                            message.sender)
        self._listen = True

    def convert(self, message):
        """ Updates the description of the target """
        raise NotImplementedError()


class RoiLink(SubsetLink):
    """ A class:`cloudviz.subset_link.SubsetLink` object that links
    RoiSubsets by syncing their roi descriptions
    """

    def __init__(self, subsets):
        SubsetLink.__init__(self, subsets)
        for s in subsets:
            if not isinstance(s, RoiSubset):
                raise TypeError("All subsets must be ROI subsets")

    def convert(self, message):
        for s in self.subsets:
            if s.roi is not message.sender.roi:
                s.roi = message.sender.roi
