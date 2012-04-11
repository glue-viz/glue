import glue
from glue.hub import HubListener
import glue.message as msg
from glue.subset import RoiSubset


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
        subsets : A list of class:`glue.subset.Subset` instances
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
        hub: class:`glue.hub.Hub` instance
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
        message: class:`glue.message.Message` instance
           The message sent from the hub
        """
        if not self._listen:
            return
        if message.sender not in self._subsets:
            raise TypeError("Linker can't handle message sent from %s " %
                            message.sender)


        #prob don't want to sync visual properties
        if message.attribute is message.sender.style:
            return

        self._listen = False
        #easy case of propagating style changes
        #if message.attribute is message.sender.style:
        #    for s in self._subsets:
        #        s.style.set(message.attribute)
        #else: # pass off harder cases

        self.convert(message)

        self._listen = True

    def convert(self, message):
        """ Updates the description of the target """
        raise NotImplementedError()

    @property
    def subsets(self):
        return self._subsets


class RoiLink(SubsetLink):
    """ A class:`glue.subset_link.SubsetLink` object that links
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

class HullLink(SubsetLink):
    def __init__(self, subsets, xatts=None, yatts=None):
        SubsetLink.__init__(self, subsets)
        self.xs = {}
        self.ys = {}
        for i,s in enumerate(subsets):
            if not isinstance(s, RoiSubset):
                raise TypeError("All subsets must be ROI subsets")
            self.xs[s] = xatts[i] if xatts else s.xatt
            self.ys[s] = yatts[i] if yatts else s.yatt

    def convert(self, message):
        from scipy.spatial import Delaunay
        import numpy as np
        from glue.roi import PolygonalROI

        indices = message.sender.to_index_list()
        if indices.size == 0:
            roi = PolygonalROI()
            for s in self.subsets:
                s.roi = roi
            return

        x = message.sender.data[self.xs[message.sender]][indices]
        y = message.sender.data[self.ys[message.sender]][indices]
        array = np.zeros((x.size, 2))
        array[:,0] = x
        array[:,1] = y
        tri = Delaunay(array)
        hull = tri.convex_hull
        roi = PolygonalROI()

        used = [0] * hull.shape[0]
        roi.add_point(x[hull[0,0]], y[hull[0,0]])
        used[0] = 1
        look = hull[0,1]
        for i in range(1, len(used)):
            found=False
            for j in range(len(used)):
                if used[j]: continue
                if hull[j,0] == look:
                    found=True
                    look = hull[j,1]
                    roi.add_point(x[hull[j,1]], y[hull[j,1]])
                elif hull[j,1] == look:
                    found = True
                    look = hull[j,0]
                    roi.add_point(x[hull[j,0]], y[hull[j,0]])
                if found:
                    used[j] = 1
                    break
            assert(found)

        for s in self.subsets:
            if s is message.sender: continue
            s.roi = roi



