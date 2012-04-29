import numpy as np
import pyfits

import glue
from glue.exceptions import IncompatibleDataException
from glue.visual import VisualAttributes

class Subset(object):
    """Base class to handle subsets of data.

    These objects both describe subsets of a dataset, and relay any
    state changes to the hub that their parent data are assigned to.

    This base class only directly impements the logic that relays
    state changes back to the hub. Subclasses implement the actual
    description and manipulation of data subsets

    Attributes:
    -----------
    data : data instance
        The dataset that this subset describes
    style : VisualAttributes instance
        Describes visual attributes of the subset
    """

    def __init__(self, data, color=None, alpha=1.0, label=None):
        """ Create a new subclass object.

        """
        self._broadcasting = False
        self.data = data
        self.style = VisualAttributes(parent=self)
        if color:
            self.style.color = color
        self.style.alpha = alpha
        self.style.label = label
        self._subset_state = None
        self.subset_state = SubsetState()

    @property
    def subset_state(self):
        return self._subset_state

    @subset_state.setter
    def subset_state(self, state):
        state.parent = self
        self._subset_state = state


    @property
    def label(self):
        """ Convenience access to subset's label """
        return self.style.label

    def register(self):
        """ Register a subset to its data, and start broadcasting
        state changes

        """
        self.data.add_subset(self)
        self.do_broadcast(True)

    def to_index_list(self):
        """
        Convert the current subset to a list of indices. These index
        the elements in the data object that belong to the subset.

        By default, this uses the output from to_mask.

        Returns:
        --------

        A numpy array, giving the indices of elements in the data that
        belong to this subset.

        Raises:
        -------
        IncompatibleDataException: if an index list cannot be created
        for the requested data set.

        """
        return self.subset_state.to_index_list()

    def to_mask(self):
        """
        Convert the current subset to a mask.

        Returns:
        --------

        A boolean numpy array, the same shape as the data, that
        defines whether each element belongs to the subset.
        """
        return self.subset_state.to_mask()

    def do_broadcast(self, value):
        """
        Set whether state changes to the subset are relayed to a hub.

        It can be useful to turn off broadcasting, when modifying the
        subset in ways that don't impact any of the clients.

        Attributes:
        value: Whether the subset should broadcast state changes (True/False)

        """
        object.__setattr__(self, '_broadcasting', value)

    def broadcast(self, attribute=None):
        """
        Explicitly broadcast a SubsetUpdateMessage to the hub

        Parameters:
        -----------
        attribute: string
                   The name of the attribute (if any) that should be
                   broadcast as updated.
        """
        if not hasattr(self, 'data') or not hasattr(self.data, 'hub'):
            return
        if not hasattr(self, '_broadcasting'):
            return

        if self._broadcasting and self.data.hub:
            msg = glue.message.SubsetUpdateMessage(self,
                                                       attribute=attribute)
            self.data.hub.broadcast(msg)

    def unregister(self):
        """Broadcast a SubsetDeleteMessage to the hub, and stop braodcasting"""
        if not hasattr(self, 'data') or not hasattr(self.data, 'hub'):
            return
        if not hasattr(self, '_broadcasting'):
            return

        dobroad = self._broadcasting and self.data.hub
        self.do_broadcast(False)
        if dobroad:
            msg = glue.message.SubsetDeleteMessage(self)
            self.data.hub.broadcast(msg)

    def write_mask(self, file_name, format="fits"):
        """ Write a subset mask out to file

        Inputs:
        -------
        file_name: String
                   name of file to write to
        format: String
                Name of format to write to. Currently, only "fits" is
                supported

        """
        mask = np.short(self.to_mask())
        if format == 'fits':
            pyfits.writeto(file_name, mask, clobber=True)
        else:
            raise AttributeError("format not supported: %s" % format)

    def __del__(self):
        self.unregister()
        super(Subset, self).__del__()

    def __setattr__(self, attribute, value):
        object.__setattr__(self, attribute, value)
        if attribute != '_braodcasting':
            self.broadcast(attribute)

    def __getitem__(self, attribute):
        il = self.to_index_list()
        if len(il) == 0:
            return np.array([])
        data = self.data[attribute]
        return data[il]

    def paste(self, other_subset):
        """paste subset state from other_subset onto self """
        state = other_subset.subset_state.copy()
        state.parent = self
        self.subset_state = state

class SubsetState(object):
    def __init__(self):
        self.parent = None

    def to_index_list(self):
        return np.where(self.to_mask())[0]

    def to_mask(self):
        return np.zeros(self.parent.data.shape, dtype=bool)

    def copy(self):
        return SubsetState()

    def __or__(self, other_state):
        return OrState(self, other_state)

    def __and__(self, other_state):
        return AndState(self, other_state)

    def __invert__(self):
        return InvertState(self)

    def __xor__(self, other_state):
        return XorState(self, other_state)

class RoiSubsetState(SubsetState):
    def __init__(self):
        super(RoiSubsetState, self).__init__()
        self.xatt = None
        self.yatt = None
        self.roi = None

    def to_mask(self):
        x = self.parent.data[self.xatt]
        y = self.parent.data[self.yatt]
        result = self.roi.contains(x, y)
        return result

    def copy(self):
        result = RoiSubsetState()
        result.xatt = self.xatt
        result.yatt = self.yatt
        result.roi = self.roi
        return result

class CompositeSubsetState(SubsetState):
    def __init__(self, state1, state2=None):
        super(CompositeSubsetState, self).__init__()
        self.state1 = state1
        self.state2 = state2

class OrState(CompositeSubsetState):
    def to_mask(self):
        return self.state1.to_mask() | self.state2.to_mask()

class AndState(CompositeSubsetState):
    def to_mask(self):
        return self.state1.to_mask() & self.state2.to_mask()

class XorState(CompositeSubsetState):
    def to_mask(self):
        return self.state1.to_mask() ^ self.state2.to_mask()

class InvertState(CompositeSubsetState):
    def to_mask(self):
        return ~self.state1.to_mask()

class TreeSubset(Subset):
    """ Subsets defined using a data's Tree attribute.

    The tree attribute in a data set defines a hierarchical
    partitioning of the data. This subset class represents subsets
    that consist of portions of this tree (i.e., a subset of the nodes
    in the tree)

    Attributes:
    -----------
    node_list: A list of integers, specifying which nodes in the tree
               belong to the subset.
    """
    def __init__(self, data, node_list=None, **kwargs):
        """ Create a new subset instance

        Parameters:
        -----------
        data: A data instance
              The data must have a tree attribute with a populated index_map

        node_list: List
                  A list of node ids, defining which parts of the data's tree
                  belong to this subset.
        """
        if not hasattr(data, 'tree'):
            raise AttributeError("Input data must contain a tree object")

        if data.tree.index_map is None:
            raise AttributeError("Input data's tree must have an index map")

        Subset.__init__(self, data, **kwargs)
        if not node_list:
            self.node_list = []
        else:
            self.node_list = node_list

    def to_mask(self, data=None):

        if data is not None and data is not self.data:
            raise IncompatibleDataException("TreeSubsets cannot cross "
                                            "data sets")

        t = self.data.tree
        im = t.index_map
        mask = np.zeros(self.data.shape, dtype=bool)
        for n in self.node_list:
            mask |= im == n
        return mask

    def to_index_list(self, data=None):

        if data is not None and data is not self.data:
            raise IncompatibleDataException("TreeSubsets cannot cross"
                                            " data sets")

        # this is inefficient for small subsets.
        return self.to_mask().nonzero()[0]
