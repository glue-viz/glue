import numpy as np

import cloudviz


class Subset(object):
    """Base class to handle subsets of data.

    These objects both describe subsets of a dataset, and relay any
    state changes to the hub that their parent data are assigned to.

    This base class only directly impements the logic that relays
    state changes back to the hub. Subclasses implement the actual
    description and manipulation of data subsets

    Attributes:
    -----------
    data: data instance
        The dataset that this subset describes
    style: dict
        A dictionary of visualization style properties (e.g., color)
        Clients are free to use this information when making plots.
    """

    class ListenDict(dict):
        """
        A small dictionary class to keep track of visual subset
        properties. Updates are broadcasted through the subset

        """
        def __init__(self, subset):
            dict.__init__(self)
            self._subset = subset

        def __setitem__(self, key, value):
            dict.__setitem__(self, key, value)
            self._subset.broadcast(self)

    def __init__(self, data):
        """ Create a new subclass object.

        """
        self.data = data
        self._broadcasting = False
        self.style = self.ListenDict(self)
        self.style['color'] = 'r'

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

        This method must be overridden by subclasses

        Returns:
        --------

        A numpy array, giving the indices of elements in the data that
        belong to this subset.
        """
        raise NotImplementedError("must be overridden by a subclass")

    def to_mask(self):
        """
        Convert the current subset to a mask.

        Returns:
        --------

        A boolean numpy array, the same shape as the data, that
        defines whether each element belongs to the subset.
        """
        raise NotImplementedError("must be overridden by a subclass")

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

        Note that in most situations, broadcasting happens
        automatically.

        Parameters:
        -----------
        attribute: string
                   The name of the attribute (if any) that should be
                   broadcast as updated.
        """

        try:
            if self._broadcasting and self.data.hub:
                msg = cloudviz.message.SubsetUpdateMessage(self,
                                                           attribute=attribute)
                self.data.hub.broadcast(msg)
        except (AttributeError):
            pass

    def unregister(self):
        """Broadcast a SubsetDeleteMessage to the hub, and stop braodcasting"""

        try:
            if self._broadcasting and self.data.hub:
                msg = cloudviz.message.SubsetDeleteMessage(self)
                self.data.hub.broadcast(msg)
        except (AttributeError):
            pass
        self._broadcasting = False

    def __del__(self):
        self.unregister()

    def __setattr__(self, attribute, value):
        object.__setattr__(self, attribute, value)
        self.broadcast(attribute)


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
    def __init__(self, data, node_list=None):
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

        Subset.__init__(self, data)
        if not node_list:
            self.node_list = []
        else:
            self.node_list = node_list

    def to_mask(self):
        t = self.data.tree
        im = t.index_map
        mask = np.zeros(self.data.shape, dtype=bool)
        for n in self.node_list:
            mask |= im == n
        return mask

    def to_index_list(self):
        # this is inefficient for small subsets.
        return self.to_mask().nonzero()[0]


class ElementSubset(Subset):
    """
    This is a simple subset object that explicitly defines
    which elements of the data set are included in the subset

    Attributes:
    -----------

    mask: A boolean numpy array, the same shape as the data.
          The true/false value determines whether each element
          belongs to the subset.
    """

    def __init__(self, data, mask=None):
        """
        Create a new subset object.

        Parameters:
        -----------
        data: data instance.
              The data to attach this subset to

        mask: Numpy array
              The mask attribute for this subset
        """
        if not mask:
            self.mask = np.zeros(data.shape, dtype=bool)
        else:
            self.mask = mask
        Subset.__init__(self, data)

    def to_mask(self):
        return self.mask

    def to_index_list(self):
        return self.mask.nonzero()[0]

    def __setattr__(self, attribute, value):
        if hasattr(self, 'mask') and attribute == 'mask':
            if value.shape != self.data.shape:
                raise Exception("Mask has wrong shape")
        Subset.__setattr__(self, attribute, value)
