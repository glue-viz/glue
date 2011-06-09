import numpy as np


class Tree(object):
    """
    Base class for hierarchical segmentations of data sets.

    The tree is represented by its root node, which contains reference
    to 0 or more children nodes.

    Attributes
    ----------
    id: Integer
          An identifier for this node.
    parent: Tree instance
          A reference to this node's parent, if any
    value:
          A value associated with this node
    children: List of Tre instances
          The children of this node
    index_map: Component instance
          The tree id that each element to which each
          element in the original data belongs.
    """

    def __init__(self, id=None, value=None, index_map=None):
        """
        Create a new Tree object.

        Parameters
        ----------
        id: Integer
              Id of the tree
        value:
              Value of the tree
        index_map: Component instance
              index_map of the data


        Raises
        ------
        TypeError: if any of the inputs are the wrong type
        """
        if (id != None):
            try:
                id = int(id)
            except ValueError:
                raise TypeError("Input id must be in integer")

        self.id = id

        self.value = value

        self.children = []

        self.parent = None

        self.index_map = index_map

        self._index = None

    def add_child(self, child):
        """
        Add a new child node to this tree.

        This is the preferred way for building trees, as it takes care
        of input checking and linking between parent and child. Do not
        append to the children attribute directly.

        Parameters
        ----------
        child: Tree instance
             The child to add

        Raises
        ------
        TypeError: If the input is not a Tree instance
        """

        if (not isinstance(child, Tree)):
            raise TypeError("Child must be a tree instance")

        self.children.append(child)
        child.parent = self

    def to_newick(self):
        """
        Convert the tree to a newick string

        Returns
        -------
        A newick string representation of the tree
        """

        result = ''
        if (self.children):
            result = '(' + ','.join([c.to_newick()[0:-1]
                                     for c in self.children]) + ')'
        if (self.id != None):
            result += ('%s' % self.id)
        if (self.value != None):
            result += (':%s' % self.value)
        return result + ';'

    def index(self):
        """
        Create a flattened index of all the nodes at and below this
        branch, and store them in the _index attribute.

        The _index attribute is a dictionary holding each node in the
        tree, keyed by the node ids. Index will only work if the node
        id's are unique.

        The user of the index is responsible for making sure that the
        tree hasn't changed since the index was created.
        """
        self._index = {}
        stack = [self]
        while stack:
            s = stack.pop()
            if s.id in self._index:
                raise KeyError("Cannot index this tree -- "
                               "node id's are non-unique")
            self._index[s.id] = s
            for c in s.children:
                stack.append(c)

    def get_subtree_indices(self):
        result = []
        stack = [self]
        while stack:
            s = stack.pop()
            result.append(s.id)
            stack += s.children
        return result


class NewickTree(Tree):
    """
    A subclass of Tree, which generates trees from Newick Strings.

    Attributes
    ----------
    newick: The newick string
    """

    def __init__(self, newick, index_map=None):
        """
        Create a new tree from a newick string representation of a
        tree

        Attributes
        ----------
        newick: String
              The newick string
        index_map: Component
              The index map of the data
        """
        self.newick = newick

        self.__validateNewick()
        (id, value) = self.__parse_id_value()
        Tree.__init__(self, index_map=index_map,
                      id=id, value=value)
        self.__parse_children()

    def __validateNewick(self):
        """
        Ensure that the suppied string represents a valid Newick
        description.

        Raises
        ------
        ValueError: If the newick string is invalid
        """
        pass

    def __parse_id_value(self):
        """
        Parse the root node id and value

        Returns
        -------
        The root's id and value, as a list
        """

        newick = self.newick
        first = max([newick.rfind(')'),
                     newick.rfind(',')]) + 1
        comma = newick.find(',', first)
        if comma == -1:
            comma = len(newick) - 1
        paren = newick.find(')', first)
        if paren == -1:
            paren = len(newick) - 1

        last = min([paren, comma])
        mid = newick.find(':', first)

        if (mid != -1):
            id = newick[first:mid]
            value = newick[mid + 1:last]
        else:
            id = newick[first:last]
            value = None
        return (id, value)

    def __parse_children(self):
        """
        Find and parse the children of the root.

        This method recursively builds the tree, and populates the
        root's children attribute.

        Side Effects
        ------------
        Any children currently stored in the root's children list are
        erased.
        """
        newick = self.newick
        if newick[0] != '(':
            return
        depth = 0
        start = 1
        self.children = []
        for i in range(1, len(newick)):
            if (newick[i] == '('):
                depth += 1
            elif (newick[i] == ')' and depth != 0):
                depth -= 1
            elif ((newick[i] == ',' or newick[i] == ')')
                  and depth == 0):
                child = NewickTree(newick[start:i] + ';',
                                   index_map=self.index_map)
                self.add_child(child)
                start = i + 1


class DendroMerge(Tree):
    """
    A dendrogram created from a merge array.

    The merge array is a [nleaf - 1, 2] array where the ith row lists
    the 2 nodes merge to form node nleaf + i.  This data structure is
    used in many older dendrogram creation tools (e.g., that of
    Rosolowsky et al. 2008ApJ...679.1338R)
    """

    def __init__(self, merge_list,
                 index_map=None, _id=-1):
        """
        Create a new DendroMerge tree

        Parameters
        ----------
        merge_list: numpy array
                  a [nleaf - 1, 2] merge list (see class description above)
        index_map: Component
                 See Tree documentation

        """

        if(_id == -1):
            self.validate_mergelist(merge_list)
            nleaf = merge_list.shape[0] + 1
            _id = 2 * nleaf - 2
        else:
            nleaf = merge_list.shape[0] + 1

        Tree.__init__(self, id=_id,
                      index_map=index_map)

        # base case: leaf
        if (_id < nleaf):
            return
        # recursive case: branch. Create children
        else:
            c1 = min(merge_list[_id - nleaf, :])
            c2 = max(merge_list[_id - nleaf, :])
            c1 = DendroMerge(merge_list,
                             index_map=index_map,
                             _id=c1)
            c2 = DendroMerge(merge_list,
                             index_map=index_map,
                             _id=c2)
            self.add_child(c1)
            self.add_child(c2)

    def validate_mergelist(self, merge_list, msg=None):
        """
        Ensure that merge_list is a vlid merge list

        A valid merge_list is a [nleaf - 1, 2] numpy array,
        that includes the numbers 0 through 2 * nleaf - 3
        exactly once.

        Parameters
        ----------
        merge_list: ndarray instance

        Raises
        ------
        TypeError: If the merge_list is invalid
        """

        if (not isinstance(merge_list, np.ndarray)):
            raise TypeError("Invalid mergelist: not a numpy array")
        if (merge_list.shape[1] != 2):
            raise TypeError("Invalid mergelist: not a 2 column array")

        f = merge_list.flatten()
        if (len(f) != len(set(f))):
            raise TypeError("Invalid mergelist: contains duplicates")
        if ((min(f) != 0) or (max(f) != len(f) - 1)):
            raise TypeError("Invalid mergelist: does not "
                            "run from 0-nleaf")
