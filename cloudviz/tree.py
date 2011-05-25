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

    def __init__(self, parent=None, id=None, value=None, index_map=None):
        """
        Create a new Tree object.

        Parameters
        ----------
        parent: Tree instance
              The new tree's parent
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

        if (parent and not isinstance(parent, Tree)):
            raise TypeError("Parent must be a tree instance")

        if (id != None):
            try:
                id = int(id)
            except ValueError:
                raise TypeError("Input id must be in integer")

        self.id = id

        self.parent = parent

        self.value = value

        self.children = []

        self.index_map = index_map

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

    def to_subset(self, single=False):
        """
        Convert the current (sub)tree to a subset object

        Parameters
        ----------
        single:
              True to use only the root node as the subset. Otherwise,
              use the root node and all descendents.

        """
        pass

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


class NewickTree(Tree):
    """
    A subclass of Tree, which generates trees from Newick Strings.

    Attributes
    ----------
    newick: The newick string
    """

    def __init__(self, newick, parent=None, index_map=None):
        """
        Create a new tree from a newick string representation of a
        tree

        Attributes
        ----------
        newick: String
              The newick string
        parent: Tree instance
              The parent of this tree
        index_map: Component
              The index map of the data
        """
        self.newick = newick

        self.__validateNewick()
        (id, value) = self.__parse_id_value()
        Tree.__init__(self, parent=parent, index_map=index_map,
                      id = id, value = value)
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
                                   parent=self,
                                   index_map=self.index_map)
                self.children.append(child)
                start = i + 1


class DendroMerge(Tree):
    """
    A dendrogram created from a merge array.

    The merge array is a [nleaf - 1, 2] array where the ith row lists
    the 2 nodes merge to form node nleaf + i.  This data structure is
    used in many older dendrogram creation tools (e.g., that of
    Rosolowsky et al. 2008ApJ...679.1338R)
    """

    def __init__(self, merge_list, parent=None, 
                 index_map=None, _id=-1):
        """
        Create a new DendroMerge tree
        
        Parameters
        ----------
        merge_list: numpy array
                  a [nleaf - 1, 2] merge list (see class description above)
        parent: Tree instance
                Any parent of the root node
        index_map: Component
                 See Tree documentation
       
        """
       
        if(_id == -1):
            if (not self.validate_mergelist(merge_list)):
                raise TypeError("input is not a valid mergelist")    
            nleaf = merge_list.shape[0] + 1
            _id = 2 * nleaf - 2
        else:
            nleaf = merge_list.shape[0] + 1
        
        Tree.__init__(self, parent=parent, id=_id, 
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

    def validate_mergelist(self, merge_list):
        if (not isinstance(merge_list, np.ndarray)):
            return False
        if (merge_list.shape[1] != 2):
            return False

        s = set(merge_list.flatten())
        if ((min(s) != 0) or (max(s) != len(s)-1)):
            return False
        
        return True
