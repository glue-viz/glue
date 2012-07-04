import numpy as np

from .tree import Tree


class TreeLayout(object):
    """ The TreeLayout class maps trees onto an xy coordinate space for
    plotting.

    TreeLayout provides a dictionary-like interface for access to the
    location of each node in a tree. The typical use looks something like:

    tl = TreeLayout(tree_object)
    x_location = tl[key].x
    y_location = t1[key].y
    width = t1[key].width
    height = t1[key].height

    where key is either a reference to one of the nodes in the tree,
    or the id of that node.

    In this base class, the layout assigns each node a width of 1. It
    places the root at (0,0). The y position of every other node is
    one higher than its parent, and the x location is such that
    subtrees are centered over the parent tree.

    Subclasses of TreeLayout can override the layout() method to
    provide alternative layout styles.
    """

    class Layout(object):
        """ A small class to hold the layout information for each
        tree node.

        Attributes:
        -----------
        node: Tree instance
              The node that this layout object describes
        x: X location of this node
        y: Y location of this node
        width: Width of this node
        height: Height of this node

        """
        def __init__(self, node, x=0., y=0., width=0., height=0.):
            self.x = x
            self.y = y
            self.width = width
            self.height = height
            self.node = node

    def __init__(self, tree):
        """ Create a new TreeLayout object

        Parameters:
        -----------
        Tree: Tree instance
              The root node of the tree to layout. The tree must be
              indexable (i.e. the call to tree.index() succeeds)

        """

        if not isinstance(tree, Tree):
            raise TypeError("Input not a tree object: %s" % type(tree))

        self.tree = tree
        self._dict = {}

        if not tree._index:
            try:
                tree.index()
            except KeyError:
                raise TypeError("Cannot create tree layout -- "
                                "input tree can't be indexed")
        self.layout()

    def __getitem__(self, key):
        return self._dict[key]

    def layout(self):
        """
        Calculate the layout of this tree.
        """
        self._tree_width(self.tree)
        self._tree_pos(self.tree)

    def _tree_width(self, tree):
        """
        Recursively calculates the width of each subtree. Also populates the
        layout dictionary.

        """
        node = TreeLayout.Layout(tree, x=0., y=0.,
                                 width=1., height=0.)
        self._dict[tree] = node
        self._dict[tree.id] = node

        width = 0.
        for c in tree.children:
            self._tree_width(c)
            width += self[c].width
            node.width = width

    def _tree_pos(self, tree):
        """
        Based on the width of each subtree, recursively moves the
        subtrees so they don't overlap.
        """
        w = 0.
        node = self[tree]
        for c in tree.children:
            self[c].x = node.x - node.width / 2. + w + self[c].width / 2.
            w += self[c].width
            self[c].y = node.y + 1
            self._tree_pos(c)

    def pick(self, x, y):
        """
        Based on the layout of the tree, choose a nearby branch to an
        x,y location

        Parameters:
        -----------
        x: The x coordinate to search from
        y: The y coordinate to search from

        Outputs:
        --------
        A reference to the closest tree node, if one is
        found. Otherwise, returns None

        """
        sz = len(self.tree._index)
        off = np.zeros(sz)
        candidate = np.zeros(sz, dtype=bool)

        for i, t in enumerate(self.tree._index):
            off[i] = abs(x - self[t].x)
            parent = self[t].node.parent
            if parent:
                candidate[i] = y < self[t].y and y > self[parent].y
            else:
                candidate[i] = y < self[t].y
        if not candidate.any():
            return None

        bad = np.where(~candidate)
        off[bad] = off.max()
        best = np.argmin(off)
        return self.tree._index[best]

    def tree_to_xy(self, tree):
        """
        Convert the locations of one or more (sub)trees into a list of
        x,y coordinates suitable for plotting.

        Parameters:
        -----------
        tree: Tree instance, or list of trees
              The (sub) tree(s) to generate xy coordinates for

        Outputs:
        --------
        A list of x and y values tracing the tree. If the input is a
        list of trees, then the xy list for each tree will be
        separated by None. This is convenient for plotting to
        matplotlib, since it will not draw lines between the different
        trees.

        """
        #code for when t is a list of trees
        try:
            for t in tree:
                x = []
                y = []
                xx, yy = self.tree_to_xy(t)
                x.extend(xx)
                y.extend(yy)
                x.append(None)
                y.append(None)
            return (x, y)
        except TypeError:  # tree is a scalar
            pass

        # code for when tree is a scalar
        x = [self[tree].x]
        y = [self[tree].y]
        for c in tree.children:
            xx, yy = self.tree_to_xy(c)
            x.extend([self[tree].x, xx[0]])
            y.extend([self[tree].y, self[tree].y])
            x += xx
            y += yy
            x.append(None)
            y.append(None)
        return (x, y)

    def branch_to_xy(self, branch):
        """
        Convert one or more single branches to a list of line segments
        for plotting.

        Parameters:
        -----------
        branch: Tree instance, or id of a tree, or a list of these
              The branch(es) to consider

        Outputs:
        --------
        A set of xy coordinates describing the branches

        """
        # code for when branch is a list of branches
        try:
            x = []
            y = []
            for b in branch:
                xx, yy = self.branch_to_xy(b)
                x.extend(xx)
                y.extend(yy)
                x.append(None)
                y.append(None)
            return (x, y)
        except TypeError:  # branch is a scalar
            pass

        #code for when branch is a scalar
        node = self[branch].node
        parent = node.parent
        if parent:
            x = [self[branch].x, self[branch].x, self[parent].x]
            y = [self[branch].y, self[parent].y, self[parent].y]
            return (x, y)
        else:
            return ([self[branch].x], [self[branch].y])


class DendrogramLayout(TreeLayout):

    def __init__(self, tree, data):
        self.data = data
        super(DendrogramLayout, self).__init__(tree)

    def layout(self):
        super(DendrogramLayout, self).layout()
        self.set_height()

    def set_height(self):

        nbranch = len(self.tree._index)
        nleaf = (nbranch + 1) / 2

        hival = self.data.max()
        for id in self.tree._index:
            self[id].y = hival

        for id in self.tree._index:
            hit = np.where(self.tree.index_map == id)
            assert(len(hit) > 0)

            if id < nleaf:
                self[id].y = self.data[hit].max()

            if len(hit) == 0:
                loval = 0
            else:
                loval = self.data[hit].min()
            parent = self[id].node.parent
            if not parent:
                continue
            self[parent].y = min(self[parent].y, loval)
