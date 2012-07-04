import glue
import glue.message as msg
from glue.viz_client import VizClient


class TreeClient(VizClient):
    """
    A client for visualizing the tree attributes of data sets
    """

    def __init__(self, data, layout=None):
        """ Create a new client

        Parameters:
        ----------
        layout: A TreeLayout object, to map the tree
                onto an xy coordinate system.
        """

        super(TreeClient, self).__init__(data)

        if (data.tree is None) or (data.tree.index_map is None):
            raise AttributeError("Input data does not have tree "
                                 "with an index_map")
        try:
            data.tree.index()
        except KeyError:
            raise TypeError("Cannot create a tree client with this data "
                            "-- tree cannot be indexed")

        self.layout = layout
        if not self.layout:
            self.layout = glue.TreeLayout(data.tree)

