import cloudviz as cv
import cloudviz.message as msg
from cloudviz.viz_client import VizClient


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
            self.layout = cv.TreeLayout(data.tree)

    def register_to_hub(self, hub):
        """
        Override the default message handling to only receive messages
        from TreeSubset objects
        """

        filter = lambda x: x.sender.data is self.data and \
            isinstance(x.sender, cv.subset.TreeSubset)

        hub.subscribe_client(self,
                             msg.SubsetCreateMessage,
                             handler=self._add_subset,
                             filter=filter)

        hub.subscribe_client(self,
                             msg.SubsetUpdateMessage,
                             handler=self._update_subset,
                             filter=filter)

        hub.subscribe_client(self,
                             msg.SubsetDeleteMessage,
                             handler=self._remove_subset,
                             filter=filter)

        hub.subscribe_client(self,
                             msg.DataMessage,
                             handler=self._update_all,
                             filter=lambda x: x.sender is self.data)
