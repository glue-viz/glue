from cloudviz.viz_client import VizClient


class ImageClient(VizClient):
    """
    A client class to display images
    """

    def __init__(self, data):

        VizClient.__init__(self, data)

        # The image data
        self._image = None

    def set_component(self, component):
        """
        Redefine which component gets plotted on the x axis

        Parameters
        ----------
        attribute: string
                 The name of the new data component to plot
        """
        self._image = self.data.components[component].data
        self.refresh()
