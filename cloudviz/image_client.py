import numpy as np

from cloudviz.viz_client import VizClient


class ImageClient(VizClient):
    """
    A client class to display images
    """

    def __init__(self, data):

        if len(data.shape) != 2:
            raise TypeError("Data must be 2 dimensional")

        VizClient.__init__(self, data)

        # The image data
        self._image = None
        self._component = None

        # Pixel coordinates for selection
        self._xdata = None
        self._ydata = None

    def set_component(self, component):
        """
        Redefine which component gets plotted on the x axis

        Parameters
        ----------
        component: string
            The name of the new data component to plot
        """
        self._image = self.data.components[component].data
        self._component = component
        x = np.arange(self._image.shape[1])
        y = np.arange(self._image.shape[0])
        self._xdata, self._ydata = np.meshgrid(x, y)
        self.refresh()
