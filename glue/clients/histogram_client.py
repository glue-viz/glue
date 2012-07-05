from .viz_client import VizClient


class HistogramClient(VizClient):
    """
    A client class to display histograms
    """

    def __init__(self, data, options=None):

        VizClient.__init__(self, data, options=options)
        self._component = None

    def set_component(self, component):
        """
        Redefine which component gets plotted

        Parameters
        ----------
        component: string
            The name of the new data component to plot
        """
        self._component = self.data.get_component(component)
        self._component = component
        self.refresh()
