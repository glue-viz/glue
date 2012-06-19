from .viz_client import VizClient
import .message as msg
from .exceptions import IncompatibleDataException

class HistogramLayerManager(object):
    def __init__(self, axes, layer):
        self._axes = axes
        self._layer = layer
        self._visible = True
        self._patchlist = None

    def is_visible(self):
        return self._visible

    def set_visible(self, state):
        self._visible = state

    def set_patches(self, patchlist):
        self.clear_patches()
        self._patchlist = patchlist

    def clear_patches(self):
        if self._patchlist is None:
            return

        for patch in self._patchlist:
            patch.remove()

        self._patchlist = None

    def __del__(self):
        self.clear_patches()


class HistogramClient(glue.Client):
    """
    A client class to display histograms
    """

    def __init__(self, data, axes):
        super(HistogramClient, self).__init__(data)

        self._managers = {}
        self._axes = axes
        self._component = None
        self._active_data = None
        self._options = {}

    def set_option(self, key, value):
        self._options[key] = value
        self.sync_all()

    def layer_present(self, layer):
        return layer in self._managers

    def add_layer(self, layer):
        if self.layer_present(layer):
            return
        if layer.data not in self.data:
            raise IncompatibleDataException("Layer not in data collection")

        self._ensure_layer_data_present(layer)

        manager = HistogramLayerManager(self._axes, layer)
        self._managers[layer] = manager

        self._ensure_subsets_present(layer)
        self.sync_all()

    def _ensure_layer_data_present(self, layer):
        if layer.data is layer:
            return
        if not self.layer_present(layer.data):
            self.add_layer(layer.data)

    def _ensure_subsets_present(self, layer):
        for subset in layer.data.subsets:
            self.add_layer(subset)

    def remove_layer(self, layer):
        if not self.layer_present(layer):
            return

        mgr = self._managers.pop(layer)
        del mgr

        if isinstance(layer, glue.Data):
            for subset in layer.subsets:
                self.remove_layer(subset)

    def draw_histogram(self):
        self._clear_patches()
        if self._active_data is None or self._component is None:
            return

        x = []
        colors = []
        managers = []

        if self.is_layer_visible(self._active_data):
            x.append(self._active_data[self._component].flat)
            colors.append(self._active_data.style.color)
            managers.append(self._managers[self._active_data])

        for subset in self._active_data.subsets:
            if not self.is_layer_visible(subset):
                continue
            pts = subset[self._component].flatten()
            if pts.size == 0:
                continue
            x.append(pts)
            colors.append(subset.style.color)
            managers.append(self._managers[subset])

        result = self._axes.hist(x, color = colors, **self._options)

        if len(x) == 1:
            patchlists = [result[2]]
        else:
            patchlists = result[2]

        for m,p in zip(managers, patchlists):
            m.set_patches(p)

    def set_layer_visible(self, layer, state):
        if not self.layer_present(layer):
            return
        self._managers[layer].set_visible(state)

    def is_layer_visible(self, layer):
        if not self.layer_present(layer):
            return
        return self._managers[layer].is_visible()

    def _clear_patches(self):
        for layer in self._managers:
            self._managers[layer].clear_patches()

    def sync_all(self):
        self.draw_histogram()

        if self._component is not None:
            self._axes.set_xlabel(self._component.label)
            self._axes.set_ylabel('N')

        self._axes.figure.canvas.draw()

    def set_data(self, data):
        if not self.layer_present(data):
            self.add_layer(data)
        self._active_data = data
        self.sync_all()
        self._relim()

    def set_component(self, component):
        """
        Redefine which component gets plotted

        Parameters
        ----------
        component: string
            The name of the new data component to plot
        """
        self._component = component
        self.sync_all()
        self._relim()

    def _relim(self):
        if self._active_data is None:
            return
        if self._component is None:
            return
        self._axes.set_xlim(min(self._active_data[self._component]),
                           max(self._active_data[self._component]))
        self._axes.figure.canvas.draw()


    def set_width(self, width):
        raise NotImplemented

    def _update_data(self, message):
        self.sync_all()

    def _update_subset(self, message):
        self.sync_all()

    def _add_subset(self, message):
        self.add_layer(message.sender)

    def _remove_data(self, message):
        self.remove_layer(message.data)

    def _remove_subset(self, message):
        self.remove_layer(message.subset)

    def register_to_hub(self, hub):
        dfilter = lambda x:x.sender.data in self._managers
        dcfilter = lambda x:x.data in self._managers
        subfilter = lambda x:x.subset in self._managers

        hub.subscribe(self,
                      msg.SubsetCreateMessage,
                      handler=self._add_subset,
                      filter=dfilter)
        hub.subscribe(self,
                      msg.SubsetUpdateMessage,
                      handler=self._update_subset,
                      filter=subfilter)
        hub.subscribe(self,
                      msg.SubsetDeleteMessage,
                      handler=self._remove_subset)
        hub.subscribe(self,
                      msg.DataUpdateMessage,
                      handler=self._update_data,
                      filter=dfilter)
        hub.subscribe(self,
                      msg.DataCollectionDeleteMessage,
                      handler=self._remove_data)
