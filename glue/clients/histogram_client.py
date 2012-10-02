import numpy as np

from ..core.client import Client
from ..core import message as msg
from ..core.data import Data
from ..core.subset import RangeSubsetState
from ..core.exceptions import IncompatibleDataException, IncompatibleAttribute
from ..core.util import relim
from ..core.edit_subset_mode import EditSubsetMode


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

    def delete(self):
        self.clear_patches()

    def has_patches(self):
        return self._patchlist is not None


class HistogramClient(Client):
    """
    A client class to display histograms
    """

    def __init__(self, data, figure):
        super(HistogramClient, self).__init__(data)

        self._managers = {}
        self._axes = figure.add_subplot(111)
        self._component = None
        self._active_data = None
        self._options = {}
        self._ymin = None
        self._ymax = None
        self._autoscale = True
        try:
            self._axes.figure.set_tight_layout(True)
        except AttributeError:  # matplotlib < 1.1
            pass

    @property
    def axes(self):
        return self._axes

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
        mgr.delete()

        if isinstance(layer, Data):
            for subset in layer.subsets:
                self.remove_layer(subset)

        self.sync_all()

    def set_normalized(self, state):
        self.set_option('normed', state)
        self.sync_all()

    def set_autoscale(self, state):
        self._autoscale = state
        self.sync_all()

    def draw_histogram(self):
        self._clear_patches()
        if self._active_data is None or self._component is None:
            return

        x = []
        colors = []
        managers = []

        if self.is_layer_visible(self._active_data):
            try:
                x.append(self._active_data[self._component].flat)
            except IncompatibleAttribute:
                return
            colors.append(self._active_data.style.color)
            managers.append(self._managers[self._active_data])

        for subset in self._active_data.subsets:

            if not self.is_layer_visible(subset):
                continue
            try:
                pts = subset[self._component].flatten()
            except IncompatibleAttribute:
                pts = np.array([])

            if pts.size == 0:
                continue
            x.append(pts)
            colors.append(subset.style.color)
            managers.append(self._managers[subset])

        if len(x) >= 1:
            result = self._axes.hist(x, color=colors, **self._options)
            self._store_ylimits(result[0])

            if len(x) == 1:
                patchlists = [result[2]]
            else:
                patchlists = result[2]
        else:
            patchlists = []

        for m, p in zip(managers, patchlists):
            m.set_patches(p)

        if self._autoscale:
            self._snap_ylimits()

    def _store_ylimits(self, vals):
        if type(vals) != list:
            vals = [vals]
        self._ymin = min(v.min() for v in vals)
        self._ymax = max(v.max() for v in vals)

    def _snap_ylimits(self):
        if self._ymin is not None and self._ymax is not None:
            lo, hi = relim(self._ymin, self._ymax)
            self._axes.set_ylim(lo, hi)

    def set_layer_visible(self, layer, state):
        if not self.layer_present(layer):
            return
        self._managers[layer].set_visible(state)
        self.sync_all()

    def is_layer_visible(self, layer):
        if not self.layer_present(layer):
            return False
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

    def get_data(self):
        return self._active_data

    @property
    def component(self):
        return self._component

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
        try:
            data = self._active_data[self._component]
        except IncompatibleAttribute:
            return

        self._axes.set_xlim(data.min(), data.max())
        self._axes.figure.canvas.draw()

    def set_nbins(self, num):
        self.set_option('bins', num)

    def _update_data(self, message):
        self.sync_all()

    def _update_subset(self, message):
        self.sync_all()

    def _add_data(self, message):
        self.add_layer(message.data)
        assert self.layer_present(message.data)
        assert self.is_layer_visible(message.data)

    def _add_subset(self, message):
        self.add_layer(message.sender)
        assert self.layer_present(message.sender)
        assert self.is_layer_visible(message.sender)

    def _remove_data(self, message):
        self.remove_layer(message.data)

    def _remove_subset(self, message):
        self.remove_layer(message.subset)

    def _apply_roi(self, roi):
        x, y = roi.to_polygon()
        lo = min(x)
        hi = max(x)
        state = RangeSubsetState(lo, hi)
        state.att = self.component
        mode = EditSubsetMode()
        mode.combine(self.data, state)

    def register_to_hub(self, hub):
        dfilter = lambda x: x.sender.data in self._managers
        dcfilter = lambda x: x.data in self._managers
        subfilter = lambda x: x.subset in self._managers

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
        hub.subscribe(self,
                      msg.DataCollectionAddMessage,
                      handler=self._add_data,
                      filter=dcfilter)
