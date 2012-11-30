import numpy as np

from ..core.client import Client
from ..core import message as msg
from ..core.data import Data
from ..core.subset import RangeSubsetState
from ..core.exceptions import IncompatibleDataException, IncompatibleAttribute
from ..core.edit_subset_mode import EditSubsetMode
from .layer_artist import HistogramLayerArtist, LayerArtistContainer
from .util import visible_limits
from ..core.callback_property import CallbackProperty, add_callback


class UpdateProperty(CallbackProperty):
    """Descriptor that calls client's sync_all() method when changed"""
    def __init__(self, default, relim=False):
        super(UpdateProperty, self).__init__(default)
        self.relim = relim

    def __set__(self, instance, value):
        changed = value != self.__get__(instance)
        super(UpdateProperty, self).__set__(instance, value)
        if not changed:
            return
        instance.sync_all()
        if self.relim:
            instance._relim()


def update_on_true(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if result:
            args[0].sync_all()
        return result
    return wrapper


class HistogramClient(Client):
    """
    A client class to display histograms
    """
    normed = UpdateProperty(False)
    cumulative = UpdateProperty(False)
    autoscale = UpdateProperty(True)
    nbins = UpdateProperty(30)
    xlog = UpdateProperty(False, relim=True)
    ylog = UpdateProperty(False)

    def __init__(self, data, figure, artist_container=None):
        super(HistogramClient, self).__init__(data)

        self._artists = artist_container or LayerArtistContainer()
        self._axes = figure.add_subplot(111)
        self._component = None

        self._xlim = {}

        try:
            self._axes.figure.set_tight_layout(True)
        except AttributeError:  # pragma: nocover (matplotlib < 1.1)
            pass

    @property
    def axes(self):
        return self._axes

    @property
    def xlimits(self):
        try:
            return self._xlim[self.component]
        except KeyError:
            pass

        lo, hi = self._default_limits()
        self._xlim[self.component] = lo, hi
        return lo, hi

    def _default_limits(self):
        if self.component is None:
            return 0, 1
        lo, hi = np.inf, -np.inf
        for a in self._artists:
            try:
                data = a.layer[self.component]
            except IncompatibleAttribute:
                continue

            if data.size == 0:
                continue
            lo = min(lo, np.nanmin(data))
            hi = max(hi, np.nanmax(data))
        return lo, hi

    @xlimits.setter
    @update_on_true
    def xlimits(self, value):
        lo, hi = value
        old = self.xlimits
        if lo is None:
            lo = old[0]
        if hi is None:
            hi = old[1]

        self._xlim[self.component] = min(lo, hi), max(lo, hi)
        self._relim()
        return True

    def layer_present(self, layer):
        return layer in self._artists

    def add_layer(self, layer):
        if layer.data not in self.data:
            raise IncompatibleDataException("Layer not in data collection")

        self._ensure_layer_data_present(layer)
        if self.layer_present(layer):
            return

        art = HistogramLayerArtist(layer, self._axes)
        self._artists.append(art)

        self._ensure_subsets_present(layer)
        self._sync_layer(layer)
        self._redraw()

    def _redraw(self):
        self._axes.figure.canvas.draw()

    def _ensure_layer_data_present(self, layer):
        if layer.data is layer:
            return
        if not self.layer_present(layer.data):
            self.add_layer(layer.data)

    def _ensure_subsets_present(self, layer):
        for subset in layer.data.subsets:
            self.add_layer(subset)

    @update_on_true
    def remove_layer(self, layer):
        if not self.layer_present(layer):
            return

        for a in self._artists.pop(layer):
            a.clear()

        if isinstance(layer, Data):
            for subset in layer.subsets:
                self.remove_layer(subset)

        return True

    @update_on_true
    def set_layer_visible(self, layer, state):
        if not self.layer_present(layer):
            return
        for a in self._artists[layer]:
            a.visible = state
        return True

    def is_layer_visible(self, layer):
        if not self.layer_present(layer):
            return False
        return any(a.visible for a in self._artists[layer])

    def _update_axis_labels(self):
        xlabel = self.component.label if self.component is not None else ''
        if self.xlog:
            xlabel = "Log %s" % xlabel
        ylabel = 'N'
        self._axes.set_xlabel(xlabel)
        self._axes.set_ylabel(ylabel)

    def _auto_nbin(self):
        data = set(a.layer.data for a in self._artists)
        if len(data) == 0:
            return
        dx = np.mean([d.size for d in data])
        self.nbins = min(max(5, (dx / 1000) ** (1. / 3.) * 30), 100)

    def _sync_layer(self, layer):
        for a in self._artists[layer]:
            a.lo, a.hi = self.xlimits
            a.nbins = self.nbins
            a.xlog = self.xlog
            a.ylog = self.ylog
            a.cumulative = self.cumulative
            a.normed = self.normed
            a.att = self._component
            a.update()

    def sync_all(self):
        layers = set(a.layer for a in self._artists)
        for l in layers:
            self._sync_layer(l)

        self._update_axis_labels()

        if self.autoscale:
            lim = visible_limits(self._artists, 1)
            if lim is not None:
                lo = 1e-5 if self.ylog else 0
                self._axes.set_ylim(lo, lim[1])

        yscl = 'log' if self.ylog else 'linear'
        self._axes.set_yscale(yscl)

        self._redraw()

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
        self._auto_nbin()
        self.sync_all()
        self._relim()

    def _relim(self):
        lim = self.xlimits
        if self.xlog:
            lim = list(np.log10(lim))
            if not np.isfinite(lim[0]):
                lim[0] = 1e-5
            if not np.isfinite(lim[1]):
                lim[1] = 1

        self._axes.set_xlim(lim)
        self._redraw()

    def _update_data(self, message):
        self.sync_all()

    def _update_subset(self, message):
        self._sync_layer(message.subset)
        self._redraw()

    def _add_subset(self, message):
        self.add_layer(message.sender)
        assert self.layer_present(message.sender)
        assert self.is_layer_visible(message.sender)

    def _remove_data(self, message):
        self.remove_layer(message.data)

    def _remove_subset(self, message):
        self.remove_layer(message.subset)

    def apply_roi(self, roi):
        x, y = roi.to_polygon()
        lo = min(x)
        hi = max(x)
        if self.xlog:
            lo = 10 ** lo
            hi = 10 ** hi

        state = RangeSubsetState(lo, hi)
        state.att = self.component
        mode = EditSubsetMode()
        visible = [d for d in self.data if self.is_layer_visible(d)]
        focus = visible[0] if len(visible) > 0 else None
        mode.update(self.data, state, focus_data=focus)

    def register_to_hub(self, hub):
        dfilter = lambda x: x.sender.data in self._artists
        dcfilter = lambda x: x.data in self._artists
        subfilter = lambda x: x.subset in self._artists

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
