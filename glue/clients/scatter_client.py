import logging
from functools import partial

import numpy as np

from ..core.client import Client
from ..core.data import Data, IncompatibleAttribute, ComponentID, CategoricalComponent
from ..core.subset import RoiSubsetState
from ..core.roi import PolygonalROI
from ..core.util import relim, lookup_class
from ..core.edit_subset_mode import EditSubsetMode
from .viz_client import init_mpl
from .layer_artist import ScatterLayerArtist, LayerArtistContainer
from .util import visible_limits
from ..core.callback_property import (CallbackProperty, add_callback,
                                      delay_callback)

MAX_CATEGORIES = 10

class ScatterClient(Client):
    """
    A client class that uses matplotlib to visualize tables as scatter plots.
    """
    xmin = CallbackProperty(0)
    xmax = CallbackProperty(1)
    ymin = CallbackProperty(0)
    ymax = CallbackProperty(1)
    ylog = CallbackProperty(False)
    xlog = CallbackProperty(False)
    yflip = CallbackProperty(False)
    xflip = CallbackProperty(False)
    xatt = CallbackProperty()
    yatt = CallbackProperty()
    xcat = CallbackProperty()
    ycat = CallbackProperty()
    jitter = CallbackProperty()

    def __init__(self, data=None, figure=None, axes=None,
                 artist_container=None):
        """
        Create a new ScatterClient object

        :param data: :class:`~glue.core.DataCollection` to use

        :param figure:
           Which matplotlib figure instance to draw to. One will be created if
           not provided

        :param axes:
           Which matplotlib axes instance to use. Will be created if necessary
        """
        Client.__init__(self, data=data)
        figure, axes = init_mpl(figure, axes)
        self.artists = artist_container
        if self.artists is None:
            self.artists = LayerArtistContainer()

        self._layer_updated = False  # debugging
        self._xset = False
        self._yset = False
        self.axes = axes

        self._connect()
        self._set_limits()

    def is_layer_present(self, layer):
        """ True if layer is plotted """
        return layer in self.artists

    def get_layer_order(self, layer):
        """If layer exists as a single artist, return its zorder.
        Otherwise, return None"""
        artists = self.artists[layer]
        if len(artists) == 1:
            return artists[0].zorder
        else:
            return None

    @property
    def layer_count(self):
        return len(self.artists)

    def _connect(self):
        add_callback(self, 'xlog', self._set_xlog)
        add_callback(self, 'ylog', self._set_ylog)

        add_callback(self, 'xflip', self._set_limits)
        add_callback(self, 'yflip', self._set_limits)
        add_callback(self, 'xmin', self._set_limits)
        add_callback(self, 'xmax', self._set_limits)
        add_callback(self, 'ymin', self._set_limits)
        add_callback(self, 'ymax', self._set_limits)
        add_callback(self, 'xatt', partial(self._set_xydata, 'x'))
        add_callback(self, 'xatt', partial(self._update_ticks, 'x'))
        add_callback(self, 'yatt', partial(self._set_xydata, 'y'))
        add_callback(self, 'yatt', partial(self._update_ticks, 'y'))
        add_callback(self, 'xcat', partial(self._update_ticks, 'x'))
        add_callback(self, 'ycat', partial(self._update_ticks, 'y'))
        add_callback(self, 'jitter', self._jitter)
        self.axes.figure.canvas.mpl_connect('draw_event',
                                            lambda x: self._pull_properties())

    def _set_limits(self, *args):
        xlim = min(self.xmin, self.xmax), max(self.xmin, self.xmax)
        if self.xflip:
            xlim = xlim[::-1]
        ylim = min(self.ymin, self.ymax), max(self.ymin, self.ymax)
        if self.yflip:
            ylim = ylim[::-1]

        xold = self.axes.get_xlim()
        yold = self.axes.get_ylim()
        self.axes.set_xlim(xlim)
        self.axes.set_ylim(ylim)
        if xlim != xold or ylim != yold:
            self._redraw()

    def plottable_attributes(self, layer, show_hidden=False):
        data = layer.data
        comp = data.components if show_hidden else data.visible_components
        return [c for c in comp if
                data.get_component(c).numeric]

    def add_layer(self, layer):
        """ Adds a new visual layer to a client, to display either a dataset
        or a subset. Updates both the client data structure and the
        plot.

        Returns the created layer artist

        :param layer: the layer to add
        :type layer: :class:`~glue.core.Data` or :class:`~glue.core.Subset`
        """
        if layer.data not in self.data:
            raise TypeError("Layer not in data collection")
        if layer in self.artists:
            return self.artists[layer][0]

        result = ScatterLayerArtist(layer, self.axes)
        self.artists.append(result)
        self._update_layer(layer)
        self._ensure_subsets_added(layer)
        return result

    def _ensure_subsets_added(self, layer):
        if not isinstance(layer, Data):
            return
        for subset in layer.subsets:
            self.add_layer(subset)

    def _visible_limits(self, axis):
        """Return the min-max visible data boundaries for given axis"""
        return visible_limits(self.artists, axis)

    def _snap_xlim(self):
        """
        Reset the plotted x rng to show all the data
        """
        is_log = self.xlog
        rng = self._visible_limits(0)
        if rng is None:
            return
        rng = relim(rng[0], rng[1], is_log)
        if self.xflip:
            rng = rng[::-1]
        self.axes.set_xlim(rng)
        self._pull_properties()

    def _snap_ylim(self):
        """
        Reset the plotted y rng to show all the data
        """
        rng = [np.infty, -np.infty]
        is_log = self.ylog

        rng = self._visible_limits(1)
        if rng is None:
            return
        rng = relim(rng[0], rng[1], is_log)

        if self.yflip:
            rng = rng[::-1]
        self.axes.set_ylim(rng)
        self._pull_properties()

    def snap(self):
        """Rescale axes to fit the data"""
        self._snap_xlim()
        self._snap_ylim()
        self._redraw()

    def set_visible(self, layer, state):
        """ Toggle a layer's visibility

        :param layer: which layer to modify
        :type layer: class:`~glue.core.Data` or :class:`~glue.coret.Subset`

        :param state: True to show. false to hide
        :type state: boolean
        """
        if layer not in self.artists:
            return
        for a in self.artists[layer]:
            a.visible = state
        self._redraw()

    def is_visible(self, layer):
        if layer not in self.artists:
            return False
        return any(a.visible for a in self.artists[layer])

    def _set_xydata(self, coord, attribute, snap=True):
        """ Redefine which components get assigned to the x/y axes

        :param coord: 'x' or 'y'
           Which axis to reassign
        :param attribute:
           Which attribute of the data to use.
        :type attribute: core.data.ComponentID
        :param snap:
           If True, will rescale x/y axes to fit the data
        :type snap: bool
        """

        if coord not in ('x', 'y'):
            raise TypeError("coord must be one of x,y")
        if not isinstance(attribute, ComponentID):
            raise TypeError("attribute must be a ComponentID")

        #update coordinates of data and subsets
        if coord == 'x':
            new_add = not self._xset
            self.xatt = attribute
            self._xset = self.xatt is not None
        elif coord == 'y':
            new_add = not self._yset
            self.yatt = attribute
            self._yset = self.yatt is not None

        #update plots
        map(self._update_layer, self.artists.layers)

        if coord == 'x' and snap:
            self._snap_xlim()
            if new_add:
                self._snap_ylim()
        elif coord == 'y' and snap:
            self._snap_ylim()
            if new_add:
                self._snap_xlim()

        self._update_axis_labels()
        self._pull_properties()
        self._redraw()

    def apply_roi(self, roi):
        # every editable subset is updated
        # using specified ROI
        subset_state = RoiSubsetState()
        subset_state.xatt = self.xatt
        subset_state.yatt = self.yatt
        x, y = roi.to_polygon()
        subset_state.roi = PolygonalROI(x, y)
        mode = EditSubsetMode()
        visible = [d for d in self._data if self.is_visible(d)]
        focus = visible[0] if len(visible) > 0 else None
        mode.update(self._data, subset_state, focus_data=focus)

    def _set_xlog(self, state):
        """ Set the x axis scaling

        :param state:
            The new scaling for the x axis
        :type state: string ('log' or 'linear')
        """
        mode = 'log' if state else 'linear'
        lim = self.axes.get_xlim()
        self.axes.set_xscale(mode)

        #Rescale if switching to log with negative bounds
        if state and min(lim) <= 0:
            self._snap_xlim()

        self._redraw()

    def _set_ylog(self, state):
        """ Set the y axis scaling

        :param state: The new scaling for the y axis
        :type state: string ('log' or 'linear')
        """
        mode = 'log' if state else 'linear'
        lim = self.axes.get_ylim()
        self.axes.set_yscale(mode)
        #Rescale if switching to log with negative bounds
        if state and min(lim) <= 0:
            self._snap_ylim()

        self._redraw()

    def _remove_data(self, message):
        """Process DataCollectionDeleteMessage"""
        for s in message.data.subsets:
            self.delete_layer(s)
        self.delete_layer(message.data)

    def _remove_subset(self, message):
        self.delete_layer(message.subset)

    def delete_layer(self, layer):
        if layer not in self.artists:
            return
        self.artists.pop(layer)
        self._redraw()
        assert not self.is_layer_present(layer)

    def _update_data(self, message):
        data = message.sender
        self._update_layer(data)

    def _redraw(self):
        self.axes.figure.canvas.draw()

    def _update_ticks(self, *args):
        coord = args[0]

        if coord == 'x':
            if self.xcat:
                self._update_categorical_data(coord)
            else:
                rng = self.axes.get_xlim()
                self.axes.set_xticks(np.linspace(rng[0], rng[1], 5))
        elif coord == 'y':
            if self.ycat:
                self._update_categorical_data(coord)
            else:
                rng = self.axes.get_ylim()
                self.axes.set_yticks(np.linspace(rng[0], rng[1], 5))
        self._redraw()

    def _jitter(self, *args):

        for attribute in [self.xatt, self.yatt]:
            if attribute is not None:
                for data in self.data:
                    try:
                        comp = data.get_component(attribute)
                    except IncompatibleAttribute:
                        continue
                    try:
                        comp.jitter(method=self.jitter)
                    except NotImplementedError:
                        pass

    def _update_categorical_data(self, coord):
        attribute = self.xatt if coord == 'x' else self.yatt
        if attribute is None:
            return
        all_categories = set()
        for data in self._data:
            try:
                all_categories |= set(data.get_component(attribute)._categories)
            except IncompatibleAttribute:
                return
        categories = sorted(all_categories)
        for data in self._data:
            data.get_component(attribute)._update_categories(categories=categories)
            data.get_component(attribute).jitter(self.jitter)

        if len(categories) > MAX_CATEGORIES:
            ticks = np.floor(np.linspace(1, len(categories), MAX_CATEGORIES))
        else:
            ticks = np.arange(1, len(categories)+1)
        categories = np.asarray(categories, dtype=np.object)[ticks.astype(np.int)-1]
        if coord == 'x':
            self.axes.set_xticks(ticks)
            self.axes.set_xticklabels(categories, rotation=90)
        if coord == 'y':
            self.axes.set_yticks(ticks)
            self.axes.set_yticklabels(categories)

    def _update_axis_labels(self, *args):
        self.axes.set_xlabel(self.xatt)
        self.axes.set_ylabel(self.yatt)
        try:
            self.xcat = isinstance(self.data[0].get_component(self.xatt), CategoricalComponent)
        except IncompatibleAttribute:
            pass
        try:
            self.ycat = isinstance(self.data[0].get_component(self.yatt), CategoricalComponent)
        except IncompatibleAttribute:
            pass

    def _add_subset(self, message):
        subset = message.sender
        #only add subset if data layer present
        if subset.data not in self.artists:
            return
        subset.do_broadcast(False)
        self.add_layer(subset)
        subset.do_broadcast(True)

    def add_data(self, data):
        result = self.add_layer(data)
        for subset in data.subsets:
            self.add_layer(subset)
        return result

    @property
    def data(self):
        """The data objects in the scatter plot"""
        return list(self._data)

    def _update_subset(self, message):
        self._update_layer(message.sender)

    def restore_layers(self, layers, context):
        """ Re-generate a list of plot layers from a glue-serialized list"""
        for l in layers:
            cls = lookup_class(l.pop('_type'))
            if cls != ScatterLayerArtist:
                raise ValueError("Scatter client cannot restore layer of type "
                                 "%s" % cls)
            props = dict((k, context.object(v)) for k, v in l.items())
            layer = self.add_layer(props['layer'])
            layer.properties = props

    def _update_layer(self, layer):
        """ Update both the style and data for the requested layer"""
        if self.xatt is None or self.yatt is None:
            return

        if layer not in self.artists:
            return

        self._layer_updated = True
        for art in self.artists[layer]:
            art.xatt = self.xatt
            art.yatt = self.yatt
            art.update()
        self._redraw()

    def _pull_properties(self):
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        xsc = self.axes.get_xscale()
        ysc = self.axes.get_yscale()

        xflip = (xlim[1] < xlim[0])
        yflip = (ylim[1] < ylim[0])

        with delay_callback(self, 'xmin', 'xmax', 'xflip', 'xlog'):
            self.xmin = min(xlim)
            self.xmax = max(xlim)
            self.xflip = xflip
            self.xlog = (xsc == 'log')

        with delay_callback(self, 'ymin', 'ymax', 'yflip', 'ylog'):
            self.ymin = min(ylim)
            self.ymax = max(ylim)
            self.yflip = yflip
            self.ylog = (ysc == 'log')
