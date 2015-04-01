from __future__ import print_function, division
from abc import ABCMeta, abstractproperty, abstractmethod
from functools import partial
from pandas import DataFrame
import numpy as np

from ...external import six
from ...utils import coerce_numeric, lookup_class
from ...core.data import IncompatibleAttribute, ComponentID, Component
from ...core.callback_property import (CallbackProperty, add_callback)

from .util import get_colors
from ...clients.scatter_client import ScatterClient
from ...clients.layer_artist import (ChangedTrigger, ScatterLayerArtist)


class ScatterGroupClient(ScatterClient):

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
    gatt = CallbackProperty()
    jitter = CallbackProperty()

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
        add_callback(self, 'yatt', partial(self._set_xydata, 'y'))
        add_callback(self, 'gatt', partial(self._set_xydata, 'g'))
        add_callback(self, 'jitter', self._jitter)
        self.axes.figure.canvas.mpl_connect('draw_event',
                                            lambda x: self._pull_properties())

    def groupable_attributes(self, layer, show_hidden=False):
        data = layer.data
        l = data._shape[0]
        if not data.find_component_id('None'):
            none_comp = Component(np.array(range(0, l)), units='None')
            data.add_component(none_comp, 'None', hidden=False)
        else:
            none_comp = data.find_component_id('None')
            to_comp = coerce_numeric(np.array(range(0, l)))
            to_comp.setflags(write=False)
            none_comp._data = to_comp

        comp = data.components if show_hidden else data.visible_components
        groups = [comp[-1]]
        for c in comp:
            if data.get_component(c).group:
                groups.append(c)
        return groups

    def add_layer(self, layer):
        """ Adds a new visual layer to a client, to display either a dataset
        or a subset. Updates both the client data structure and the
        plot.

        Returns the created layer artist

        :param layer: the layer to add
        :type layer: :class:`~glue.core.data.Data` or :class:`~glue.core.subset.Subset`
        """
        if layer.data not in self.data:
            raise TypeError("Layer not in data collection")
        if layer in self.artists:
            return self.artists[layer][0]

        result = ScatterGroupLayerArtist(layer, self.axes)
        self.artists.append(result)
        self._update_layer(layer)
        self._ensure_subsets_added(layer)
        return result

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

        if coord not in ('x', 'y', 'g'):
            raise TypeError("coord must be one of x, y, g")
        if not isinstance(attribute, ComponentID):
            raise TypeError("attribute must be a ComponentID")

        # update coordinates of data and subsets
        if coord == 'x':
            new_add = not self._xset
            self.xatt = attribute
            self._xset = self.xatt is not None
        elif coord == 'y':
            new_add = not self._yset
            self.yatt = attribute
            self._yset = self.yatt is not None
        elif coord == 'g':
             self.gatt = attribute
             self._gset = self.gatt is not None

        # update plots
        list(map(self._update_layer, self.artists.layers))

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

    def restore_layers(self, layers, context):
        """ Re-generate a list of plot layers from a glue-serialized list"""
        for l in layers:
            cls = lookup_class(l.pop('_type'))
            if cls != ScatterGroupLayerArtist:
                raise ValueError("Scatter client cannot restore layer of type "
                                 "%s" % cls)
            props = dict((k, context.object(v)) for k, v in l.items())
            layer = self.add_layer(props['layer'])
            layer.properties = props

    def _update_layer(self, layer, force=False):
        """ Update both the style and data for the requested layer"""
        if self.xatt is None or self.yatt is None:
            return

        if layer not in self.artists:
            return

        self._layer_updated = True
        for art in self.artists[layer]:
            art.xatt = self.xatt
            art.yatt = self.yatt
            art.gatt = self.gatt
            art.force_update() if force else art.update()
        self._redraw()

    def _on_component_replace(self, msg):
        old = msg.old
        new = msg.new

        if self.xatt is old:
            self.xatt = new
        if self.yatt is old:
            self.yatt = new
        if self.gatt is old:
            self.gatt = new


@six.add_metaclass(ABCMeta)
class ScatterGroupLayerBase(object):

    # which ComponentID to assign to X axis
    xatt = abstractproperty()

    # which ComponentID to assign to Y axis
    yatt = abstractproperty()

    # which ComponentID to group by
    gatt = abstractproperty()

    @abstractmethod
    def get_data(self):
        """
        Return the scatterpoint data as an (N, 2) array
        """
        pass

class ScatterGroupLayerArtist(ScatterLayerArtist, ScatterGroupLayerBase):

    xatt = ChangedTrigger()
    yatt = ChangedTrigger()
    gatt = ChangedTrigger()
    _property_set = ScatterLayerArtist._property_set + ['gatt']

    def _sync_style(self):
        style = self.layer.style
        for artist in self.artists:
            edgecolor = style.color
            mew = 3 if style.marker == '+' else 0.01
            artist.set_markeredgecolor(edgecolor)
            artist.set_markeredgewidth(mew)
            artist.set_markerfacecolor(style.color)
            artist.set_marker(style.marker)
            artist.set_markersize(style.markersize)
            artist.set_alpha(style.alpha)
            artist.set_zorder(self.zorder)
            artist.set_visible(self.visible and self.enabled)

    def _recalc(self):
        self.clear()
        assert len(self.artists) == 0

        try:
            x = self.layer[self.xatt].ravel()
            y = self.layer[self.yatt].ravel()
            g = self.layer[self.gatt].ravel()
        except IncompatibleAttribute as exc:
            self.disable_invalid_attributes(*exc.args)
            return False

        self.artists = self._axes.plot(x, y, '.')

        df = DataFrame({'g': g, 'x': x, 'y': y})
        groups = df.groupby('g')
        if int(len(groups)) < int(len(x)):
            colors = get_colors(len(groups))
            for grp, c in zip(groups, colors):
                art = self._axes.plot(grp[1]['x'], grp[1]['y'], '.-', color=c)
                self.artists.extend(art)
        return True

