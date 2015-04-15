from __future__ import print_function, division
from abc import ABCMeta, abstractproperty, abstractmethod
from functools import partial
from pandas import DataFrame
import numpy as np

from ...external import six
from ...utils import coerce_numeric
from ...core.data import IncompatibleAttribute, ComponentID, Component
from ...core.callback_property import (CallbackProperty, add_callback)

from .util import get_colors
from ...clients.scatter_client import ScatterClient
from ...clients.layer_artist import (ChangedTrigger, ScatterLayerArtist)


@six.add_metaclass(ABCMeta)
class LineLayerBase(object):
    xatt = abstractproperty()
    yatt = abstractproperty()
    gatt = abstractproperty()

    @abstractmethod
    def get_data(self):
        pass


class LineLayerArtist(ScatterLayerArtist, LineLayerBase):

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
        if len(groups) < len(x):
            colors = get_colors(len(groups))
            for grp, c in zip(groups, colors):
                art = self._axes.plot(grp[1]['x'], grp[1]['y'],
                                      '.-', color=c)
                self.artists.extend(art)
        return True


class LineClient(ScatterClient):
    gatt = CallbackProperty()
    layer_artist_class = LineLayerArtist

    def _connect(self):
        add_callback(self, 'gatt', partial(self._set_xydata, 'g'))
        super(LineClient, self)._connect()

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

    def _set_xydata(self, coord, attribute, snap=True):
        if coord not in ('x', 'y', 'g'):
            raise TypeError("coord must be one of x, y, g")
        if not isinstance(attribute, ComponentID):
            raise TypeError("attribute must be a ComponentID")
        if coord == 'g':
            self.gatt = attribute
            self._gset = self.gatt is not None
            list(map(self._update_layer, self.artists.layers))
            self._update_axis_labels()
            self._pull_properties()
            self._redraw()
        else:
            super(LineClient, self)._set_xydata(coord, attribute, snap)

    def _update_layer(self, layer, force=False):
        """ Update both the style and data for the requested layer"""
        for art in self.artists[layer]:
            art.gatt = self.gatt
        super(LineClient, self)._update_layer(layer, force)

    def _on_component_replace(self, msg):
        old = msg.old
        new = msg.new
        if self.gatt is old:
            self.gatt = new
        super(LineClient, self)._on_component_replace(msg)


