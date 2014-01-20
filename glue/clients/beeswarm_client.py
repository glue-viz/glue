import logging
from functools import partial
from collections import defaultdict

import numpy as np

from ..core.client import Client
from ..core.data import Data, ComponentID
from ..core.subset import RoiSubsetState
from ..core.roi import PolygonalROI
from ..core.util import relim, lookup_class
from ..core.edit_subset_mode import EditSubsetMode
from .viz_client import init_mpl
from .layer_artist import LayerArtistContainer, BeeSwarmArtist
from .util import visible_limits
from ..core.callback_property import (CallbackProperty, add_callback,
                                      delay_callback)
from scatter_client import ScatterClient


class BeeSwarmClient(ScatterClient):
    """
    A client class to visualize numerical-data grouped by categorical-data.
    Currently a shallow-subclassing of ScatterClient
    """

    def __init__(self, data=None, figure=None, axes=None,
                 artist_container=None):
        """
        Create a new BeeSwarmClient object

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

        result = BeeSwarmArtist(layer, self.axes)
        self.artists.append(result)
        self._update_layer(layer)
        self._ensure_subsets_added(layer)
        return result

    def _update_categorical_labels(self):

        if self.xatt is None:
            return

        data = self.data[0]
        try:
            #sometimes xatt is ComponentID
            x_comp_id = data.find_component_id(self.xatt._label)
        except AttributeError:
            #sometimes its a string
            x_comp_id = data.find_component_id(self.xatt)
        categories = data.get_component(x_comp_id)._categories
        self.axes.set_xticks(np.arange(len(categories)))
        self.axes.set_xticklabels(categories)

    def _set_xydata(self, coord, attribute, snap=True):
        """ Redefine which components get assigned to the x/y axes

        :param coord: 'x' or 'y'
           Which axis to reassign
        :param attribute:
           Which attribute of the data to use.
        :type attribute: str
        :param snap:
           If True, will rescale x/y axes to fit the data
        :type snap: bool
        """

        if coord not in ('x', 'y'):
            raise TypeError("coord must be one of x,y")

        #update coordinates of data and subsets
        if coord == 'x':
            new_add = not self._xset
            self.xatt = attribute
            self._xset = self.xatt is not None
        elif coord == 'y':
            new_add = not self._yset
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
        self._update_categorical_labels()