import logging
from functools import partial
from collections import defaultdict

import numpy as np

from ..core.client import Client
from ..core.data import Data
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

