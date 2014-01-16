import logging
from functools import partial

import numpy as np

from ..core.client import Client
from ..core.data import Data
from ..core.subset import RoiSubsetState
from ..core.roi import PolygonalROI
from ..core.util import relim, lookup_class
from ..core.edit_subset_mode import EditSubsetMode
from .viz_client import init_mpl
from .layer_artist import ScatterLayerArtist, LayerArtistContainer
from .util import visible_limits
from ..core.callback_property import (CallbackProperty, add_callback,
                                      delay_callback)
from scatter_client import ScatterClient


class BeeSwarmClient(ScatterClient):
    """
    A client class to visualize numerical-data grouped by categorical-data.
    Currently a shallow-subclassing of ScatterClient
    """

    pass