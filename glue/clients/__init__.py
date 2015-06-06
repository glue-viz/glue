from matplotlib import rcParams, rcdefaults
# standardize mpl setup
rcdefaults()

from .histogram_client import HistogramClient
from .image_client import ImageClient
from .scatter_client import ScatterClient

from .viz_client import GenericMplClient
from .layer_artist import LayerArtist
