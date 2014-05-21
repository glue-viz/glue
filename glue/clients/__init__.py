from matplotlib import rcParams, rcdefaults
# standardize mpl setup
rcdefaults()
rcParams['axes.color_cycle'] = ['#377EB8', '#E41A1C', '#4DAF4A',
                                '#984EA3', '#FF7F00']
from .histogram_client import HistogramClient
from .image_client import ImageClient
from .scatter_client import ScatterClient

from .viz_client import GenericMplClient
from .layer_artist import LayerArtist
