from matplotlib import rcParams, rcdefaults

# standardize mpl setup
rcdefaults()

from ..external.qt import is_pyqt5
if is_pyqt5():
    rcParams['backend'] = 'Qt5Agg'
else:
    rcParams['backend'] = 'Qt4Agg'

from .histogram_client import HistogramClient
from .image_client import ImageClient
from .scatter_client import ScatterClient

from .viz_client import GenericMplClient
from .layer_artist import LayerArtist
