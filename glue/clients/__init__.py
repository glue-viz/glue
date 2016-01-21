from matplotlib import rcParams, rcdefaults

# standardize mpl setup
rcdefaults()

from glue.external.qt import is_pyqt5
if is_pyqt5():
    rcParams['backend'] = 'Qt5Agg'
else:
    rcParams['backend'] = 'Qt4Agg'

# The following is a workaround for the fact that Matplotlib checks the
# rcParams at import time, not at run-time. I have opened an issue with
# Matplotlib here: https://github.com/matplotlib/matplotlib/issues/5513
from matplotlib import get_backend
from matplotlib import backends
backends.backend = get_backend()

from .histogram_client import HistogramClient
from .image_client import ImageClient
from .scatter_client import ScatterClient

from .viz_client import GenericMplClient
from .layer_artist import LayerArtist
