try:
    from .client import *
    from .qt_widget import *
except ImportError:
    import warnings
    warnings.warn("Could not import ginga plugin, since ginga is required")
else:
    # Register qt client
    from ...config import qt_client
    qt_client.add(GingaWidget)
