try:
    from .client import *
    from .qt_widget import *
except ImportError:
    import warnings
    warnings.warn("Could not import ginga plugin, since ginga is required")
