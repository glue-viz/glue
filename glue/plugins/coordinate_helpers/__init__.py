try:
    from .link_helpers import *
except ImportError:
    import warnings
    warnings.warn("Could not import coordinate_helpers plugin, since Astropy>=0.4 is required")