def setup():
    try:
        from . import link_helpers
    except ImportError:
        raise ImportError("Astropy >= 0.4 is required")
