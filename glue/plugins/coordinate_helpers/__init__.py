def setup():
    from ...logger import logger
    try:
        from . import link_helpers
    except ImportError:
        import warnings
        logger.info("Could not import coordinate_helpers plugin, since Astropy>=0.4 is required")
    else:
        logger.info("Loaded coordinate helpers plugin")
