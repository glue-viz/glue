def setup():
    from ...logger import logger
    try:
        from .qt_widget import GingaWidget
    except ImportError:
        logger.info("Could not load ginga viewer plugin, since ginga is required")
    else:
        from ...config import qt_client
        qt_client.add(GingaWidget)
        logger.info("Loaded ginga viewer plugin")
