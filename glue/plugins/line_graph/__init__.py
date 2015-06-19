def setup():
    from ...logger import logger
    try:
        from .qt_widget import LineWidget
    except ImportError:
        logger.info("Could not load line graph plugin")
    else:
        from ...config import qt_client
        qt_client.add(LineWidget)
        logger.info("Loaded line graph plugin")