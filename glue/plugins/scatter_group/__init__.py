def setup():
    from ...logger import logger
    try:
        from .qt_widget import ScatterGroupWidget
    except ImportError:
        logger.info("Could not load scatter group plugin")
    else:
        from ...config import qt_client
        qt_client.add(ScatterGroupWidget)
        logger.info("Loaded scatter group plugin")