def setup():
    from ...logger import logger
    try:
        from .qt_widget import GingaWidget
    except ImportError:
        raise Exception("ginga is required")
    else:
        from ...config import qt_client
        qt_client.add(GingaWidget)
