def setup():
    try:
        from .qt_widget import GingaWidget
    except ImportError:
        raise ImportError("ginga is required")
    else:
        from ...config import qt_client
        qt_client.add(GingaWidget)
