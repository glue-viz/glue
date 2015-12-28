def setup():
    from .qt_widget import DendroWidget
    from ...config import qt_client
    qt_client.add(DendroWidget)
