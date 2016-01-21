def setup():
    from .qt_widget import DendroWidget
    from glue.config import qt_client
    qt_client.add(DendroWidget)
