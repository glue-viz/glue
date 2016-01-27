def setup():
    from glue.config import qt_client
    from .qt import ScatterWidget
    qt_client.add(ScatterWidget)
