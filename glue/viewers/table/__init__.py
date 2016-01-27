def setup():
    from glue.config import qt_client
    from .qt import TableWidget
    qt_client.add(TableWidget)
