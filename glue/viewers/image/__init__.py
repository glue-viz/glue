def setup():
    from glue.config import qt_client
    from .qt import ImageWidget
    qt_client.add(ImageWidget)
