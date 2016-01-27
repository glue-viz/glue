def setup():
    from glue.config import qt_client
    from .qt import HistogramWidget
    qt_client.add(HistogramWidget)
