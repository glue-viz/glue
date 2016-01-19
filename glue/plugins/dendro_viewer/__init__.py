def setup():
    from glue.config import qt_client
    from .qt_widget import DendroWidget
    from .data_factory import load_dendro
    qt_client.add(DendroWidget)
