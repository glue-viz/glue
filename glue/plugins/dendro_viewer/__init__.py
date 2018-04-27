def setup():
    from glue.config import qt_client
    from .qt.data_viewer import DendrogramViewer
    from .data_factory import load_dendro  # noqa
    qt_client.add(DendrogramViewer)
