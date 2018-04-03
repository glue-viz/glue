def setup():
    from glue.config import qt_client
    from .qt.data_viewer import ProfileViewer
    qt_client.add(ProfileViewer)
