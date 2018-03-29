def setup():
    from glue.config import qt_client
    from .qt.data_viewer import ProfileViewer
    print("HERE")
    qt_client.add(ProfileViewer)
