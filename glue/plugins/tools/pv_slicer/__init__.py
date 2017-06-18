def setup():
    from glue.viewers.image_new.qt import ImageViewer
    from glue.plugins.tools.pv_slicer.qt import PVSlicerMode  # noqa
    ImageViewer.tools.append('slice')
