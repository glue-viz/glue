def setup():
    from glue.viewers.image.qt import ImageWidget
    from glue.plugins.tools.pv_slicer.qt import PVSlicerMode
    ImageWidget.modes.append('Slice')
