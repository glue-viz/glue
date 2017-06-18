def setup():
    from glue.viewers.image_new.qt import ImageViewer
    from glue.plugins.tools.spectrum_tool.qt import SpectrumExtractorMode  # noqa
    print(ImageViewer.tools)
    ImageViewer.tools.append('spectrum')
