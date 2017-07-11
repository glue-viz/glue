def setup():
    from glue.viewers.image.qt import ImageViewer
    from glue.plugins.tools.spectrum_tool.qt import SpectrumExtractorMode  # noqa
    ImageViewer.tools.append('spectrum')
