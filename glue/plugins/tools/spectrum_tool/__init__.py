def setup():
    from glue.viewers.image.qt import ImageWidget
    from glue.plugins.tools.spectrum_tool.qt import SpectrumExtractorMode
    ImageWidget.tools.append('spectrum')
