def setup():
    from glue.config import tool_registry
    from glue.viewers.image.qt import ImageWidget
    from glue.plugins.tools.spectrum_tool.qt import SpectrumTool
    tool_registry.add(SpectrumTool, widget_cls=ImageWidget)

