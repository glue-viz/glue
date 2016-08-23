def setup():
    from glue.config import tool_registry
    from glue.viewers.image.qt import ImageWidget
    from glue.plugins.tools.pv_slicer.qt import PVSlicerTool
    #tool_registry.add(PVSlicerTool, widget_cls=ImageWidget)
